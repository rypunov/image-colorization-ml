import torch
import torch.nn as nn
from torchvision import models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    """
    U-Net с энкодером на базе предобученного ResNet18
    Вход: 1 канал (ч/б) → выход: 3 канала (RGB)
    """

    def __init__(self, n_class=3):
        super().__init__()

        # Загружаем предобученный ResNet18
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.base_model.children())

        # Энкодер (копируем слои из ResNet)
        # layer0: первые слои (conv1, bn, relu, maxpool) → 64 канала, размер /4
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, H/2, W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)

        # layer1: 64 канала, размер /4
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, H/4, W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)

        # layer2: 128 каналов, размер /8
        self.layer2 = self.base_layers[5]  # size=(N, 128, H/8, W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)

        # layer3: 256 каналов, размер /16
        self.layer3 = self.base_layers[6]  # size=(N, 256, H/16, W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)

        # layer4: 512 каналов, размер /32
        self.layer4 = self.base_layers[7]  # size=(N, 512, H/32, W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        # Апсемплинг (билинейная интерполяция)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Декодер (расшифровка + склеивание с skip connections)
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        # Обработка оригинального размера
        self.conv_original_size0 = convrelu(1, 64, 3, 1)  # ← вход 1 канал (ч/б)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        # Выходной слой: 64 канала → 3 (RGB)
        self.conv_last = nn.Conv2d(64, n_class, 1)

        # Активация Tanh на выходе (значения в [-1, 1])
        self.tanh = nn.Tanh()

    def forward(self, input):
        # Входное изображение (ч/б) → 1 канал → 64 канала для skip connection
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        # Энкодер (ResNet) — последовательное сжатие

        # Преобразуем 1 канал → 3 канала (повторяем ч/б 3 раза)
        input_3ch = input.repeat(1, 3, 1, 1)  # [B,1,H,W] → [B,3,H,W]

        layer0 = self.layer0(input_3ch)  # skip connection 1
        layer1 = self.layer1(layer0)  # skip connection 2
        layer2 = self.layer2(layer1)  # skip connection 3
        layer3 = self.layer3(layer2)  # skip connection 4
        layer4 = self.layer4(layer3)  # самый сжатый слой

        # Декодер — расшифровка с использованием skip connections
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        # Финальный апсемплинг и склейка с оригиналом
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        # Выход: 3 канала (RGB) + Tanh
        out = self.conv_last(x)
        return self.tanh(out)


def test_resunet():
    """Тестовая функция"""
    model = ResNetUNet(n_class=3)
    x = torch.randn(4, 1, 128, 128)  # батч 4, 1 канал (ч/б), 128×128
    y = model(x)
    print(f"Вход: {x.shape}")
    print(f"Выход: {y.shape}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_resunet()