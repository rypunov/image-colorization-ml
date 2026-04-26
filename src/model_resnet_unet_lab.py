import torch
import torch.nn as nn
from torchvision import models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class=2):  # Теперь предсказываем 2 канала (a и b)
        super().__init__()

        # Загружаем предобученный ResNet18
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.base_model.children())

        # Энкодер
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # 64 канала
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # 64 канала
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # 128 каналов
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # 256 каналов
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # 512 каналов
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        # Декодер
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(512 + 256, 256, 3, 1)
        self.conv_up2 = convrelu(256 + 128, 128, 3, 1)
        self.conv_up1 = convrelu(128 + 64, 64, 3, 1)
        self.conv_up0 = convrelu(64 + 64, 64, 3, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, n_class, kernel_size=1),
            nn.Tanh()  # Выход в диапазоне [-1, 1] для каналов a и b
        )

    def forward(self, input):
        # input: это канал L (1 канал).
        # ResNet обучен на 3-х канальных изображениях, поэтому дублируем L.
        x = torch.cat([input, input, input], dim=1)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

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

        x = self.upsample(x)
        return self.conv_last(x)


def test_resunet():
    # Указываем n_class=2 для Lab
    model = ResNetUNet(n_class=2)

    # Вход теперь строго 1 канал (L)
    x = torch.randn(4, 1, 128, 128)
    y = model(x)

    print("--- Тестирование ResNetUNet (Lab) ---")
    print(f"Вход (L-канал): {x.shape}")
    print(f"Выход (ab-каналы): {y.shape}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Проверка диапазона после Tanh
    print(f"Мин/Макс выхода: {y.min().item():.2f} / {y.max().item():.2f}")

    assert y.shape == (4, 2, 128, 128), "Ошибка в размерности выхода!"
    print("Тест пройден успешно!")


if __name__ == "__main__":
    test_resunet()