import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Двойной сверточный слой: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    DownBlock: MaxPool (2x2) -> DoubleConv
    Уменьшает размер картинки в 2 раза
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    UpBlock: Upsample (транспонированная свертка) -> Concatenate (skip) -> DoubleConv
    Увеличивает размер картинки в 2 раза
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        # Транспонированная свертка для апсемпла
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Двойной сверточный слой
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 - с нижнего уровня (декодер)
        # x2 - skip connection (энкодер)

        x1 = self.up(x1)

        # Обрезаем x2 если нужно (на случай нечетных размеров)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        # Склеиваем по каналам
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    """
    Выходной сверточный слой: Conv 1x1 -> нужное количество каналов
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Полная архитектура U-Net
    """
    def __init__(self, in_channels=1, out_channels=3, features=[64, 128, 256, 512]):
        """
        Args:
            in_channels: входные каналы (1 для ч/б)
            out_channels: выходные каналы (3 для RGB)
            features: количество каналов на каждом уровне
        """
        super(UNet, self).__init__()

        self.encoder1 = DoubleConv(in_channels, features[0])
        self.encoder2 = Down(features[0], features[1])
        self.encoder3 = Down(features[1], features[2])
        self.encoder4 = Down(features[2], features[3])

        # Бутылочное горлышко (самый сжатый слой)
        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        # Декодер
        self.decoder4 = Up(features[3] * 2, features[3])
        self.decoder3 = Up(features[3], features[2])
        self.decoder2 = Up(features[2], features[1])
        self.decoder1 = Up(features[1], features[0])

        # Выходной слой
        self.out_conv = OutConv(features[0], out_channels)

    def forward(self, x):
        # Энкодер (с сохранением skip-connection)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Бутылочное горлышко
        bottleneck = self.bottleneck(enc4)

        # Декодер (с передачей skip)
        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        # Выход
        return self.out_conv(dec1)


def test_model():
    """
    Тестовая функция для проверки архитектуры
    """
    # Создаем модель
    model = UNet(in_channels=1, out_channels=3)

    # Создаем случайный вход (batch=4, channels=1, height=32, width=32)
    x = torch.randn(4, 1, 32, 32)

    # Прогоняем через модель
    y = model(x)

    print("✅ Модель создана")
    print(f"   Вход: {x.shape}")
    print(f"   Выход: {y.shape}")
    print(f"   Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Проверка на разных размерах
    print("\n🔍 Проверка на других размерах:")
    for size in [32, 64, 128]:
        x_test = torch.randn(2, 1, size, size)
        y_test = model(x_test)
        print(f"   {size}×{size}: вход {x_test.shape} → выход {y_test.shape}")


if __name__ == "__main__":
    test_model()