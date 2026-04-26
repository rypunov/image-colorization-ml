import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms


class ColorizationDataset(Dataset):
    """
    Датасет для колоризации: загружает пары ч/б и цветных изображений
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: папка, в которой лежат color/ и gray/
            transform: преобразования для изображений
        """
        self.data_dir = Path(root_dir)
        self.color_dir = self.data_dir / 'color'
        self.gray_dir = self.data_dir / 'gray'

        # Получаем список всех файлов
        self.files = sorted([file.name for file in self.color_dir.iterdir() if file.is_file()])

        # Преобразования
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # [0, 255] → [0, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Имя файла
        filename = self.files[idx]

        # Загружаем цветное и ч/б
        color_path = self.color_dir / filename
        gray_path = self.gray_dir / filename

        color_img = Image.open(color_path).convert('RGB')  # TODO Возможно, стоит убрать convert
        gray_img = Image.open(gray_path).convert('L')  # TODO Возможно, стоит убрать convert

        # Применяем преобразования
        color_tensor = self.transform(color_img)
        gray_tensor = self.transform(gray_img)

        return gray_tensor, color_tensor


def get_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=2):
    """
    Создает DataLoader для обучения
    """
    dataset = ColorizationDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    # Тест загрузчика
    dataloader = get_dataloader('../data/processed/cifar10', batch_size=4)
    gray_batch, color_batch = next(iter(dataloader)) # Берем первый батч

    print(f"✅ DataLoader работает")
    print(f"   Батч ч/б: {gray_batch.shape}")
    print(f"   Батч цветных: {color_batch.shape}")