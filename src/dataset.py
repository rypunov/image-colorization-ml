import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from skimage.color import rgb2lab
import numpy as np


class ColorizationDataset(Dataset):
    """
    Обновленный датасет: использует пространство Lab и нормализацию [-1, 1]
    """

    def __init__(self, root_dir, transform=None):
        self.data_dir = Path(root_dir)
        # Мы используем только папку color, так как канал L (ч/б) получим из нее
        self.color_dir = self.data_dir / 'color'

        self.files = sorted([file.name for file in self.color_dir.iterdir() if file.is_file()])

        if transform is None:
            # Оставляем только базовый Resize, ToTensor здесь не нужен в обычном виде
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        color_path = self.color_dir / filename

        # Загружаем цветное изображение
        img = Image.open(color_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        img_array = np.array(img)

        # Конвертация в Lab: вход [0, 255] -> выход Lab
        # skimage автоматически нормализует вход, если он в uint8
        lab_img = rgb2lab(img_array)

        # Разделяем каналы и нормализуем в диапазон [-1, 1]
        # Канал L (яркость) исходно 0..100
        l_chan = lab_img[:, :, 0]
        l_norm = (l_chan / 50.0) - 1.0

        # Каналы ab (цвета) исходно -128..127
        ab_chan = lab_img[:, :, 1:]
        ab_norm = ab_chan / 128.0

        # Превращаем в тензоры
        l_tensor = torch.from_numpy(l_norm).unsqueeze(0).float()  # [1, 128, 128]
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float()  # [2, 128, 128]

        return l_tensor, ab_tensor


def get_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=2):
    dataset = ColorizationDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    try:
        dataloader = get_dataloader('../data/processed/cifar10', batch_size=4)
        l_batch, ab_batch = next(iter(dataloader))
        print(f"✅ DataLoader работает")
        print(f"   Батч L (вход): {l_batch.shape} (min: {l_batch.min():.2f}, max: {l_batch.max():.2f})")
        print(f"   Батч ab (цель): {ab_batch.shape} (min: {ab_batch.min():.2f}, max: {ab_batch.max():.2f})")
    except Exception as e:
        print(f"❌ Ошибка: {e}")