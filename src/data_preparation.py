from PIL import Image # для работы с изображениями
import numpy as np # для работы с массивами
from pathlib import Path # для удобной работы с путями файлов

# 1 способ преобразования цветного изображения в ч/б: gray = (R + G + B) / 3
def rgb_to_gray_average(color_img):
    """
    Конвертирует RGB изображение в grayscale методом усреднения.
    Args:
        color_img: PIL.Image в режиме RGB
    Returns:
        PIL.Image в режиме L (grayscale)
    Formula:
        gray = (R + G + B) / 3
    """
    color_array = np.array(color_img) # Преобразовываем изображение в массив numpy
    height, width, channels = color_array.shape
    gray_array = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            sum_rgb = 0.0
            for k in range(channels):
                sum_rgb += color_array[i, j, k]
            gray_array[i][j] = int(sum_rgb / 3.0)

    gray_image = Image.fromarray(gray_array) # Преобразовываем массив numpy в изображение
    return gray_image

# 2 способ преобразования цветного изображения в ч/б: через формулу яркости (gray = 0.299*R + 0.587*G + 0.114*B)
def rgb_to_gray_luma(color_img):
    """
    Конвертирует RGB изображение в grayscale через формулу яркости.
    Args:
        color_img: PIL.Image в режиме RGB
    Returns:
        PIL.Image в режиме L (grayscale)
    Formula:
        gray = 0.299*R + 0.587*G + 0.114*B
    """
    color_array = np.array(color_img).astype(np.float32) # Преобразовываем изображение в массив numpy
    height, width, channels = color_array.shape
    gray_array = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            gray_array[i][j] = (
                0.299 * color_array[i][j][0] +
                0.587 * color_array[i][j][1] +
                0.114 * color_array[i][j][2]
            )

    gray_array = np.clip(gray_array, 0, 255).astype(np.uint8)
    gray_image = Image.fromarray(gray_array) # Преобразовываем массив numpy в изображение
    return gray_image

# Функция преобразования цветного изображения в ч/б с выбором метода
def rgb_to_gray(image, method='luma'):
    """
    Конвертирует RGB изображение в grayscale.
    Args:
        image: PIL.Image в режиме RGB
        method: метод конвертации в ч/б ('luma', 'average', или 'pil')
    Returns:
        PIL.Image в режиме L (grayscale)
    """
    if method == 'average':
        return rgb_to_gray_average(image)
    elif method == 'luma':
        return rgb_to_gray_luma(image)
    elif method == 'pil':
        return image.convert('L') # Стандартный метод
    else:
        raise ValueError(f"Unknown method: {method}. Use 'average', 'luma' or 'pil'")

# Создает пары: цветное фото → ч/б фото
def create_color_gray_pairs(color_folder, output_folder, method='luma'):
    """
    Создает пары: цветное фото → ч/б фото
    Args:
        color_folder: папка с исходными цветными фото
        output_folder: куда сохранять пары (создаст подпапки color/ и gray/)
        method: метод конвертации в ч/б ('luma', 'average', или 'pil')
    """
    # Преобразовываем путь к папке из строки к Path
    color_path = Path(color_folder)
    output_path = Path(output_folder)

    # Создаем папки для сохранения изображений
    color_out = output_path / 'color'
    gray_out = output_path / 'gray'
    color_out.mkdir(parents=True, exist_ok=True)
    gray_out.mkdir(parents=True, exist_ok=True)

    # Поддерживаемые форматы
    photo_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    processed = 0  # Сколько обработали успешно
    failed = 0  # Сколько не получилось

    for file_path  in color_path.iterdir():
        # Пропускаем если это не файл
        if not file_path.is_file():
            continue

        # Пропускаем если неподходящее расширение
        if file_path.suffix.lower() not in photo_extensions:
            continue

        try:
            color_image = Image.open(file_path)
            if color_image.mode != 'RGB':
                color_image = color_image.convert('RGB')

            # Сохраняем цветное фото в папку color/
            color_save_path = color_out / f"{file_path.stem}.png"
            color_image.save(color_save_path, 'PNG')  # Сохраняем как PNG

            gray_image = rgb_to_gray(color_image, method)
            gray_save_path = gray_out / f"{file_path.stem}.png"
            gray_image.save(gray_save_path, 'PNG')

            processed += 1
            if processed % 10 == 0:
                print(f"Обработано: {processed} файлов")

        except Exception as e:
            failed += 1
            print(f"Ошибка с файлом {file_path.name}: {e}")

    print(f"\nГотово!")
    print(f"Успешно: {processed} пар")
    print(f"Не удалось: {failed} файлов")

if __name__ == "__main__":
    # Тест функции для создания пар (цветное фото → ч/б фото)
    create_color_gray_pairs(
        color_folder='../data/raw',
        output_folder='../data/processed',
        method='luma'
    )
