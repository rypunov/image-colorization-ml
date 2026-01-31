from PIL import Image # для работы с изображениями
import numpy as np # для работы с массивами

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

def rgb_to_gray(image, method='luma'):
    """
    Конвертирует RGB изображение в grayscale.
    Args:
        image: PIL.Image в режиме RGB
        method: 'average' | 'luma' | 'pil'
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
