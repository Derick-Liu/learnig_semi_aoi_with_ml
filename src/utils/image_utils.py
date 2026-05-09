import cv2
import numpy as np
import random


def add_noise(image, mean=0, std=10):
    # 加入 Gaussian noise
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def adjust_brightness(image, min_factor=0.8, max_factor=1.2):
    # 隨機調整亮度
    factor = random.uniform(min_factor, max_factor)
    adjusted_image = image * factor
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)


def apply_blur(image, probability=0.5):
    # 隨機套用 Gaussian blur
    if random.random() < probability:
        return cv2.GaussianBlur(image, (3, 3), 0)
    return image


def shift_image(image, max_shift=5):
    # 隨機平移影像
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)

    transform_matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    shifted_image = cv2.warpAffine(
        image,
        transform_matrix,
        (image.shape[1], image.shape[0])
    )

    return shifted_image