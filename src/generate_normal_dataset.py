import numpy as np
import cv2
import os
import random

# 路徑
INPUT_PATH = "data/raw/golden_reference.png"
TRAIN_DIR = "data/train/normal"
TEST_DIR = "data/test/normal"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# 讀取原圖
base_img = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)

# 資料增強函數
def add_noise(img):
    noise = np.random.normal(0, 10, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_brightness(img):
    factor = random.uniform(0.8, 1.2)
    bright = img * factor
    return np.clip(bright, 0, 255).astype(np.uint8)

def apply_blur(img):
    if random.random() < 0.5:
        return cv2.GaussianBlur(img, (3, 3), 0)
    return img

def shift_image(img):
    dx = random.randint(-5, 5)
    dy = random.randint(-5, 5)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted


# 生成 normal images
def generate_images(output_dir, count, prefix):
    for i in range(count):
        img = base_img.copy()

        img = adjust_brightness(img)
        img = add_noise(img)
        img = apply_blur(img)
        img = shift_image(img)

        filename = f"{prefix}_{i:03d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), img)

    print(f"Generated {count} images in {output_dir}")

# 執行
generate_images(TRAIN_DIR, 100, "normal_train")
generate_images(TEST_DIR, 20, "normal_test")