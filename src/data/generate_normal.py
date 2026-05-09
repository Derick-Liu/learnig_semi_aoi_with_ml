import os
import cv2
import random
import numpy as np

from src.paths import GOLDEN_REFERENCE_PATH, TRAIN_NORMAL_DIR, TEST_NORMAL_DIR
from src.config import TRAIN_NORMAL_COUNT, TEST_NORMAL_COUNT, RANDOM_SEED
from src.utils.image_utils import add_noise, adjust_brightness, apply_blur, shift_image


def load_reference_image():
    # 讀取 golden reference 影像
    image = cv2.imread(str(GOLDEN_REFERENCE_PATH), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot find image: {GOLDEN_REFERENCE_PATH}")

    return image


def generate_normal_image(base_image):
    # 複製基準影像
    image = base_image.copy()

    # 套用正常製程變動
    image = adjust_brightness(image)
    image = add_noise(image)
    image = apply_blur(image)
    image = shift_image(image)

    return image


def generate_images(output_dir, count, prefix, base_image):
    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    for index in range(count):
        # 生成 normal image
        image = generate_normal_image(base_image)

        # 儲存影像
        filename = f"{prefix}_{index:03d}.png"
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), image)

    print(f"Generated {count} normal images in {output_dir}")


def main():
    # 固定隨機種子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 讀取基準影像
    base_image = load_reference_image()

    # 生成訓練與測試 normal images
    generate_images(
        TRAIN_NORMAL_DIR,
        TRAIN_NORMAL_COUNT,
        "normal_train",
        base_image
    )

    generate_images(
        TEST_NORMAL_DIR,
        TEST_NORMAL_COUNT,
        "normal_test",
        base_image
    )


if __name__ == "__main__":
    main()