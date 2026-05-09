import os
import cv2
import numpy as np

from src.paths import GOLDEN_REFERENCE_PATH
from src.config import REFERENCE_IMAGE_SIZE


def create_base_image(size):
    # 建立黑底影像
    image = np.zeros((size, size), dtype=np.uint8)
    return image


def draw_horizontal_lines(image):
    # 畫水平線路
    for y in range(50, 450, 50):
        cv2.line(image, (50, y), (450, y), color=200, thickness=2)

    return image


def draw_vertical_lines(image):
    # 畫垂直線路
    for x in range(50, 450, 50):
        cv2.line(image, (x, 50), (x, 450), color=200, thickness=2)

    return image


def draw_pads(image):
    # 畫方形 pads
    for i in range(5):
        for j in range(5):
            top_left = (60 + i * 80, 60 + j * 80)
            bottom_right = (top_left[0] + 20, top_left[1] + 20)

            cv2.rectangle(
                image,
                top_left,
                bottom_right,
                color=255,
                thickness=-1
            )

    return image


def apply_blur(image):
    # 套用 Gaussian blur 模擬真實影像
    return cv2.GaussianBlur(image, (3, 3), 0)


def save_image(image):
    # 建立資料夾
    os.makedirs(GOLDEN_REFERENCE_PATH.parent, exist_ok=True)

    # 儲存影像
    cv2.imwrite(str(GOLDEN_REFERENCE_PATH), image)

    print(f"Saved: {GOLDEN_REFERENCE_PATH}")


def main():
    # 建立基礎影像
    image = create_base_image(REFERENCE_IMAGE_SIZE)

    # 畫線路與結構
    image = draw_horizontal_lines(image)
    image = draw_vertical_lines(image)
    image = draw_pads(image)

    # 模擬影像模糊
    image = apply_blur(image)

    # 儲存影像
    save_image(image)


if __name__ == "__main__":
    main()