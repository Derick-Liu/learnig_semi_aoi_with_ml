import cv2
import numpy as np


def extract_features(image_path):
    # 讀取灰階影像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 計算亮度特徵
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    max_intensity = np.max(image)
    min_intensity = np.min(image)

    # 計算邊緣特徵
    edges = cv2.Canny(image, 50, 150)
    edge_count = np.sum(edges > 0)

    # 計算亮暗像素數量
    bright_pixel_count = np.sum(image > 220)
    dark_pixel_count = np.sum(image < 30)

    return [
        mean_intensity,
        std_intensity,
        max_intensity,
        min_intensity,
        edge_count,
        bright_pixel_count,
        dark_pixel_count
    ]