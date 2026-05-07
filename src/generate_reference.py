import numpy as np
import cv2
import os

# 建立輸出資料夾
os.makedirs("data/raw", exist_ok=True)

# 建立空白影像（黑底）
img_size = 512
img = np.zeros((img_size, img_size), dtype=np.uint8)

# 畫「水平線路」
for y in range(50, 450, 50):
    cv2.line(img, (50, y), (450, y), color=200, thickness=2)

# 畫「垂直線路」
for x in range(50, 450, 50):
    cv2.line(img, (x, 50), (x, 450), color=200, thickness=2)

# 畫「方形 pads」
for i in range(5):
    for j in range(5):
        top_left = (60 + i * 80, 60 + j * 80)
        bottom_right = (top_left[0] + 20, top_left[1] + 20)
        cv2.rectangle(img, top_left, bottom_right, color=255, thickness=-1)

# 加一點輕微 Gaussian blur（模擬真實影像）
img = cv2.GaussianBlur(img, (3, 3), 0)

output_path = "data/raw/golden_reference.png"
cv2.imwrite(output_path, img)

print(f"Saved: {output_path}")