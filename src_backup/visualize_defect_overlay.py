import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

LABELS_PATH = "data/labels.csv"
MASK_DIR = "data/masks"
OUTPUT_DIR = "outputs/figures"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "defect_overlay.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 讀取 labels.csv
df = pd.read_csv(LABELS_PATH)

# 取得 defect 範例
defect_samples = df[df["label_name"] == "defect"].sample(4, random_state=7)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for col, (_, row) in enumerate(defect_samples.iterrows()):
    img_path = row["image_path"]
    img_name = os.path.basename(img_path)
    mask_name = img_name.replace(".png", "_mask.png")
    mask_path = os.path.join(MASK_DIR, mask_name)

    # 讀取 defect image 與 mask
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 建立紅色 overlay
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    overlay[mask > 0] = [255, 0, 0]

    blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

    # 顯示原始 defect image
    axes[0, col].imshow(img, cmap="gray")
    axes[0, col].set_title("Defect Image")
    axes[0, col].axis("off")

    # 顯示 mask overlay
    axes[1, col].imshow(blended)
    axes[1, col].set_title("Mask Overlay")
    axes[1, col].axis("off")

# 儲存圖表
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print(f"Saved: {OUTPUT_PATH}")