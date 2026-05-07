import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

LABELS_PATH = "data/labels.csv"
OUTPUT_DIR = "outputs/figures"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sample_dataset.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 讀取 labels.csv
df = pd.read_csv(LABELS_PATH)

# 取得 normal 與 defect 範例
normal_samples = df[df["label_name"] == "normal"].sample(4, random_state=42)
defect_samples = df[df["label_name"] == "defect"].sample(4, random_state=42)

samples = pd.concat([normal_samples, defect_samples])

# 建立圖表
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax, (_, row) in zip(axes.flatten(), samples.iterrows()):
    img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)

    ax.imshow(img, cmap="gray")
    ax.set_title(row["label_name"])
    ax.axis("off")

# 儲存圖表
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print(f"Saved: {OUTPUT_PATH}")