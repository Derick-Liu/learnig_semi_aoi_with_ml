import os
import pandas as pd

TRAIN_NORMAL_DIR = "data/train/normal"
TRAIN_DEFECT_DIR = "data/train/defect"
TEST_NORMAL_DIR = "data/test/normal"
TEST_DEFECT_DIR = "data/test/defect"

OUTPUT_PATH = "data/labels.csv"

records = []

# 加入 normal 資料
for split, folder in [("train", TRAIN_NORMAL_DIR), ("test", TEST_NORMAL_DIR)]:
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            records.append({
                "image_path": os.path.join(folder, filename),
                "split": split,
                "label": 0,
                "label_name": "normal",
                "defect_type": "none"
            })

# 加入 defect 資料
for split, folder in [("train", TRAIN_DEFECT_DIR), ("test", TEST_DEFECT_DIR)]:
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            records.append({
                "image_path": os.path.join(folder, filename),
                "split": split,
                "label": 1,
                "label_name": "defect",
                "defect_type": "unknown"
            })

# 儲存 labels.csv
df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved: {OUTPUT_PATH}")
print(df.head())
print(f"Total images: {len(df)}")