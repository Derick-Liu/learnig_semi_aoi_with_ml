import os
import pandas as pd

from src.paths import (
    TRAIN_NORMAL_DIR,
    TRAIN_DEFECT_DIR,
    TEST_NORMAL_DIR,
    TEST_DEFECT_DIR,
    LABELS_PATH
)


def collect_normal_records(folder, split):
    # 收集 normal 圖片資料
    records = []

    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            records.append({
                "image_path": str(folder / filename),
                "split": split,
                "label": 0,
                "label_name": "normal",
                "defect_type": "none"
            })

    return records


def collect_defect_records(folder, split):
    # 收集 defect 圖片資料
    records = []

    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            records.append({
                "image_path": str(folder / filename),
                "split": split,
                "label": 1,
                "label_name": "defect",
                "defect_type": "unknown"
            })

    return records


def create_labels():
    # 建立 labels records
    records = []

    records.extend(collect_normal_records(TRAIN_NORMAL_DIR, "train"))
    records.extend(collect_normal_records(TEST_NORMAL_DIR, "test"))
    records.extend(collect_defect_records(TRAIN_DEFECT_DIR, "train"))
    records.extend(collect_defect_records(TEST_DEFECT_DIR, "test"))

    # 轉成 DataFrame
    labels_df = pd.DataFrame(records)

    return labels_df


def main():
    # 建立 labels.csv
    labels_df = create_labels()

    # 儲存 labels.csv
    labels_df.to_csv(LABELS_PATH, index=False)

    print(f"Saved: {LABELS_PATH}")
    print(labels_df.head())
    print(f"Total images: {len(labels_df)}")


if __name__ == "__main__":
    main()