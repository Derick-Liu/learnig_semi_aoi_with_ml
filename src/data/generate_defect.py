import os
import cv2
import random
import numpy as np

from src.paths import GOLDEN_REFERENCE_PATH, TRAIN_DEFECT_DIR, TEST_DEFECT_DIR, MASK_DIR
from src.config import TRAIN_DEFECT_COUNT, TEST_DEFECT_COUNT, RANDOM_SEED, DEFECT_TYPES


def load_reference_image():
    # 讀取 golden reference 影像
    image = cv2.imread(str(GOLDEN_REFERENCE_PATH), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot find image: {GOLDEN_REFERENCE_PATH}")

    return image


def add_scratch(image, mask):
    # 加入 scratch 缺陷
    x1, y1 = random.randint(50, 450), random.randint(50, 450)
    x2, y2 = x1 + random.randint(-100, 100), y1 + random.randint(-100, 100)

    cv2.line(image, (x1, y1), (x2, y2), color=255, thickness=2)
    cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=2)

    return image, mask


def add_particle(image, mask):
    # 加入 particle 缺陷
    x, y = random.randint(50, 450), random.randint(50, 450)
    radius = random.randint(3, 8)

    cv2.circle(image, (x, y), radius, color=255, thickness=-1)
    cv2.circle(mask, (x, y), radius, color=255, thickness=-1)

    return image, mask


def add_open_circuit(image, mask):
    # 加入 open circuit 缺陷
    x, y = random.randint(100, 400), random.randint(100, 400)

    cv2.rectangle(image, (x, y), (x + 20, y + 5), color=0, thickness=-1)
    cv2.rectangle(mask, (x, y), (x + 20, y + 5), color=255, thickness=-1)

    return image, mask


def add_bridge(image, mask):
    # 加入 bridge 缺陷
    x1, y1 = random.randint(50, 450), random.randint(50, 450)
    x2, y2 = x1 + random.randint(-30, 30), y1 + random.randint(-30, 30)

    cv2.line(image, (x1, y1), (x2, y2), color=255, thickness=3)
    cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=3)

    return image, mask


def add_missing_pad(image, mask):
    # 加入 missing pad 缺陷
    x, y = random.randint(60, 400), random.randint(60, 400)

    cv2.rectangle(image, (x, y), (x + 20, y + 20), color=0, thickness=-1)
    cv2.rectangle(mask, (x, y), (x + 20, y + 20), color=255, thickness=-1)

    return image, mask


def add_stain(image, mask):
    # 加入 stain 缺陷
    x, y = random.randint(50, 450), random.randint(50, 450)
    radius = random.randint(10, 25)

    overlay = image.copy()
    cv2.circle(overlay, (x, y), radius, color=150, thickness=-1)

    image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
    cv2.circle(mask, (x, y), radius, color=255, thickness=-1)

    return image, mask


def apply_defect(image, defect_type):
    # 建立 defect mask
    mask = np.zeros_like(image)

    # 根據 defect type 套用缺陷
    if defect_type == "scratch":
        image, mask = add_scratch(image, mask)
    elif defect_type == "particle":
        image, mask = add_particle(image, mask)
    elif defect_type == "open_circuit":
        image, mask = add_open_circuit(image, mask)
    elif defect_type == "bridge":
        image, mask = add_bridge(image, mask)
    elif defect_type == "missing_pad":
        image, mask = add_missing_pad(image, mask)
    elif defect_type == "stain":
        image, mask = add_stain(image, mask)
    else:
        raise ValueError(f"Unknown defect type: {defect_type}")

    return image, mask


def generate_defect_image(base_image):
    # 複製基準影像
    image = base_image.copy()

    # 隨機選擇 defect type
    defect_type = random.choice(DEFECT_TYPES)

    # 套用 defect
    image, mask = apply_defect(image, defect_type)

    return image, mask, defect_type


def generate_images(output_dir, count, prefix, base_image):
    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    records = []

    for index in range(count):
        # 生成 defect image 與 mask
        image, mask, defect_type = generate_defect_image(base_image)

        # 設定檔名
        image_name = f"{prefix}_{index:03d}.png"
        mask_name = f"{prefix}_{index:03d}_mask.png"

        image_path = output_dir / image_name
        mask_path = MASK_DIR / mask_name

        # 儲存影像與 mask
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), mask)

        records.append({
            "image_name": image_name,
            "mask_name": mask_name,
            "split_prefix": prefix,
            "defect_type": defect_type
        })

    print(f"Generated {count} defect images in {output_dir}")

    return records


def main():
    # 固定隨機種子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 讀取基準影像
    base_image = load_reference_image()

    # 生成 train defect images
    train_records = generate_images(
        TRAIN_DEFECT_DIR,
        TRAIN_DEFECT_COUNT,
        "defect_train",
        base_image
    )

    # 生成 test defect images
    test_records = generate_images(
        TEST_DEFECT_DIR,
        TEST_DEFECT_COUNT,
        "defect_test",
        base_image
    )

    print(f"Train defect records: {len(train_records)}")
    print(f"Test defect records: {len(test_records)}")


if __name__ == "__main__":
    main()