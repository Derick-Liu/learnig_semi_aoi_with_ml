import numpy as np
import cv2
import os
import random

INPUT_PATH = "data/raw/golden_reference.png"
TRAIN_DIR = "data/train/defect"
TEST_DIR = "data/test/defect"
MASK_DIR = "data/masks"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

base_img = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)

def add_scratch(img, mask):
    x1, y1 = random.randint(50, 450), random.randint(50, 450)
    x2, y2 = x1 + random.randint(-100, 100), y1 + random.randint(-100, 100)
    cv2.line(img, (x1, y1), (x2, y2), color=255, thickness=2)
    cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=2)
    return img, mask

def add_particle(img, mask):
    x, y = random.randint(50, 450), random.randint(50, 450)
    r = random.randint(3, 8)
    cv2.circle(img, (x, y), r, color=255, thickness=-1)
    cv2.circle(mask, (x, y), r, color=255, thickness=-1)
    return img, mask

def add_open_circuit(img, mask):
    x, y = random.randint(100, 400), random.randint(100, 400)
    cv2.rectangle(img, (x, y), (x + 20, y + 5), color=0, thickness=-1)
    cv2.rectangle(mask, (x, y), (x + 20, y + 5), color=255, thickness=-1)
    return img, mask

def add_bridge(img, mask):
    x1, y1 = random.randint(50, 450), random.randint(50, 450)
    x2, y2 = x1 + random.randint(-30, 30), y1 + random.randint(-30, 30)
    cv2.line(img, (x1, y1), (x2, y2), color=255, thickness=3)
    cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=3)
    return img, mask

def add_missing_pad(img, mask):
    x, y = random.randint(60, 400), random.randint(60, 400)
    cv2.rectangle(img, (x, y), (x + 20, y + 20), color=0, thickness=-1)
    cv2.rectangle(mask, (x, y), (x + 20, y + 20), color=255, thickness=-1)
    return img, mask

def add_stain(img, mask):
    x, y = random.randint(50, 450), random.randint(50, 450)
    r = random.randint(10, 25)
    overlay = img.copy()
    cv2.circle(overlay, (x, y), r, color=150, thickness=-1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    cv2.circle(mask, (x, y), r, color=255, thickness=-1)
    return img, mask

def generate_defect(img):
    mask = np.zeros_like(img)

    defect_type = random.choice([
        "scratch",
        "particle",
        "open_circuit",
        "bridge",
        "missing_pad",
        "stain"
    ])

    if defect_type == "scratch":
        img, mask = add_scratch(img, mask)
    elif defect_type == "particle":
        img, mask = add_particle(img, mask)
    elif defect_type == "open_circuit":
        img, mask = add_open_circuit(img, mask)
    elif defect_type == "bridge":
        img, mask = add_bridge(img, mask)
    elif defect_type == "missing_pad":
        img, mask = add_missing_pad(img, mask)
    elif defect_type == "stain":
        img, mask = add_stain(img, mask)

    return img, mask, defect_type

def generate_images(output_dir, count, prefix):
    records = []

    for i in range(count):
        img = base_img.copy()

        img, mask, defect_type = generate_defect(img)

        img_name = f"{prefix}_{i:03d}.png"
        mask_name = f"{prefix}_{i:03d}_mask.png"

        cv2.imwrite(os.path.join(output_dir, img_name), img)
        cv2.imwrite(os.path.join(MASK_DIR, mask_name), mask)

        records.append((img_name, defect_type))

    print(f"Generated {count} defect images in {output_dir}")
    return records

train_records = generate_images(TRAIN_DIR, 100, "defect_train")
test_records = generate_images(TEST_DIR, 20, "defect_test")