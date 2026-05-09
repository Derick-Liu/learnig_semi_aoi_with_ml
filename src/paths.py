from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MASK_DIR = DATA_DIR / "masks"

TRAIN_NORMAL_DIR = TRAIN_DIR / "normal"
TRAIN_DEFECT_DIR = TRAIN_DIR / "defect"
TEST_NORMAL_DIR = TEST_DIR / "normal"
TEST_DEFECT_DIR = TEST_DIR / "defect"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

LABELS_PATH = DATA_DIR / "labels.csv"
GOLDEN_REFERENCE_PATH = RAW_DIR / "golden_reference.png"