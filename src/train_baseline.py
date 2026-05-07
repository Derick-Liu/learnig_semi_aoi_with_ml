import os
import cv2
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

LABELS_PATH = "data/labels.csv"
OUTPUT_FIG_DIR = "outputs/figures"
OUTPUT_MODEL_DIR = "outputs/models"

os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# 從影像萃取簡單特徵
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    max_intensity = np.max(img)
    min_intensity = np.min(img)

    edges = cv2.Canny(img, 50, 150)
    edge_count = np.sum(edges > 0)

    bright_pixel_count = np.sum(img > 220)
    dark_pixel_count = np.sum(img < 30)

    return [
        mean_intensity,
        std_intensity,
        max_intensity,
        min_intensity,
        edge_count,
        bright_pixel_count,
        dark_pixel_count
    ]

# 讀取 labels.csv
df = pd.read_csv(LABELS_PATH)

# 分成 train / test
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

# 建立訓練資料
X_train = np.array([extract_features(path) for path in train_df["image_path"]])
y_train = train_df["label"].values

# 建立測試資料
X_test = np.array([extract_features(path) for path in test_df["image_path"]])
y_test = test_df["label"].values

# 訓練 Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=["normal", "defect"]))

# 儲存模型
model_path = os.path.join(OUTPUT_MODEL_DIR, "baseline_random_forest.joblib")
joblib.dump(model, model_path)
print(f"Saved model: {model_path}")

# 建立 confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["normal", "defect"]
)

disp.plot(cmap="Blues")
plt.title("Baseline Random Forest Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_FIG_DIR, "baseline_confusion_matrix.png"), dpi=300)
plt.close()

print("Saved: outputs/figures/baseline_confusion_matrix.png")