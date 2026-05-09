import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

from src.paths import LABELS_PATH, FIGURE_DIR, MODEL_DIR
from src.models.baseline import extract_features


def prepare_features():
    # 讀取 labels.csv
    df = pd.read_csv(LABELS_PATH)

    # 分成 train 與 test
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # 建立訓練特徵
    X_train = np.array([
        extract_features(path)
        for path in train_df["image_path"]
    ])

    y_train = train_df["label"].values

    # 建立測試特徵
    X_test = np.array([
        extract_features(path)
        for path in test_df["image_path"]
    ])

    y_test = test_df["label"].values

    return X_train, y_train, X_test, y_test


def train_random_forest(X_train, y_train):
    # 建立 Random Forest 模型
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # 訓練模型
    model.fit(X_train, y_train)

    return model


def save_confusion_matrix(y_test, y_pred):
    # 建立 confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["normal", "defect"]
    )

    disp.plot(cmap="Blues")
    plt.title("Baseline Random Forest Confusion Matrix")

    output_path = FIGURE_DIR / "baseline_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    # 建立輸出資料夾
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 準備特徵資料
    X_train, y_train, X_test, y_test = prepare_features()

    # 訓練模型
    model = train_random_forest(X_train, y_train)

    # 預測
    y_pred = model.predict(X_test)

    # 評估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["normal", "defect"]
        )
    )

    # 儲存模型
    model_path = MODEL_DIR / "baseline_random_forest.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")

    # 儲存圖表
    save_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    main()