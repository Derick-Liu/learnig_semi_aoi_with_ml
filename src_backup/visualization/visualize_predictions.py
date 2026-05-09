import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.paths import LABELS_PATH, FIGURE_DIR, MODEL_DIR
from src.data.aoi_dataset import AOIDataset
from src.models.cnn import SimpleCNN
from src.config import BATCH_SIZE


def load_model(device):
    # 載入 CNN 模型
    model = SimpleCNN().to(device)

    model_path = MODEL_DIR / "cnn_aoi_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    return model


def prepare_dataloader():
    # 讀取 labels.csv
    df = pd.read_csv(LABELS_PATH)

    # 只使用 test data
    test_df = df[df["split"] == "test"]

    # 建立 Dataset 與 DataLoader
    dataset = AOIDataset(test_df)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return loader, test_df


def run_inference(model, loader, device):
    # 執行預測
    results = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            for i in range(len(labels)):
                results.append({
                    "true_label": labels[i].item(),
                    "pred_label": predicted[i].item(),
                    "confidence": confidence[i].item()
                })

    return results


def attach_image_paths(results, df):
    # 將 image_path 補回結果
    df = df.reset_index(drop=True)

    for i in range(len(results)):
        results[i]["image_path"] = df.iloc[i]["image_path"]

    return results


def select_samples(results):
    # 分類結果
    wrong_results = [x for x in results if x["true_label"] != x["pred_label"]]
    correct_results = [x for x in results if x["true_label"] == x["pred_label"]]

    # 再細分
    wrong_normal = [x for x in wrong_results if x["true_label"] == 0]
    wrong_defect = [x for x in wrong_results if x["true_label"] == 1]

    correct_normal = [x for x in correct_results if x["true_label"] == 0]
    correct_defect = [x for x in correct_results if x["true_label"] == 1]

    # 組合樣本
    selected = (
        wrong_normal[:2] +
        wrong_defect[:2] +
        correct_normal[:2] +
        correct_defect[:2]
    )

    # 補滿 8 張
    if len(selected) < 8:
        remaining = [x for x in results if x not in selected]
        selected += remaining[:(8 - len(selected))]

    return selected[:8]


def plot_predictions(samples):
    # 建立圖表
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    class_names = ["normal", "defect"]

    for ax, item in zip(axes.flatten(), samples):
        image = cv2.imread(item["image_path"], cv2.IMREAD_GRAYSCALE)

        true_name = class_names[item["true_label"]]
        pred_name = class_names[item["pred_label"]]
        confidence = item["confidence"]

        status = "Correct" if true_name == pred_name else "Wrong"

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"{status}\nTrue: {true_name}\nPred: {pred_name}\nConf: {confidence:.2f}",
            fontsize=9
        )
        ax.axis("off")

    # 隱藏空格
    for ax in axes.flatten()[len(samples):]:
        ax.axis("off")

    return fig


def save_figure(fig):
    # 建立輸出資料夾
    os.makedirs(FIGURE_DIR, exist_ok=True)

    output_path = FIGURE_DIR / "cnn_prediction_examples.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    # 選擇裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 載入模型
    model = load_model(device)

    # 準備資料
    loader, df = prepare_dataloader()

    # 執行預測
    results = run_inference(model, loader, device)

    # 加上 image path
    results = attach_image_paths(results, df)

    # 選擇樣本
    samples = select_samples(results)

    # 畫圖
    fig = plot_predictions(samples)

    # 儲存
    save_figure(fig)


if __name__ == "__main__":
    main()