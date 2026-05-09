import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE
from src.paths import LABELS_PATH, FIGURE_DIR, MODEL_DIR
from src.data.aoi_dataset import AOIDataset
from src.models.cnn import SimpleCNN


def prepare_dataloaders():
    # 讀取 labels.csv
    df = pd.read_csv(LABELS_PATH)

    # 分成 train 與 test
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # 建立 Dataset
    train_dataset = AOIDataset(train_df)
    test_dataset = AOIDataset(test_df)

    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    # 切換成訓練模式
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一輪梯度
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(images)

        # 計算 loss
        loss = criterion(outputs, labels)

        # 反向傳播
        loss.backward()

        # 更新模型參數
        optimizer.step()

        total_loss += loss.item()

        # 計算訓練準確率
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def evaluate_model(model, test_loader, device):
    # 切換成測試模式
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 執行預測
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return all_labels, all_predictions


def save_training_curve(train_losses, train_accuracies):
    # 繪製訓練曲線
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("CNN Training Curve")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURE_DIR / "cnn_training_curve.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_confusion_matrix(all_labels, all_predictions):
    # 建立 confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["normal", "defect"]
    )

    disp.plot(cmap="Blues")
    plt.title("CNN Confusion Matrix")

    output_path = FIGURE_DIR / "cnn_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    # 建立輸出資料夾
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 選擇運算裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 準備資料
    train_loader, test_loader = prepare_dataloaders()

    # 建立模型
    model = SimpleCNN().to(device)

    # 建立 loss function 與 optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_losses = []
    train_accuracies = []

    # 訓練模型
    for epoch in range(EPOCHS):
        epoch_loss, epoch_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.4f}"
        )

    # 儲存模型
    model_path = MODEL_DIR / "cnn_aoi_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")

    # 測試模型
    all_labels, all_predictions = evaluate_model(
        model,
        test_loader,
        device
    )

    # 顯示分類報告
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["normal", "defect"]
        )
    )

    # 儲存圖表
    save_training_curve(train_losses, train_accuracies)
    save_confusion_matrix(all_labels, all_predictions)


if __name__ == "__main__":
    main()