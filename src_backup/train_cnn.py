import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

LABELS_PATH = "data/labels.csv"
OUTPUT_FIG_DIR = "outputs/figures"
OUTPUT_MODEL_DIR = "outputs/models"

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class AOIDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        # 讀取灰階影像
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)

        # 調整影像尺寸
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # 正規化到 0 到 1
        image = image.astype(np.float32) / 255.0

        # 加入 channel 維度
        image = np.expand_dims(image, axis=0)

        # 轉成 tensor
        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(row["label"], dtype=torch.long)

        return image_tensor, label_tensor


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 建立卷積特徵萃取層
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 建立分類層
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 1.讀取 labels.csv
df = pd.read_csv(LABELS_PATH)

# 2.分成 train 與 test
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

# 3.建立 Dataset 4.圖片前處理
train_dataset = AOIDataset(train_df)
test_dataset = AOIDataset(test_df)

# 5.建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 6.建立模型
model = SimpleCNN().to(device)

# 7.建立 loss function 與 optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
train_accuracies = []

# 8.訓練模型
for epoch in range(EPOCHS):
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

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}")

# 9.儲存模型
model_path = os.path.join(OUTPUT_MODEL_DIR, "cnn_aoi_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Saved model: {model_path}")

# 10.測試模型
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# 顯示分類報告
print(classification_report(all_labels, all_predictions, target_names=["normal", "defect"]))

# 儲存訓練曲線
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(train_accuracies, label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("CNN Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, "cnn_training_curve.png"), dpi=300)
plt.close()

print("Saved: outputs/figures/cnn_training_curve.png")

# 儲存 confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["normal", "defect"]
)

disp.plot(cmap="Blues")
plt.title("CNN Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_FIG_DIR, "cnn_confusion_matrix.png"), dpi=300)
plt.close()

print("Saved: outputs/figures/cnn_confusion_matrix.png")