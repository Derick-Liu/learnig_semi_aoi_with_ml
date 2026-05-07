import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset, DataLoader

LABELS_PATH = "data/labels.csv"
MODEL_PATH = "outputs/models/cnn_aoi_model.pth"
OUTPUT_DIR = "outputs/figures"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cnn_prediction_examples.png")

IMAGE_SIZE = 128
BATCH_SIZE = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # 正規化到 0 到 1
        image_normalized = image_resized.astype(np.float32) / 255.0

        # 加入 channel 維度
        image_tensor = np.expand_dims(image_normalized, axis=0)

        # 轉成 tensor
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
        label_tensor = torch.tensor(row["label"], dtype=torch.long)

        return image_tensor, label_tensor, row["image_path"]


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
        # 萃取影像特徵
        x = self.features(x)

        # 輸出分類結果
        x = self.classifier(x)

        return x


# 讀取 labels.csv
df = pd.read_csv(LABELS_PATH)

# 只使用 test data
test_df = df[df["split"] == "test"]

# 建立 Dataset 與 DataLoader
test_dataset = AOIDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 載入 CNN 模型
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

class_names = ["normal", "defect"]

results = []

# 執行預測
with torch.no_grad():
    for images, labels, image_paths in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        for i in range(len(image_paths)):
            results.append({
                "image_path": image_paths[i],
                "true_label": labels[i].item(),
                "pred_label": predicted[i].item(),
                "confidence": confidence[i].item()
            })

# 優先挑出錯誤預測，再補正確預測
# 分類結果
wrong_results = [item for item in results if item["true_label"] != item["pred_label"]]
correct_results = [item for item in results if item["true_label"] == item["pred_label"]]

# 再細分
wrong_normal = [x for x in wrong_results if x["true_label"] == 0]
wrong_defect = [x for x in wrong_results if x["true_label"] == 1]

correct_normal = [x for x in correct_results if x["true_label"] == 0]
correct_defect = [x for x in correct_results if x["true_label"] == 1]

# 組合（每種抓幾張）
selected_results = (
    wrong_normal[:2] +
    wrong_defect[:2] +
    correct_normal[:2] +
    correct_defect[:2]
)

# 如果不夠 8 張就補
if len(selected_results) < 8:
    remaining = [x for x in results if x not in selected_results]
    selected_results += remaining[:(8 - len(selected_results))]

selected_results = selected_results[:8]

# 讓錯誤案例優先顯示
selected_results = sorted(
    selected_results,
    key=lambda x: x["true_label"] == x["pred_label"]
)

# 建立預測範例圖
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for ax, item in zip(axes.flatten(), selected_results):
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

# 隱藏沒有用到的 subplot
for ax in axes.flatten()[len(selected_results):]:
    ax.axis("off")

# 儲存圖表
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print(f"Saved: {OUTPUT_PATH}")
print(f"Wrong predictions: {len(wrong_results)}")
print(f"Correct predictions: {len(correct_results)}")