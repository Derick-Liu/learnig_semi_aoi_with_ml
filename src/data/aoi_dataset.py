import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from src.config import IMAGE_SIZE


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