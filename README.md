# Synthetic Semiconductor AOI Defect Detection (Python + Machine Learning)

本專案以 Python 為核心，模擬半導體 AOI（Automated Optical Inspection）檢測流程，透過合成影像資料（synthetic dataset）建立完整的 Machine Learning pipeline，包含資料生成、缺陷模擬、模型訓練與視覺化分析。

This project simulates a semiconductor AOI inspection workflow using synthetic image generation and machine learning models.

---

# Project Overview（專案概述）

本專案目標：

- 模擬半導體 AOI 缺陷檢測流程
- 建立可控的 synthetic dataset
- 訓練 Machine Learning 與 CNN 模型
- 視覺化模型結果與缺陷分析

核心流程：

Golden Reference Image
↓
Normal Dataset Generation
↓
Defect Dataset Generation (with masks)
↓
Baseline ML (Random Forest)
↓
CNN Model (PyTorch)
↓
Visualization & Analysis

---

# AOI Background（AOI 背景）

AOI（Automated Optical Inspection）常用於半導體製程中，用來檢測：

- 刮痕（scratch）
- 顆粒污染（particle）
- 線路斷裂（open circuit）
- 線路短路（bridge）
- 元件缺失（missing pattern）
- 污漬（stain）

In real-world semiconductor manufacturing, AOI systems detect defects that may impact yield and device functionality.

---

# Dataset Generation（資料生成）

本專案不使用真實晶圓資料，而是使用 Python 合成影像：

## 1️⃣ Golden Reference

- 建立無瑕疵標準影像
- 模擬規則線路與元件結構

## 2️⃣ Normal Dataset

- 加入 brightness / noise / blur / shift
- 模擬正常製程變動

## 3️⃣ Defect Dataset

- 隨機生成缺陷
- 同時產生 defect mask（位置標註）

---

# Defect Types（缺陷類型）

| 類型 | 說明 |
|---|---|
| scratch | 表面刮痕 |
| particle | 污染顆粒 |
| open_circuit | 線路斷裂 |
| bridge | 線路短路 |
| missing_pad | 元件缺失 |
| stain | 局部污染 |

---

# Dataset Visualization（資料視覺化）

## Sample Dataset

![Sample Dataset](outputs/figures/sample_dataset.png)

## Defect Mask Overlay

![Defect Overlay](outputs/figures/defect_overlay.png)

---

# Machine Learning Models

## 1️⃣ Baseline Model（Random Forest）

- 使用手工特徵（亮度、邊緣等）
- 建立基本分類能力

## 2️⃣ CNN Model（PyTorch）

- 直接從影像學習特徵
- 模擬真實 AOI 系統

---

# Results（結果分析）

## CNN Confusion Matrix

![CNN Confusion Matrix](outputs/figures/cnn_confusion_matrix.png)

模型結果：

- Accuracy: 95%
- Defect Recall: 100%
- 無漏檢（No False Negative）

👉 在 AOI 中，避免漏檢 defect 比誤判 normal 更重要

---

## Training Curve

![Training Curve](outputs/figures/cnn_training_curve.png)

---

## Prediction Examples

![Prediction Examples](outputs/figures/cnn_prediction_examples.png)

可視化模型預測結果，包含：

- 正確預測（Correct）
- 錯誤預測（Wrong）
- 信心值（Confidence）

---

# How to Run（如何執行）

## 1️⃣ 安裝環境

```bash
pip install -r requirements.txt
```

## 2️⃣ 生成資料

```bash
python src/generate_reference.py
python src/generate_normal_dataset.py
python src/generate_defect_dataset.py
python src/create_labels.py
```

## 3️⃣ 訓練模型

```bash
python src/train_baseline.py
python src/train_cnn.py
```

## 4️⃣ 視覺化結果

```bash
python src/visualize_dataset.py
python src/visualize_defect_overlay.py
python src/visualize_cnn_predictions.py
```

---

# Project Structure

```bash
learnig_semi_aoi_with_ml/
│
├── data/
│   ├── raw/
│   ├── train/
│   ├── test/
│   ├── masks/
│   └── labels.csv
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── src/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Future Improvements（未來改進）

- 增加更細微的 defect（hard cases）
- 加入 defect classification（多分類）
- 使用 YOLO / segmentation 做 defect localization
- 加入真實 AOI dataset（若可取得）
- 建立 Web UI（例如 Django 或 Streamlit）

---

# Learning Note（學習說明）

本專案為自學專案，過程中：

- 使用 Python 建立影像處理與 ML pipeline
- 結合 Computer Vision 與 Deep Learning
- 藉由 ChatGPT 協助理解 AOI、CNN 與實作流程
- 從零完成一個完整的 AI 專案

This project was developed as a self-learning project, with guidance and conceptual support from ChatGPT to understand AOI systems and machine learning workflows.

---

# Author

Derick Liu

- Background: Art&Design + TechArt
- Interest: Machine Learning, Computer Vision, Semiconductor AI Applications