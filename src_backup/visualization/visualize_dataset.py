import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from src.paths import LABELS_PATH, FIGURE_DIR


def load_labels():
    # 讀取 labels.csv
    df = pd.read_csv(LABELS_PATH)
    return df


def sample_images(df):
    # 抽取 normal 與 defect 範例
    normal_samples = df[df["label_name"] == "normal"].sample(4, random_state=42)
    defect_samples = df[df["label_name"] == "defect"].sample(4, random_state=42)

    samples = pd.concat([normal_samples, defect_samples])
    return samples


def plot_samples(samples):
    # 建立圖表
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for ax, (_, row) in zip(axes.flatten(), samples.iterrows()):
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)

        ax.imshow(image, cmap="gray")
        ax.set_title(row["label_name"])
        ax.axis("off")

    return fig


def save_figure(fig):
    # 建立輸出資料夾
    os.makedirs(FIGURE_DIR, exist_ok=True)

    output_path = FIGURE_DIR / "sample_dataset.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    # 讀取 labels
    df = load_labels()

    # 抽樣圖片
    samples = sample_images(df)

    # 畫圖
    fig = plot_samples(samples)

    # 儲存圖表
    save_figure(fig)


if __name__ == "__main__":
    main()