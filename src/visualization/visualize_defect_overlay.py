import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from src.paths import LABELS_PATH, FIGURE_DIR, MASK_DIR


def load_labels():
    # 讀取 labels.csv
    df = pd.read_csv(LABELS_PATH)
    return df


def sample_defect_images(df):
    # 抽取 defect 範例
    defect_samples = df[df["label_name"] == "defect"].sample(4, random_state=7)
    return defect_samples


def get_mask_path(image_path):
    # 取得對應 mask 路徑
    image_name = os.path.basename(image_path)
    mask_name = image_name.replace(".png", "_mask.png")
    mask_path = MASK_DIR / mask_name

    return mask_path


def create_overlay(image, mask):
    # 建立紅色 mask overlay
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    overlay = image_rgb.copy()

    overlay[mask > 0] = [255, 0, 0]

    blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

    return blended


def plot_overlay_samples(defect_samples):
    # 建立圖表
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for col, (_, row) in enumerate(defect_samples.iterrows()):
        image_path = row["image_path"]
        mask_path = get_mask_path(image_path)

        # 讀取 defect image 與 mask
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # 建立 overlay
        blended = create_overlay(image, mask)

        # 顯示原始 defect image
        axes[0, col].imshow(image, cmap="gray")
        axes[0, col].set_title("Defect Image")
        axes[0, col].axis("off")

        # 顯示 mask overlay
        axes[1, col].imshow(blended)
        axes[1, col].set_title("Mask Overlay")
        axes[1, col].axis("off")

    return fig


def save_figure(fig):
    # 建立輸出資料夾
    os.makedirs(FIGURE_DIR, exist_ok=True)

    output_path = FIGURE_DIR / "defect_overlay.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    # 讀取 labels
    df = load_labels()

    # 抽樣 defect 圖片
    defect_samples = sample_defect_images(df)

    # 畫 defect overlay
    fig = plot_overlay_samples(defect_samples)

    # 儲存圖表
    save_figure(fig)


if __name__ == "__main__":
    main()