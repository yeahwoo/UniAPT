import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

def tsne_umap_visualize(csv_file):
    # 读取CSV
    df = pd.read_csv(csv_file)
    if "label" not in df.columns:
        raise ValueError("CSV中必须包含 'label' 列")

    # 提取特征和标签
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    # 标准化
    X = StandardScaler().fit_transform(X)

    # ===== t-SNE =====
    tsne = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate=200,
        n_iter=1000,
        init="pca",
        random_state=42,
        verbose=1
    )
    Z_tsne = tsne.fit_transform(X)

    # ===== UMAP =====
    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42
    )
    Z_umap = reducer.fit_transform(X)

    # ===== 可视化 =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["t-SNE Visualization", "UMAP Visualization"]
    embeddings = [Z_tsne, Z_umap]

    for ax, Z, title in zip(axes, embeddings, titles):
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], c="lightgray", label="label=0", s=20, alpha=0.7)
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], c="red", label="label=1", s=20, alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()

    # 保存图片（与csv同目录同名）
    out_path = os.path.splitext(csv_file)[0] + "_tsne_umap.png"
    plt.savefig(out_path, dpi=200)
    print(f"[信息] 可视化结果已保存到: {out_path}")
    plt.close()



if __name__ == "__main__":
    csv_file = "results/CICAPT/checkpoint-wget-80-20.csv"
    tsne_umap_visualize(csv_file)
