import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Literal, Optional
from PIL import Image
import os


def plot_auc_f1(csv_path, save_path=None, dpi=300):
    """
    读取包含列(model, metric, mean, std)的CSV文件，绘制四个模型在AUC与F1上的分组柱状图。
    - AUC为实心柱，F1为同色+阴影纹理
    - 使用std绘制误差棒
    - y轴范围包含误差棒
    - 图例统一用灰色底，通过实心/斜纹区分 AUC 与 F1
    """
    df = pd.read_csv(csv_path)
    sub = df[df["metric"].isin(["AUC", "F1"])].copy()

    model_order = [
        "No Time Feature",
        "No Node Mask",
        "No Edge Reconstruction",
        "Full Model",
    ]
    exist_models = [m for m in model_order if m in sub["model"].unique()]
    if len(exist_models) != 4:
        exist_models = list(sub["model"].unique())

    def pick(metric, field):
        out = []
        for m in exist_models:
            s = sub[(sub.model == m) & (sub.metric == metric)]
            if s.empty:
                raise ValueError(f"缺少 {m} 的 {metric} 数据")
            out.append(float(s[field].values[0]))
        return out

    auc_vals = pick("AUC", "mean")
    f1_vals = pick("F1", "mean")
    auc_std = pick("AUC", "std")
    f1_std = pick("F1", "std")

    # 原配色保持柱子区分度
    colors = ["#82B0D2", "#BEB8DC", "#E7DAD2", "#999999"][: len(exist_models)]

    x = np.arange(len(exist_models))
    width = 0.32
    err_kw = dict(elinewidth=1.0, ecolor="black", capsize=3, capthick=1.0)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    # AUC 实心柱
    bars_auc = ax.bar(
        x - width / 2,
        auc_vals,
        yerr=auc_std,
        error_kw=err_kw,
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.8,
        label="AUC",
    )
    # F1 斜纹柱
    bars_f1 = ax.bar(
        x + width / 2,
        f1_vals,
        yerr=f1_std,
        error_kw=err_kw,
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        hatch="////",
        alpha=0.8,
        label="F1",
    )

    # y轴范围考虑误差棒
    all_low = np.concatenate(
        [np.array(auc_vals) - np.array(auc_std), np.array(f1_vals) - np.array(f1_std)]
    )
    all_high = np.concatenate(
        [np.array(auc_vals) + np.array(auc_std), np.array(f1_vals) + np.array(f1_std)]
    )
    vmin, vmax = float(all_low.min()), float(all_high.max())
    margin = (vmax - vmin) * 0.15
    ax.set_ylim(vmin - margin, vmax + margin)

    # y 轴格式
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))

    ax.set_xticks(x, exist_models, rotation=0)
    ax.set_ylabel("Score")
    ax.set_title("AUC & F1 across Models", fontweight="bold", fontsize=14, pad=18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # 图例：统一灰色，靠实心/斜纹区分
    handles = [
        Patch(facecolor="lightgray", edgecolor="black", linewidth=0.6, label="AUC"),
        Patch(
            facecolor="lightgray",
            edgecolor="black",
            linewidth=0.6,
            hatch="////",
            label="F1",
        ),
    ]
    leg = ax.legend(handles=handles, frameon=True)
    leg.get_frame().set_alpha(0.9)

    # 柱顶标注
    def annotate(bars):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    annotate(bars_auc)
    annotate(bars_f1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_precision_recall_boxplot(csv_path, save_path=None, dpi=300, n_samples=200):
    """
    从CSV文件读取Precision和Recall的均值和标准差，生成近似分布数据绘制横向箱型图。
    - 左子图: Precision
    - 右子图: Recall
    - 模型名字在纵轴，显示完全
    - 颜色与plot_auc_f1保持一致，透明度更低(alpha=0.6)
    """
    df = pd.read_csv(csv_path)
    sub = df[df["metric"].isin(["PRECISION", "RECALL"])].copy()

    # 模型顺序
    model_order = [
        "No Time Feature",
        "No Node Mask",
        "No Edge Reconstruction",
        "Full Model",
    ]
    exist_models = [m for m in model_order if m in sub["model"].unique()]
    if len(exist_models) != 4:
        exist_models = list(sub["model"].unique())

    # 提取Precision和Recall的均值和std
    precision_stats = [
        (
            sub[(sub.model == m) & (sub.metric == "PRECISION")]["mean"].values[0],
            sub[(sub.model == m) & (sub.metric == "PRECISION")]["std"].values[0],
        )
        for m in exist_models
    ]
    recall_stats = [
        (
            sub[(sub.model == m) & (sub.metric == "RECALL")]["mean"].values[0],
            sub[(sub.model == m) & (sub.metric == "RECALL")]["std"].values[0],
        )
        for m in exist_models
    ]

    # 生成模拟数据
    precision_data = [
        np.random.normal(mu, sigma, n_samples) for mu, sigma in precision_stats
    ]
    recall_data = [np.random.normal(mu, sigma, n_samples) for mu, sigma in recall_stats]

    # 颜色（莫兰迪低饱和度）
    colors = ["#82B0D2", "#BEB8DC", "#E7DAD2", "#999999"]
    colors = colors[: len(exist_models)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    # Precision 子图 (横向)
    bplot1 = axes[0].boxplot(
        precision_data, vert=False, patch_artist=True, labels=exist_models
    )
    for patch, color in zip(bplot1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_title("Precision")
    axes[0].set_xlabel("Score")
    axes[0].grid(axis="x", linestyle="--", alpha=0.35)

    # Recall 子图 (横向)
    bplot2 = axes[1].boxplot(
        recall_data, vert=False, patch_artist=True, labels=exist_models
    )
    for patch, color in zip(bplot2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_title("Recall")
    axes[1].set_xlabel("Score")
    axes[1].grid(axis="x", linestyle="--", alpha=0.35)

    # 给长模型名留空间
    plt.subplots_adjust(left=0.35)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_confusion_matrices(csv_path, save_path=None, dpi=300, normalize=False):
    """
    从CSV读取四个模型的 TN/FN/TP/FP (mean)，绘制 2x2 混淆矩阵子图。
    - 行=Actual(−/+)，列=Predicted(−/+)
    - 右侧色条更靠右；标题更大、更靠上，且与子图留更大空白
    """
    df = pd.read_csv(csv_path)

    # 模型顺序（若不完整则退回出现顺序）
    model_order = [
        "No Time Feature",
        "No Node Mask",
        "No Edge Reconstruction",
        "Full Model",
    ]
    exist_models = [m for m in model_order if m in df["model"].unique()]
    if len(exist_models) != 4:
        exist_models = list(df["model"].unique())[:4]

    def get_mean(m, metric):
        sel = df[(df.model == m) & (df.metric == metric)]
        if sel.empty:
            raise ValueError(f"找不到 {m} 的 {metric}")
        return float(sel["mean"].values[0])

    mats = []
    for m in exist_models:
        tn = get_mean(m, "TN")
        fn = get_mean(m, "FN")
        tp = get_mean(m, "TP")
        fp = get_mean(m, "FP")
        # 行=Actual(-,+)，列=Predicted(-,+)
        mat = np.array([[tn, fp], [fn, tp]], dtype=float)
        if normalize:
            row_sum = mat.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            mat = mat / row_sum
        mats.append(mat)

    vmin = 0.0
    vmax = max(mat.max() for mat in mats)

    # 子图稍小，间隔更大；右侧留更宽空间给色条
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    axes = axes.ravel()

    cmap = plt.cm.Blues
    ims = []
    for ax, mat, title in zip(axes, mats, exist_models):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
        ims.append(im)
        ax.set_title(title, fontsize=11, pad=6)

        ax.set_xticks([0, 1], labels=["Predicted −", "Predicted +"])
        ax.set_yticks([0, 1], labels=["Actual −", "Actual +"])

        # 清晰边界
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_edgecolor("#444")

        # 数值标注
        for i in range(2):
            for j in range(2):
                v = mat[i, j]
                txt = (
                    f"{v:.2f}"
                    if normalize
                    else (f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.2f}")
                )
                ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=11)

    # 标题：更大、更靠上；并与子图留更大空白
    fig.suptitle("Confusion Matrices", fontweight="bold", fontsize=20, y=0.995)

    # 调整留白与间距（加大顶部留白、右侧给色条更多空间；增大子图间距）
    plt.subplots_adjust(
        left=0.07, right=0.82, top=0.88, bottom=0.07, wspace=0.36, hspace=0.46
    )

    # 色条：进一步右移，不与子图挤在一起
    cbar_ax = fig.add_axes([0.90, 0.18, 0.025, 0.64])  # [left, bottom, width, height]
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.ax.set_ylabel("Count" if not normalize else "Proportion")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_pr_f1(csv_path, save_path=None, dpi=300):
    """
    从CSV(列: model, metric, mean, std)读取 Precision / Recall / F1，
    绘制两簇柱状图(Precision, Recall)并用折线(小黑方块)标注每个柱子的F1。
    - 图像更紧凑、柱更细
    - x轴标签 Precision / Recall 加粗放大
    - 去掉 'Metric' 标签
    - 标题上移、加粗放大：'Precision & Recall with F1 Overlay'
    """
    df = pd.read_csv(csv_path).copy()
    df["metric"] = df["metric"].str.upper()

    model_order = [
        "No Time Feature",
        "No Node Mask",
        "No Edge Reconstruction",
        "Full Model",
    ]
    exist_models = [m for m in model_order if m in df["model"].unique()]
    if len(exist_models) != 4:
        exist_models = list(df["model"].unique())

    colors = ["#82B0D2", "#BEB8DC", "#E7DAD2", "#999999"][: len(exist_models)]

    def pick(metric, models):
        vals = []
        for m in models:
            sub = df[(df.model == m) & (df.metric == metric)]
            if sub.empty:
                raise ValueError(f"CSV中缺少 {m} 的 {metric} 数据")
            vals.append(float(sub["mean"].values[0]))
        return np.array(vals, dtype=float)

    prec_vals = pick("PRECISION", exist_models)
    rec_vals = pick("RECALL", exist_models)
    f1_vals = pick("F1", exist_models)

    n_models = len(exist_models)
    width = 0.14  # 柱宽缩小
    gap_between_clusters = 1.1  # 簇间距略收紧
    cluster_x = np.array([0.0, gap_between_clusters])
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * (width + 0.02)

    x_prec = cluster_x[0] + offsets
    x_rec = cluster_x[1] + offsets

    # 图宽缩小
    fig, ax = plt.subplots(figsize=(6, 4.6))

    # Precision 簇
    for x, h, c in zip(x_prec, prec_vals, colors):
        ax.bar(x, h, width=width, color=c, edgecolor="black", linewidth=0.6, alpha=0.8)

    # Recall 簇
    for x, h, c in zip(x_rec, rec_vals, colors):
        ax.bar(x, h, width=width, color=c, edgecolor="black", linewidth=0.6, alpha=0.8)

    # F1 折线（两簇叠加）
    ax.plot(
        x_prec,
        f1_vals,
        linestyle="-",
        marker="s",
        markersize=4,
        linewidth=1.2,
        color="black",
        label="F1",
    )
    ax.plot(
        x_rec,
        f1_vals,
        linestyle="-",
        marker="s",
        markersize=4,
        linewidth=1.2,
        color="black",
    )

    # 纵轴范围（凸显差异）
    all_vals = np.concatenate([prec_vals, rec_vals, f1_vals])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    margin = min(max((vmax - vmin) * 0.18, 0.005), 0.05)
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))

    # x轴：只保留两簇标签并加粗放大
    ax.set_xticks(cluster_x)
    ax.set_xticklabels(["Precision", "Recall"], fontweight="bold", fontsize=12)
    ax.set_ylabel("Score")
    # 去掉 'Metric' 标签：不设置 xlabel

    # 标题上移、加粗放大，并增大与图主体的空隙
    ax.set_title(
        "Precision & Recall with F1 Overlay", fontweight="bold", fontsize=14, pad=18
    )

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # 图例：4个模型 + F1
    handles = [
        Patch(facecolor=colors[i], edgecolor="black", label=exist_models[i], alpha=0.8)
        for i in range(n_models)
    ]
    handles.append(
        Line2D(
            [0], [0], color="black", marker="s", markersize=5, linewidth=1.2, label="F1"
        )
    )
    leg = ax.legend(handles=handles, loc="upper left", frameon=True, ncol=1, fontsize=8)
    leg.get_frame().set_alpha(0.6)

    # 进一步拉大标题与主体的间距
    plt.subplots_adjust(top=0.80)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def merge_two_figs(
    wget_dir: str,
    cic_dir: str,
    out_path: Optional[str] = None,
    orientation: Literal["h", "horizontal", "v", "vertical"] = "h",
    dpi: int = 300,
    pad_ratio: float = 0.02,
    font_size: int = 12,
):
    """
    将两张图片合并到一张Matplotlib图中，并在每张图片的**下方**加标签。
    - wget_dir:  第一张图片路径（标注为 'Wget Dataset'）
    - cic_dir:   第二张图片路径（标注为 'CICAPT_IIOT Dataset'）
    - out_path:  输出文件路径（如不传则仅显示不保存）
    - orientation: 'h'/'horizontal' 横向并排；'v'/'vertical' 纵向叠放（默认'h'）
    - dpi:       输出分辨率
    - pad_ratio: 子图之间间距（相对图宽/高）
    - font_size: 标签字号
    """
    assert os.path.isfile(wget_dir), f"File not found: {wget_dir}"
    assert os.path.isfile(cic_dir), f"File not found: {cic_dir}"

    # 用 PIL 读取尺寸用于设定画布比例
    img1 = Image.open(wget_dir).convert("RGB")
    img2 = Image.open(cic_dir).convert("RGB")
    w1, h1 = img1.size
    w2, h2 = img2.size

    # 根据方向自适应画布比例，尽量保持原图纵横比视觉均衡
    if orientation in ("v", "vertical"):
        # 纵向：宽取两者最大，高为两图高度之和
        W = max(w1, w2)
        H = h1 + h2
        figsize = (W / 100, H / 100)  # 经验换算，避免图过小
        nrows, ncols = 2, 1
    else:
        # 横向：高取两者最大，宽为两图宽度之和
        W = w1 + w2
        H = max(h1, h2)
        figsize = (W / 100, H / 100)
        nrows, ncols = 1, 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)

    # 统一成可迭代列表
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = list(axes.flat)

    # 显示两张图
    axes[0].imshow(img1)
    axes[0].axis("off")
    axes[0].set_xlabel("Wget Dataset", fontsize=font_size, labelpad=10)

    axes[1].imshow(img2)
    axes[1].axis("off")
    axes[1].set_xlabel("CICAPT_IIOT Dataset", fontsize=font_size, labelpad=10)

    # 子图间距
    if orientation in ("v", "vertical"):
        plt.subplots_adjust(hspace=pad_ratio, wspace=0.0)
    else:
        plt.subplots_adjust(wspace=pad_ratio, hspace=0.0)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        print(f"Saved to: {out_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    wget_dir = "results/wget/auc_f1.png"
    cic_dir = "results/CICAPT/auc_f1.png"
    save_path = "results/auc_f1.png"

    merge_two_figs(wget_dir, cic_dir, out_path=save_path, orientation="h")
