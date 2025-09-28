import xxhash
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np
import re


def hashgen(l):
    """从列表中生成单个哈希值。@l是一个字符串列表，可以是节点/边的属性。
    此函数返回一个哈希后的整数值。"""
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()


def parse_nodes_from_csv(file_path, node_dict=None):
    """
    解析CSV文件，构建节点到subtype和label的映射。

    参数:
        file_path (str): CSV文件路径
        node_dict (dict): 可选参数，外部传入的字典用于存储结果。如果为None则新建。

    返回:
        dict: {id: {"subtype": subtype, "label": label}}
    """
    if node_dict is None:
        node_dict = {}

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            node_id = row.get("id", "").strip()
            if node_id:  # 只处理id不为空的
                subtype = row.get("subtype", "").strip() or "None"
                label = row.get("label", "").strip() or "None"
                node_dict[node_id] = {"subtype": subtype, "label": label}
    return node_dict


def parse_edges_to_log(csv_path, node_dict, hashgen, encoding="utf-8"):
    """
    解析CSV中的边（from/to均不为空），输出同名.log文件。

    每行输出格式：
        <源节点ID>\t<目的结点ID>\t<源结点类型>:<目的结点类型>:<边类型>:<边操作>:<时间戳>
    其中：
      - 源/目的节点ID：对原始 from/to 字符串调用 hashgen() 得到的数字
      - 源/目的节点类型：从 node_dict[id] 取（若缺失或空则用 "None"）
      - 边类型：CSV 的 type 字段
      - 边操作：CSV 的 operation 字段
      - 时间戳：CSV 的 time 字段

    返回：
        output_path (str): 生成的 .log 文件路径
        edge_count (int): 写出的边条数
    """
    csv_path = Path(csv_path)
    output_path = csv_path.with_suffix(".log")

    edge_count = 0
    with open(csv_path, "r", encoding=encoding, newline="") as f_in, open(
        output_path, "w", encoding=encoding, newline="\n"
    ) as f_out:

        reader = csv.DictReader(f_in)
        # 容错：字段名大小写/空格可能不一致时，统一用get读取
        for row in reader:
            src_raw = (row.get("from") or "").strip()
            dst_raw = (row.get("to") or "").strip()
            if not src_raw or not dst_raw:
                continue  # 仅解析 from/to 都不为空的边

            # 结点类型（来自 node_dict 的 subtype），缺失则 "None"
            src_type = (node_dict.get(src_raw) or "None").strip() or "None"
            dst_type = (node_dict.get(dst_raw) or "None").strip() or "None"

            # 生成哈希ID
            try:
                src_id = str(hashgen(src_raw))
            except Exception:
                # 若 hashgen 对异常输入出错，兜底使用原字符串
                src_id = src_raw
            try:
                dst_id = str(hashgen(dst_raw))
            except Exception:
                dst_id = dst_raw

            # 边属性
            edge_type = (row.get("type") or "").strip()
            operation = (row.get("operation") or "").strip()
            timestamp = (row.get("time") or "").strip()

            # 拼接一行
            # <src_id>\t<dst_id>\t<src_type>:<dst_type>:<edge_type>:<operation>:<timestamp>
            payload = f"{src_type}:{dst_type}:{edge_type}:{operation}:{timestamp}"
            line = f"{src_id}\t{dst_id}\t{payload}\n"
            f_out.write(line)
            edge_count += 1

    return str(output_path), edge_count


def parse_edges_to_log_from_csv(csv_path, out_dir, out_filename, encoding="utf-8"):
    """
    读取已标准化的边CSV，输出为.log：
      输入CSV需要包含列：
        src_id, dst_id, src_type, dst_type, edge_type, operation, time, label, time_norm
      其中：time_norm 与 label 不使用

    每行输出格式：
      <src_id>\\t<dst_id>\\t<src_type>:<dst_type>:<edge_type>:<operation>:<time>

    参数：
      csv_path (str | Path): 输入CSV路径
      out_dir  (str | Path): 输出目录（自动创建）
      encoding (str): 文件编码

    返回：
      (str, int): (输出.log文件路径, 写出的边条数)
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 目标文件名：与输入同名但后缀改为 .log，输出到 out_dir 下
    output_path = out_dir / out_filename

    edge_count = 0
    with open(csv_path, "r", encoding=encoding, newline="") as f_in, open(
        output_path, "w", encoding=encoding, newline="\n"
    ) as f_out:
        reader = csv.DictReader(f_in)

        # 逐行读取并写出
        for row in reader:
            # 必要字段读取与清洗
            src_id = (row.get("src_id") or "").strip()
            dst_id = (row.get("dst_id") or "").strip()
            src_type = (row.get("src_type") or "").strip() or "None"
            dst_type = (row.get("dst_type") or "").strip() or "None"
            edge_ty = (row.get("edge_type") or "").strip()
            oper = (row.get("operation") or "").strip()
            ts = (row.get("time") or "").strip()

            # 仅当 src_id/dst_id 非空时写出
            if not src_id or not dst_id:
                continue

            payload = f"{src_type}:{dst_type}:{edge_ty}:{oper}:{ts}"
            line = f"{src_id}\t{dst_id}\t{payload}\n"
            f_out.write(line)
            edge_count += 1

    return str(output_path), edge_count


def parse_edges_to_csv(
    csv_path, node_dict, hashgen, encoding="utf-8", output_path=None
):
    """
    解析CSV中的边（from/to均不为空），输出为CSV文件，列为：
    src_id, dst_id, src_type, dst_type, edge_type, operation, time, label

    参数:
        csv_path (str | Path): 输入CSV路径
        node_dict (dict): {id: {"subtype": subtype, "label": label}} 的映射
        hashgen (callable): hashgen(str) -> int，返回数字id
        encoding (str): 文件编码
        output_path (str | Path | None): 可选，输出CSV路径。默认为 <输入文件名>.edges.csv

    返回:
        (str, int): (输出文件路径, 写出的边条数)
    """
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_suffix(".edges.csv")
    else:
        output_path = Path(output_path)

    header = [
        "src_id",
        "dst_id",
        "src_type",
        "dst_type",
        "edge_type",
        "operation",
        "time",
        "label",
    ]
    edge_count = 0

    with open(csv_path, "r", encoding=encoding, newline="") as f_in, open(
        output_path, "w", encoding=encoding, newline=""
    ) as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(header)

        for row in reader:
            src_raw = (row.get("from") or "").strip()
            dst_raw = (row.get("to") or "").strip()
            if not src_raw or not dst_raw:
                continue  # 仅解析 from/to 都不为空的边

            # 结点类型（来自 node_dict 的 subtype），缺失或空 -> "None"
            src_info = node_dict.get(src_raw, {})
            dst_info = node_dict.get(dst_raw, {})
            src_type = (
                src_info.get("subtype") if isinstance(src_info, dict) else src_info
            ) or "None"
            dst_type = (
                dst_info.get("subtype") if isinstance(dst_info, dict) else dst_info
            ) or "None"

            # 生成数字ID（异常兜底回原字符串）
            try:
                src_id = str(hashgen(src_raw))
            except Exception:
                src_id = src_raw
            try:
                dst_id = str(hashgen(dst_raw))
            except Exception:
                dst_id = dst_raw

            # 边属性
            edge_type = (row.get("type") or "").strip()
            operation = (row.get("operation") or "").strip()
            timestamp = (row.get("time") or "").strip()

            # 边label逻辑：
            # 1) 如果from或to结点的label=1，则边label=1
            # 2) 否则用row里的label
            src_label = src_info.get("label") if isinstance(src_info, dict) else None
            dst_label = dst_info.get("label") if isinstance(dst_info, dict) else None
            if src_label == "1" or dst_label == "1":
                label = "1"
            else:
                label = (row.get("label") or "").strip() or "0"

            writer.writerow(
                [
                    src_id,
                    dst_id,
                    src_type,
                    dst_type,
                    edge_type,
                    operation,
                    timestamp,
                    label,
                ]
            )
            edge_count += 1

    return str(output_path), edge_count


def time_normalize(filename):
    # 1. 读取所有行，提取时间戳统计信息
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    stats = []
    for line in lines:
        parts = line.strip().split("\t")
        fields = parts[2].split(":")
        stats.append(float(fields[-1]))  # 最后一个就是时间戳统计信息

    # 2. min-max归一化
    vmin, vmax = min(stats), max(stats)
    if vmax == vmin:
        norm_stats = [0.0] * len(stats)
    else:
        norm_stats = [(v - vmin) / (vmax - vmin) for v in stats]

    # 3. 重新写入文件，每行在末尾追加归一化时间
    with open(filename, "w", encoding="utf-8") as f:
        for line, norm in zip(lines, norm_stats):
            line = line.strip()
            new_line = f"{line}:{norm:.6f}\n"
            f.write(new_line)


def normalize_time_in_csv(csv_path, encoding="utf-8"):
    """
    读取CSV文件，读取'time'字段，归一化后追加'time_norm'字段，并覆盖保存。

    归一化规则：
        time_min = df['time'].min()
        df['time_norm'] = (df['time'] - time_min).round().astype(int)

    参数:
        csv_path (str): 输入CSV文件路径（会被覆盖保存）
        encoding (str): 文件编码，默认 utf-8

    返回:
        pandas.DataFrame: 修改后的DataFrame（包含time_norm列）
    """
    df = pd.read_csv(csv_path, encoding=encoding)

    if "time" not in df.columns:
        raise ValueError("CSV 文件中缺少 'time' 字段")

    time_min = df["time"].min()
    df["time_norm"] = (df["time"] - time_min).round().astype(int)

    # 覆盖原文件
    df.to_csv(csv_path, index=False, encoding=encoding)

    return df


def plot_event_and_kde(csv_path, encoding="utf-8"):
    """
    从处理好的CSV读取 label 与 time_norm，绘制：
      1) 事件时间轴（eventplot）
      2) KDE密度曲线（随时间对比不同label的分布）

    要求：
      - CSV 至少包含列：'label', 'time_norm'
      - 不保存图片，直接 plt.show()

    参数:
      csv_path (str): 输入CSV路径
      encoding (str): 文件编码
    """
    # 读取与基本清洗
    df = pd.read_csv(csv_path, encoding=encoding)
    if "label" not in df.columns or "time_norm" not in df.columns:
        raise ValueError("CSV 文件中必须包含 'label' 与 'time_norm' 两列")

    # label 转数值，缺失→0
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    # 保证 time_norm 为数值
    df["time_norm"] = pd.to_numeric(df["time_norm"], errors="coerce")
    df = df.dropna(subset=["time_norm"]).copy()
    df = df.sort_values("time_norm").reset_index(drop=True)

    # 颜色（按示例固定：0=灰，1=红；其他标签给默认色）
    COLOR_NORMAL = "#cccccc"  # 灰
    COLOR_MAL = "#e41a1c"  # 红
    color_map = {0: COLOR_NORMAL, 1: COLOR_MAL}

    # ============== 图1：事件时间轴（Timeline / Eventplot） ==============
    plt.figure(figsize=(12, 2.2))

    # 仅绘制存在的标签
    for lab in sorted(df["label"].unique()):
        xs = df.loc[df["label"] == lab, "time_norm"].to_numpy()
        if xs.size == 0:
            continue
        plt.eventplot(
            xs,
            lineoffsets=0.5,
            linelengths=0.6,
            colors=color_map.get(lab, None),
            linewidths=1,
        )

    plt.xlabel("Normalized Time")
    plt.yticks([])
    plt.title("Event Timeline (Gray=0, Red=1)")

    # 图例（仅包含0/1两类；若需要可按unique动态生成）
    handles = []
    if 0 in df["label"].unique():
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=COLOR_NORMAL,
                marker="|",
                linestyle="None",
                markersize=12,
                label="Label=0",
            )
        )
    if 1 in df["label"].unique():
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=COLOR_MAL,
                marker="|",
                linestyle="None",
                markersize=12,
                label="Label=1",
            )
        )
    if handles:
        plt.legend(handles=handles, loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()

    # ============== 图2：KDE 密度曲线（按类别对比） ==============
    plt.figure(figsize=(12, 3.2))

    # 为避免KDE在数据过少时报错，逐类判断数量
    unique_labels = sorted(df["label"].unique())
    any_plotted = False
    for lab in unique_labels:
        xs = df.loc[df["label"] == lab, "time_norm"].dropna()
        if xs.shape[0] < 2:
            continue  # KDE至少需要一些点
        sns.kdeplot(
            data=xs,
            fill=True,
            alpha=0.25,
            color=color_map.get(lab, None),
            label=f"Label={lab}",
        )
        # 可选rug：点很多时可注释
        sns.rugplot(x=xs, height=0.03, color=color_map.get(lab, None), alpha=0.6)
        any_plotted = True

    plt.xlabel("Normalized Time")
    plt.ylabel("Density")
    plt.title("KDE Density over Time (Gray=0, Red=1)")
    if any_plotted:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def snapshot_edges_inplace(
    csv_path,
    time_col="time_norm",
    target_snapshots=35,
    burst_frac=0.7,  # “爆发区占比”，默认前2/3时间为爆发区；若想关闭爆发划分，可传0
    encoding="utf-8",
    na_policy="drop",  # 'drop'：丢弃time缺失行；'keep'：保留并标记snapshot=-1
):
    """
    根据 time_col(归一化时间) 对边做快照划分，并在源CSV末尾添加 snapshot 列（0..K-1）。

    参数
    ----
    csv_path : str | Path
        输入CSV路径（将被覆盖写回）
    time_col : str
        时间列名（归一化后的时间），默认 'time_norm'
    target_snapshots : int
        目标快照总数（爆发区+平稳区）
    burst_frac : float in [0,1)
        爆发区时间占比阈值，例如 2/3 表示前2/3时间为爆发区；0 表示不划分爆发区
    encoding : str
        文件编码
    na_policy : {'drop','keep'}
        对 time_col 缺失/非数值的处理策略

    返回
    ----
    int
        实际写入的快照数（可能与 target_snapshots 一致）
    """
    csv_path = Path(csv_path)

    # 读入
    df = pd.read_csv(csv_path, encoding=encoding)

    if time_col not in df.columns:
        raise ValueError(f"列 {time_col} 不存在。")

    # 处理时间缺失/非法
    t = pd.to_numeric(df[time_col], errors="coerce")
    if na_policy == "drop":
        valid_mask = t.notna()
        df_valid = df.loc[valid_mask].copy()
        t_valid = t.loc[valid_mask].to_numpy()
    else:
        # 保留非法行，稍后标记 snapshot = -1
        df_valid = df.copy()
        t_valid = pd.to_numeric(df_valid[time_col], errors="coerce").to_numpy()
        keep_invalid = True

    if len(df_valid) == 0:
        # 没有可用时间数据
        if na_policy == "keep":
            df["snapshot"] = -1
            df.to_csv(csv_path, index=False, encoding=encoding)
            return 0
        else:
            raise ValueError("没有有效的时间数据可用于切分。")

    # ===== 1) 排序 & 定义两段 =====
    df_sorted = df_valid.sort_values(time_col).reset_index()  # 保留原索引以便回写
    t_sorted = pd.to_numeric(df_sorted[time_col], errors="coerce").to_numpy()

    Tmin, Tmax = float(t_sorted[0]), float(t_sorted[-1])
    Tburst = Tmin + float(burst_frac) * (Tmax - Tmin)

    # 按“时间阈值”分两段（半开区间）
    idx_burst_end = int(np.searchsorted(t_sorted, Tburst, side="left"))
    n_total = len(t_sorted)
    n_burst = idx_burst_end
    n_calm = n_total - n_burst

    # ===== 2) 按事件占比分配快照数 =====
    # 至少各1个窗口，以避免分母为0；若 burst_frac=0，爆发区自动为1个但大小可能为0，后续会跳过空窗口
    snap_burst = (
        max(1, int(round(target_snapshots * n_burst / n_total)))
        if target_snapshots > 1
        else 1
    )
    snap_calm = max(1, target_snapshots - snap_burst)

    # ===== 3) 工具函数：等事件数切分（按索引均分）=====
    def equal_count_edges(n_events, k_windows, start_idx=0):
        """
        把 [start_idx, start_idx+n_events) 这段索引均分为 k 段（尽量均匀），
        返回 k 个 (idx_start, idx_end) 半开区间。
        """
        if k_windows <= 0:
            return []
        if n_events <= 0:
            # 全空
            return [(start_idx, start_idx) for _ in range(k_windows)]
        base = n_events // k_windows
        rem = n_events % k_windows
        sizes = [base + (1 if i < rem else 0) for i in range(k_windows)]
        edges = []
        cur = start_idx
        for sz in sizes:
            edges.append((cur, cur + sz))
            cur += sz
        return edges

    # 爆发区/平稳区按等事件数切分（索引区间）
    edges_burst_idx = equal_count_edges(n_burst, snap_burst, start_idx=0)
    edges_calm_idx = equal_count_edges(n_calm, snap_calm, start_idx=idx_burst_end)

    # ===== 4) 给排序后的行分配 snapshot id =====
    snap_id_sorted = np.full(n_total, -1, dtype=int)  # 先全部-1

    # 按顺序编号：爆发区窗口 0..(B-1)，平稳区继续编号 B..(B+C-1)
    cur_id = 0
    for s, e in edges_burst_idx:
        if e > s:
            snap_id_sorted[s:e] = cur_id
        cur_id += 1
    for s, e in edges_calm_idx:
        if e > s:
            snap_id_sorted[s:e] = cur_id
        cur_id += 1

    # 实际使用到的最大编号（可能因为空窗口导致有的编号没分配到行）
    used_mask = snap_id_sorted >= 0
    if not np.any(used_mask):
        # 极端：所有窗口为空（几乎不可能，除非n_total=0）
        actual_snapshots = 0
    else:
        # 将可能出现的“空洞编号”压实为 0..(K-1)
        used_ids = np.unique(snap_id_sorted[used_mask])
        id_remap = {old: i for i, old in enumerate(used_ids)}
        snap_id_sorted = np.array(
            [id_remap.get(x, -1) for x in snap_id_sorted], dtype=int
        )
        actual_snapshots = len(used_ids)

    # 把 snapshot 写回到原索引
    df_valid_with_snap = df_sorted.copy()
    df_valid_with_snap["snapshot"] = snap_id_sorted

    # 回写到 df（保持原行顺序）
    if na_policy == "drop":
        df["snapshot"] = -1  # 先标记所有为-1
        df.loc[df_valid_with_snap["index"], "snapshot"] = df_valid_with_snap[
            "snapshot"
        ].to_numpy()
    else:
        # keep：已经保留全部行，只需在原索引上写值
        df["snapshot"] = -1
        df.loc[df_valid_with_snap["index"], "snapshot"] = df_valid_with_snap[
            "snapshot"
        ].to_numpy()

    # 覆盖写回源文件
    df.to_csv(csv_path, index=False, encoding=encoding)

    return actual_snapshots


def label_snapshots_inplace(
    csv_path,
    snapshot_col="snapshot",
    edge_label_col="label",
    snapshot_label_col="snapshot_label",
    THRESH_P=0.003,  # 恶意占比阈值（建议你用KDE得到后传入）
    MIN_MAL=1,  # 防止极少数恶意点误判；若不需要，设为1
    encoding="utf-8",
):
    """
    基于每个快照中恶意边(label==1)占比对快照打标，并把 snapshot_label 写回原CSV。
    若 snapshot_label 列已存在，会先删除再重建。

    返回:
        (n_mal_snapshots, mal_ratio_over_snaps)
        —— 恶意快照数量、恶意快照占比（仅在 snapshot>=0 的快照上统计）
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, encoding=encoding)

    # 如果已有 snapshot_label 列，先删掉
    if snapshot_label_col in df.columns:
        df.drop(columns=[snapshot_label_col], inplace=True)

    # 基本列检查
    for c in [snapshot_col, edge_label_col]:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")

    # 仅在有效快照上分组（snapshot>=0）
    df["_snap_"] = (
        pd.to_numeric(df[snapshot_col], errors="coerce").fillna(-1).astype(int)
    )
    df["_lbl_"] = (
        pd.to_numeric(df[edge_label_col], errors="coerce").fillna(0).astype(int)
    )

    valid_mask = df["_snap_"] >= 0
    grp = df.loc[valid_mask].groupby("_snap_")["_lbl_"]
    stat = pd.DataFrame({"n_total": grp.size(), "n_mal": grp.sum()}).reset_index()

    # 计算占比并打快照标签
    stat["mal_ratio"] = stat["n_mal"] / stat["n_total"].replace(0, 1)
    stat[snapshot_label_col] = (
        (stat["mal_ratio"] >= THRESH_P) & (stat["n_mal"] >= MIN_MAL)
    ).astype(int)

    # 合并回原表
    df = df.merge(
        stat[["_snap_", snapshot_label_col]],
        how="left",
        left_on="_snap_",
        right_on="_snap_",
    )
    df[snapshot_label_col] = df[snapshot_label_col].fillna(-1).astype(int)

    # 去掉临时列
    df.drop(columns=["_snap_", "_lbl_"], inplace=True)

    # 写回原CSV
    df.to_csv(csv_path, index=False, encoding=encoding)

    # 统计并返回
    if len(stat) == 0:
        return 0, 0.0
    n_mal = int(stat[snapshot_label_col].sum())
    mal_ratio = float(n_mal / len(stat))
    return n_mal, mal_ratio


def split_snapshots_to_csv(
    csv_path,
    out_dir,
    snapshot_col="snapshot",
    label_col="snapshot_label",
    encoding="utf-8",
    skip_invalid=True,
):
    """
    将带有 snapshot 与 snapshot_label 列的CSV拆分为多个子CSV。
    - 良性快照文件编号从0开始： {i}-benign.csv
    - 恶意快照文件编号从0开始： {j}-attack.csv
    - 输出前丢弃列 snapshot 与 snapshot_label
    - 返回生成的快照文件总数
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding=encoding)

    # 基本检查
    for col in (snapshot_col, label_col):
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 规范化类型
    df["_snap_"] = pd.to_numeric(df[snapshot_col], errors="coerce")
    df["_lab_"] = pd.to_numeric(df[label_col], errors="coerce")

    # 过滤无效
    valid_mask = np.ones(len(df), dtype=bool)
    if skip_invalid:
        valid_mask &= df["_snap_"].notna() & (df["_snap_"] >= 0)
        valid_mask &= df["_lab_"].isin([0, 1])

    df_valid = df.loc[valid_mask].copy()
    if df_valid.empty:
        return 0  # 没有有效快照

    # 分组
    df_valid["_snap_"] = df_valid["_snap_"].astype(int)
    groups = df_valid.groupby("_snap_", sort=True)

    benign_idx = 0
    attack_idx = 0
    total_files = 0

    for snap_id, g in groups:
        uniq = g["_lab_"].dropna().unique()
        if len(uniq) == 0:
            if skip_invalid:
                continue
            lbl = 0
        elif len(uniq) == 1:
            lbl = int(uniq[0])
        else:
            lbl = 1  # 混合视为恶意

        if lbl == 1:
            fname = f"{attack_idx}-attack.csv"
            attack_idx += 1
        else:
            fname = f"{benign_idx}-benign.csv"
            benign_idx += 1

        out_file = out_dir / fname
        drop_cols = [
            c for c in (snapshot_col, label_col, "_snap_", "_lab_") if c in g.columns
        ]
        sub = g.drop(columns=drop_cols, errors="ignore")
        sub.to_csv(out_file, index=False, encoding=encoding)

        total_files += 1

    return total_files


def split_benign_snapshots_to_csv(
    csv_path,
    out_dir,
    snapshot_col="snapshot",
    start_idx=16,  # 文件编号起点
    encoding="utf-8",
    skip_invalid=True,
):
    """
    根据 snapshot 列划分快照，把每个快照输出为一个CSV文件。
    - 所有快照都标记为 benign
    - 文件名从 {start_idx}-benign.csv 开始递增
    - 输出前丢弃 snapshot 列
    - 返回生成的快照文件总数
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding=encoding)

    if snapshot_col not in df.columns:
        raise ValueError(f"缺少必要列: {snapshot_col}")

    # 规范化 snapshot 列
    df["_snap_"] = pd.to_numeric(df[snapshot_col], errors="coerce")

    # 过滤无效
    valid_mask = df["_snap_"].notna()
    if skip_invalid:
        valid_mask &= df["_snap_"] >= 0

    df_valid = df.loc[valid_mask].copy()
    if df_valid.empty:
        return 0  # 没有有效快照

    # 分组
    df_valid["_snap_"] = df_valid["_snap_"].astype(int)
    groups = df_valid.groupby("_snap_", sort=True)

    total_files = 0
    file_idx = start_idx

    for snap_id, g in groups:
        fname = f"{file_idx}-benign.csv"
        out_file = out_dir / fname

        drop_cols = [c for c in (snapshot_col, "_snap_") if c in g.columns]
        sub = g.drop(columns=drop_cols, errors="ignore")
        sub.to_csv(out_file, index=False, encoding=encoding)

        total_files += 1
        file_idx += 1

    return total_files


def collect_numbered_files(dir_path: Path, suffix: str):
    """
    在目录下收集形如 'N-<suffix>.csv' 的文件，并按 N 升序返回文件名列表。
    suffix 示例：'attack' 或 'benign'
    """
    pattern = re.compile(rf"^(\d+)-{re.escape(suffix)}\.csv$", re.IGNORECASE)
    numbered = []
    for p in dir_path.glob(f"*-{suffix}.csv"):
        m = pattern.match(p.name)
        if m:
            idx = int(m.group(1))
            numbered.append((idx, p.name))
    numbered.sort(key=lambda x: x[0])
    return [name for _, name in numbered]


def batch_generate_logs(graphs_dir, out_dir):
    graphs_dir = Path(graphs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 先处理恶意 attack，再处理良性 benign
    attack_files = collect_numbered_files(graphs_dir, "attack")
    benign_files = collect_numbered_files(graphs_dir, "benign")

    # 若你想强制“只处理前14个恶意文件（从0开始编号）”，可取消下一行注释：
    # attack_files = [fn for fn in attack_files if int(fn.split('-')[0]) <= 13]

    log_id = 1  # 输出 .log 文件名从 1 开始
    attack_count = 0
    benign_count = 0

    # 先恶意
    for fname in attack_files:
        file_path = str(graphs_dir / fname)
        out_filename = f"{log_id}.log"
        path, count = parse_edges_to_log_from_csv(file_path, str(out_dir), out_filename)
        time_normalize(path)
        print(path, count)
        log_id += 1
        attack_count += 1

    # 再良性
    for fname in benign_files:
        file_path = str(graphs_dir / fname)
        out_filename = f"{log_id}.log"
        path, count = parse_edges_to_log_from_csv(file_path, str(out_dir), out_filename)
        time_normalize(path)
        print(path, count)
        log_id += 1
        benign_count += 1

    return attack_count, benign_count


if __name__ == "__main__":
    file_path = "data/CICAPT_IIOT/graphs/3-attack.csv"
    out_path = "data/CICAPT_IIOT/graphs_benign/"
    out_dir = "data/CICAPT_IIOT/processed/"

    csv_dir = "data/CICAPT_IIOT/graphs/"
    log_dir = "data/CICAPT_IIOT/processed/"

    attack_count, benign_count = batch_generate_logs(csv_dir, log_dir)

    print(attack_count, benign_count)
