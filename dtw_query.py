# -*- coding: utf-8 -*-
"""
dtw_query.py — DTW 时序查询核心

新增独立查询路径，不修改现有 PlanningEngine / query.py。
相似度度量：加权 cos 相似度
    - 标准化：z-score（全量数据预计算 mean/std，保存到 paths.dtw_norm_stats_path）
    - 加权：特征乘 √w（权重0的特征自动排除）
    - DTW 对齐代价 = 1 - cos（cos 越大越相似，代价越小越好）
    - 对齐后相似度 = (cos+1)/2 均值，映射到 [0,1]
依赖：
    - plan_center.features（add_residual_features / load_residual_models）
    - plan_center.config（DTWQueryConfig / PlanningConfig / build_feature_weights）
    - plan_center.schemas（PlanResult / plan_result_to_row / build_output_dataframe）
    - plan_center.batch（_resolve_time_col）

用法：
    from plan_center.dtw_query import DTWQueryEngine, query_dtw
    engine = DTWQueryEngine(cfg)
    result = engine.query_one(timestamp="2024-11-01 08:00:00")
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")

from .config import DTWQueryConfig, PlanningConfig, build_feature_weights
from .features import add_residual_features, load_residual_models
from .schemas import PlanResult
from .batch import _resolve_time_col


# ============================================================
# 工具：DTW 对齐（加权 cos 相似度代价，numpy 手动实现）
# ============================================================


def dtw_align_cos(
    query_seq: np.ndarray,
    cand_seq: np.ndarray,
) -> tuple[list[tuple[int, int]], float, np.ndarray]:
    """
    标准 DTW 动态规划对齐（cos 距离代价）。

    输入序列需预先做 z-score 标准化 + √w 加权。

    参数：
        query_seq: (T_q, D) 查询序列（已标准化+加权）
        cand_seq: (T_c, D) 候选序列（已标准化+加权）

    返回：
        (aligned_pairs, path_cost, cos_matrix)
        aligned_pairs: [(i_q, i_c), ...]，正序
        path_cost: 最小累积代价（DTW 距离）
        cos_matrix: (T_q, T_c) 逐点 cos 相似度矩阵，供 dtw_cos_mean 复用
    """
    n, m = query_seq.shape[0], cand_seq.shape[0]

    # 预计算逐点 cos 距离矩阵 (n, m)，cos_dist = 1 - cos
    cos_matrix = cosine_similarity(query_seq, cand_seq)  # (n, m) ∈ [-1, 1]
    dist_matrix = 1.0 - cos_matrix  # [0, 2]

    # DP 成本矩阵
    D = np.full((n, m), np.inf, dtype=float)
    D[0, 0] = dist_matrix[0, 0]

    # 首行
    for j in range(1, m):
        D[0, j] = D[0, j - 1] + dist_matrix[0, j]

    # 首列
    for i in range(1, n):
        D[i, 0] = D[i - 1, 0] + dist_matrix[i, 0]

    # 内点
    for i in range(1, n):
        for j in range(1, m):
            D[i, j] = dist_matrix[i, j] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # 路径回溯
    aligned_pairs: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    aligned_pairs.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates_d = [D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]
            argmin = int(np.argmin(candidates_d))
            if argmin == 0:
                i -= 1
                j -= 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
        aligned_pairs.append((i, j))

    aligned_pairs.reverse()
    return aligned_pairs, float(D[-1, -1]), cos_matrix


def dtw_align_with_coverage_cos(
    query_seq: np.ndarray,
    cand_seq: np.ndarray,
    min_coverage: int = 4,
) -> tuple[list[tuple[int, int]], float, bool, np.ndarray]:
    """
    DTW 对齐 + 候选覆盖帧检查（cos 距离代价）。

    比 dtw_align_cos() 多返回一个 bool: 路径中涉及的不重复候选帧数 >= min_coverage。

    参数：
        query_seq: (T_q, D) 查询序列（已标准化+加权）
        cand_seq: (T_c, D) 候选序列（已标准化+加权）
        min_coverage: 最少不重复候选帧数（默认 4）

    返回：
        (aligned_pairs, path_cost, coverage_ok, cos_matrix)
    """
    aligned_pairs, path_cost, cos_matrix = dtw_align_cos(query_seq, cand_seq)
    # 计算路径覆盖的不重复候选帧数
    cand_indices = np.array([j for _, j in aligned_pairs])
    n_unique = int(np.unique(cand_indices).size)
    coverage_ok = n_unique >= min_coverage
    return aligned_pairs, path_cost, coverage_ok, cos_matrix


# ============================================================
# 工具：DTW 对齐后 cos 相似度均值（映射到 [0,1]）
# ============================================================


def dtw_cos_mean(
    cos_matrix: np.ndarray,
    aligned_pairs: list[tuple[int, int]],
) -> float:
    """
    对齐后从 cos_matrix 直接取值，映射 (cos+1)/2 到 [0,1]，取均值。

    参数：
        cos_matrix: (T_q, T_c) 逐点 cos 相似度矩阵（由 dtw_align_cos 预计算）
        aligned_pairs: DTW 对齐索引对 [(i_q, i_c), ...]

    返回：
        float，均值相似度 [0, 1]
    """
    if not aligned_pairs:
        return 0.0

    # 直接从 cos_matrix 取对齐点对的 cos 值
    cos_vals = np.array([cos_matrix[i, j] for i, j in aligned_pairs])
    cos_vals = np.clip(cos_vals, -1.0, 1.0)

    # 映射到 [0, 1] 后取均值
    sims = (cos_vals + 1.0) / 2.0
    return float(np.mean(sims))


# ============================================================
# 工具：残差缓存预处理
# ============================================================


def ensure_resid_cache(
    raw_parquet: str,
    model_dir: str,
    feat: Any,
    cache_parquet: str,
    feature_cols: list[str],
    alias_map: dict[str, str] | None = None,
) -> str:
    """
    确保残差缓存 parquet 存在。

    第一次调用时：
        1. 加载 #4_df_all_1min.parquet
        2. 列名别名映射（如 吨煤产汽量（2h）→ 吨煤产气量）
        3. 加载 residual_models/
        4. 全量计算 resid_* 列（add_residual_features）
        5. 保存缓存为 #4_df_all_1min_with_resid.parquet

    后续调用：直接返回缓存路径。

    参数：
        raw_parquet: 原始分钟级 parquet 路径
        model_dir: 残差模型目录
        feat: FeatureConfig
        cache_parquet: 缓存 parquet 路径
        feature_cols: 需要的特征列名列表（别名映射后）
        alias_map: 列名别名映射 {实际列名: 标准列名}

    返回：
        缓存 parquet 的绝对路径
    """
    cache_path = Path(cache_parquet)

    if cache_path.exists():
        print(f"[DTW] 残差缓存已存在: {cache_path}")
        return str(cache_path)

    print(f"[DTW] 首次查询，正在生成残差缓存 ...")
    print(f"[DTW]   加载原始数据: {raw_parquet}")
    t0 = time.time()

    df = pd.read_parquet(raw_parquet)
    print(f"[DTW]   原始数据 shape: {df.shape}")

    # 列名别名映射
    if alias_map:
        df = df.rename(columns=alias_map)
        print(f"[DTW]   列名别名映射: {alias_map}")

    # 加载残差模型
    print(f"[DTW]   加载残差模型: {model_dir}")
    models = load_residual_models(model_dir, feat.residual_targets)

    # 全量计算残差
    print(f"[DTW]   计算残差特征 ...")
    df_with_resid = add_residual_features(df, models, feat, name="全量缓存")

    # 保存缓存
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_resid.to_parquet(cache_path, index=False)
    elapsed = time.time() - t0
    print(f"[DTW]   残差缓存已保存: {cache_path}")
    print(f"[DTW]   耗时: {elapsed:.1f}s，shape: {df_with_resid.shape}")

    return str(cache_path)


# ============================================================
# 预筛辅助函数：负荷初筛 + cos 相似度预筛
# ============================================================


def load_filter_segments(
    ref_window: pd.DataFrame,
    load_col: str,
    q_load_mean: float,
    threshold: float,
    min_length: int,
) -> list[tuple[int, int]]:
    """
    负荷初筛：按主汽流量阈值筛选连续片段。

    参数：
        ref_window: 参考窗口 DataFrame
        load_col: 主汽流量列名
        q_load_mean: 查询序列平均主汽流量
        threshold: 负荷偏差阈值（t/h）
        min_length: 最小片段长度（行数）

    返回：
        [(start_idx, end_idx), ...] 连续片段列表（基于 ref_window 的行索引，左闭右开）
    """
    loads = ref_window[load_col].values
    mask = np.abs(loads - q_load_mean) <= threshold

    segments = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_length:
                segments.append((i, j))
            i = j
        else:
            i += 1

    return segments


def cos_prefilter(
    query_matrix: np.ndarray,       # (T_q, D) 查询序列（已标准化+加权）
    ref_window: pd.DataFrame,        # 参考窗口
    segments: list[tuple[int, int]], # 负荷初筛通过的片段
    sim_feature_cols: list[str],     # 15维特征列名
    cos_threshold: float,            # cos 相似度阈值
    cos_slide_step: int,             # 滑窗步长
    col_mean: np.ndarray,            # (D,) 全量数据均值（z-score 参数）
    col_std: np.ndarray,             # (D,) 全量数据标准差（z-score 参数）
    weight_sqrt: np.ndarray,         # (D,) √w 加权向量
    verbose: bool = True,
) -> list[int]:
    """
    cos 相似度预筛：在负荷初筛通过的片段上，等长滑窗 + z-score 标准化（全量统计量）+ √w 加权 + 逐点 cos。

    向量化实现：整个片段一次性标准化，所有滑窗的逐点 cos 用矩阵运算批量计算。

    参数：
        query_matrix: (T_q, D) 查询序列（原始值，内部做标准化+加权）
        ref_window: 参考窗口 DataFrame
        segments: 负荷初筛通过的片段 [(start, end), ...]
        sim_feature_cols: 特征列名列表
        cos_threshold: cos 相似度阈值，低于此值的窗口删掉
        cos_slide_step: 滑窗步长（分钟）
        col_mean: (D,) 全量数据均值
        col_std: (D,) 全量数据标准差
        weight_sqrt: (D,) √w 加权向量
        verbose: 是否打印进度

    返回：
        通过 cos 筛选的窗口起点索引列表（基于 ref_window 的行索引）
    """
    T_q, D = query_matrix.shape

    # 1. 查询序列 z-score 标准化 + 加权 + 预计算范数
    query_zw = ((query_matrix - col_mean) / col_std) * weight_sqrt  # (T_q, D)
    q_norm = np.linalg.norm(query_zw, axis=1)  # (T_q,)
    q_norm_safe = np.where(q_norm < 1e-12, 1e-12, q_norm)
    query_unit = query_zw / q_norm_safe[:, None]  # (T_q, D) 单位向量

    # 2. 在每个片段内批量计算所有滑窗的逐点 cos
    passed_starts: list[int] = []
    n_checked = 0
    n_passed = 0

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < T_q:
            continue

        # 提取片段特征矩阵
        seg_features = ref_window[sim_feature_cols].iloc[seg_start:seg_end].values.astype(float)
        # 填充 NaN
        if np.isnan(seg_features).any():
            inds = np.where(np.isnan(seg_features))
            seg_features[inds] = np.take(col_mean, inds[1])

        # z-score 标准化 + 加权（整个片段一次性）
        seg_zw = ((seg_features - col_mean) / col_std) * weight_sqrt  # (seg_len, D)
        # 预计算每个时间点的范数
        seg_norm = np.linalg.norm(seg_zw, axis=1)  # (seg_len,)
        seg_norm_safe = np.where(seg_norm < 1e-12, 1e-12, seg_norm)
        seg_unit = seg_zw / seg_norm_safe[:, None]  # (seg_len, D) 单位向量

        # 批量计算所有滑窗的逐点 cos
        # query_unit: (T_q, D), seg_unit: (seg_len, D)
        # 逐点 cos 矩阵: (T_q, seg_len) = query_unit @ seg_unit.T
        cos_pt = query_unit @ seg_unit.T  # (T_q, seg_len) ∈ [-1, 1]

        # 滑窗：每个窗口 cos 均值 = mean over t of cos_pt[t, win_start+t]
        n_windows = seg_len - T_q + 1
        for win_start in range(0, n_windows, cos_slide_step):
            # 提取对角线 cos 值：cos_pt[t, win_start + t] for t in range(T_q)
            diag_idx = np.arange(T_q) + win_start
            cos_vals = cos_pt[np.arange(T_q), diag_idx]  # (T_q,)
            cos_mean = float(np.mean(cos_vals))

            n_checked += 1
            if cos_mean >= cos_threshold:
                passed_starts.append(seg_start + win_start)
                n_passed += 1

    if verbose:
        print(f"[DTW] cos 预筛: 检查 {n_checked} 个窗口，通过 {n_passed} 个（阈值 {cos_threshold}）")

    return passed_starts


# ============================================================
# 并行 DTW 对齐辅助函数
# ============================================================


def _dtw_process_one(args: tuple) -> dict:
    """
    模块级 worker 函数，供 ThreadPoolExecutor/ProcessPoolExecutor 调用。
    参数通过 tuple 传递以支持 pickle 序列化。
    """
    query_matrix, cand_matrix, min_coverage = args
    aligned_pairs, path_cost, coverage_ok, cos_matrix = dtw_align_with_coverage_cos(
        query_matrix, cand_matrix, min_coverage=min_coverage
    )
    if not coverage_ok:
        sim = 0.0
    else:
        sim = dtw_cos_mean(cos_matrix, aligned_pairs)
    return {
        "sim": sim,
        "path_cost": path_cost,
        "path_length": len(aligned_pairs),
        "coverage_ok": coverage_ok,
    }


def _parallel_dtw_align(
    query_matrix: np.ndarray,
    candidates: list[dict],
    min_coverage: int,
    n_workers: int = 4,
) -> list[dict]:
    """
    并行计算多个候选的 DTW 对齐 + cos 相似度。

    使用 ThreadPoolExecutor，配合 numpy 矩阵运算释放 GIL 可部分并行。
    相比 ProcessPool 无 pickle 序列化开销，适合小任务。

    参数：
        query_matrix: (T_q, D) 查询序列（已标准化+加权）
        candidates: 候选列表，每个含 "matrix" 字段（已标准化+加权）
        min_coverage: 最小覆盖帧数
        n_workers: 并行线程数

    返回：
        [{"sim", "path_cost", "path_length", "coverage_ok"}, ...]
    """
    from concurrent.futures import ThreadPoolExecutor

    args_list = [
        (query_matrix, cand["matrix"], min_coverage)
        for cand in candidates
    ]

    # 串行回退（n_workers <= 1 或候选数太少）
    if n_workers <= 1 or len(candidates) <= 4:
        return [_dtw_process_one(args) for args in args_list]

    # ThreadPool 并行（numpy 运算释放 GIL）
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_dtw_process_one, args_list))

    return results


# ============================================================
# DTW 查询主函数
# ============================================================


def query_dtw(
    query_ts: str | pd.Timestamp,
    ref_df_resid: pd.DataFrame,
    feat: Any,
    dtw_cfg: DTWQueryConfig,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
    weight_sqrt: np.ndarray | None = None,
    time_col: str | None = None,
    alias_map: dict[str, str] | None = None,
    verbose: bool = True,
    n_workers: int = 4,
) -> PlanResult:
    """
    DTW 时序查询主函数（加权 cos 相似度）。

    流程：
        1. 解析查询时间戳，定位 ref_df 中的行号
        2. 截取参考窗口（t - ref_days天 ~ t）
        3. 接收标准化参数（norm_mean/norm_std/weight_sqrt）
        4. 提取查询序列（ref_window 末尾 query_seq_len 个点）和候选序列（dtw_max_len min 滑窗）
        5. DTW 对齐（cos 距离代价）+ cos 相似度均值 → 序列相似度
        6. Top-k 排序
        7. 生成规划中心

    参数：
        query_ts: 查询时间戳（字符串或 pd.Timestamp）
        ref_df_resid: 带残差特征的参考 DataFrame（分钟级，时间有序）
        feat: FeatureConfig
        dtw_cfg: DTWQueryConfig
        norm_mean: (D,) 全量数据均值（z-score 参数，从 dtw_norm_stats.json 加载）
        norm_std: (D,) 全量数据标准差
        weight_sqrt: (D,) √w 加权向量
        time_col: 时间列名（None=自动识别）
        alias_map: 列名别名映射
        verbose: 是否打印进度
        n_workers: 并行进程数

    返回：
        PlanResult
    """
    if verbose:
        print(f"\n[DTW] ===== DTW 查询 =====")
        print(f"[DTW] 查询时间戳: {query_ts}")

    # ---- 1. 解析时间戳，定位行号 ----
    time_col = _resolve_time_col("", time_col, ref_df_resid)

    # 避免 SettingWithCopyWarning：用 .copy() 确保独立
    ref_df_resid = ref_df_resid.copy()
    ref_df_resid[time_col] = pd.to_datetime(ref_df_resid[time_col], errors="coerce")
    query_ts = pd.to_datetime(query_ts)

    # 找到 query_ts 对应的行号
    time_mask = ref_df_resid[time_col] <= query_ts
    if time_mask.sum() == 0:
        raise ValueError(f"查询时间戳 {query_ts} 早于参考数据最早时间 {ref_df_resid[time_col].min()}")

    query_row_idx = int(time_mask.values.nonzero()[0][-1])
    if verbose:
        print(f"[DTW] 查询行索引: {query_row_idx}，时间: {ref_df_resid[time_col].iloc[query_row_idx]}")

    # ---- 2. 截取参考窗口 ----
    t_start = query_ts - pd.Timedelta(days=dtw_cfg.ref_days)
    ref_mask = (ref_df_resid[time_col] > t_start) & (ref_df_resid[time_col] <= query_ts)
    ref_window = ref_df_resid[ref_mask].copy().reset_index(drop=True)
    ref_n = len(ref_window)

    if verbose:
        print(f"[DTW] 参考窗口: {ref_window[time_col].min()} ~ {ref_window[time_col].max()}")
        print(f"[DTW] 参考窗口行数: {ref_n}")

    if ref_n < dtw_cfg.dtw_max_len + dtw_cfg.query_seq_len:
        raise ValueError(
            f"参考窗口行数 ({ref_n}) 不足以支撑最小候选序列长度 "
            f"({dtw_cfg.dtw_max_len}) + 查询序列 ({dtw_cfg.query_seq_len})"
        )

    # ---- 3. 标准化参数（z-score + √w 加权，从 dtw_norm_stats.json 加载）----
    all_feature_cols = feat.raw_features + [f"resid_{t}" for t in feat.residual_targets]
    sim_feature_cols = list(dict.fromkeys(all_feature_cols))  # 去重保序

    if norm_mean is None or norm_std is None or weight_sqrt is None:
        raise ValueError("DTW cos 相似度需要 norm_mean/norm_std/weight_sqrt，请先运行 compute_dtw_norm_stats.py 生成 dtw_norm_stats.json")

    col_mean = np.asarray(norm_mean, dtype=float)
    col_std = np.asarray(norm_std, dtype=float)
    w_sqrt = np.asarray(weight_sqrt, dtype=float)

    if verbose:
        print(f"[DTW] 特征数: {len(sim_feature_cols)}，cos 加权（√w 非零维度: {int(np.sum(w_sqrt > 1e-12))}）")

    # ---- 4. 提取查询序列（ref_window 末尾 query_seq_len 个点）----
    # 查询序列 = ref_window 末尾 query_seq_len 个点（不含 query_ts，已由 <= 条件排除）
    q_start = max(0, ref_n - dtw_cfg.query_seq_len)
    q_end = ref_n
    query_df = ref_window.iloc[q_start:q_end].copy()
    if len(query_df) < dtw_cfg.query_seq_len:
        pad_needed = dtw_cfg.query_seq_len - len(query_df)
        pad_df = ref_window.iloc[:pad_needed].copy()
        query_df = pd.concat([pad_df, query_df], ignore_index=True)

    query_matrix = query_df[sim_feature_cols].values.astype(float)  # (T_q, D)

    # ---- 4.5 填充 NaN 值 + 标准化 + 加权 ----
    # 某些列（如 吨煤产气量）可能有 NaN，用全量均值填充
    for i in range(query_matrix.shape[1]):
        if np.isnan(query_matrix[:, i]).any():
            if pd.notna(col_mean[i]):
                query_matrix[np.isnan(query_matrix[:, i]), i] = col_mean[i]
                if verbose:
                    print(f"[DTW] 填充 NaN: {sim_feature_cols[i]} -> 均值 {col_mean[i]:.4f}")

    # 标准化 + 加权
    query_matrix = ((query_matrix - col_mean) / col_std) * w_sqrt  # (T_q, D)

    # ---- 4.6 负荷初筛（主汽流量阈值筛选连续片段）----
    pre_cfg = dtw_cfg.prefilter
    if pre_cfg.enable:
        q_load_mean = float(query_df[feat.load_col].mean())
        segments = load_filter_segments(
            ref_window=ref_window,
            load_col=feat.load_col,
            q_load_mean=q_load_mean,
            threshold=pre_cfg.load_threshold,
            min_length=dtw_cfg.query_seq_len,
        )
        if verbose:
            total_seg_rows = sum(e - s for s, e in segments)
            print(f"[DTW] 负荷初筛: 查询平均负荷 {q_load_mean:.2f} t/h，阈值 ±{pre_cfg.load_threshold} t/h")
            print(f"[DTW]   连续片段 {len(segments)} 个，共 {total_seg_rows} 行（参考窗口 {ref_n} 行）")

        if not segments:
            # 负荷初筛全部被筛掉，回退到无预筛模式
            if verbose:
                print(f"[DTW] 负荷初筛无通过片段，回退到无预筛模式")
            segments = [(0, ref_n)]
    else:
        segments = [(0, ref_n)]

    # ---- 4.7 cos 相似度预筛 ----
    # 注意：cos_prefilter 内部用原始 query_matrix 做标准化，故传入未标准化的副本
    # query_matrix 此时已是标准化+加权后的，需要传原始值给 cos_prefilter
    query_matrix_raw = query_df[sim_feature_cols].values.astype(float)
    # 填充 NaN
    for i in range(query_matrix_raw.shape[1]):
        if np.isnan(query_matrix_raw[:, i]).any():
            if pd.notna(col_mean[i]):
                query_matrix_raw[np.isnan(query_matrix_raw[:, i]), i] = col_mean[i]

    if pre_cfg.enable and segments != [(0, ref_n)]:
        passed_starts = cos_prefilter(
            query_matrix=query_matrix_raw,
            ref_window=ref_window,
            segments=segments,
            sim_feature_cols=sim_feature_cols,
            cos_threshold=pre_cfg.cos_threshold,
            cos_slide_step=pre_cfg.cos_slide_step,
            col_mean=col_mean,
            col_std=col_std,
            weight_sqrt=w_sqrt,
            verbose=verbose,
        )

        if not passed_starts:
            # cos 预筛全部被筛掉，回退到负荷初筛结果（用片段起点）
            if verbose:
                print(f"[DTW] cos 预筛无通过窗口，回退到负荷初筛片段起点")
            passed_starts = [s for s, e in segments if e - s >= dtw_cfg.query_seq_len]
    else:
        # 无预筛模式：用 slide_step 生成全部起点
        passed_starts = list(range(0, ref_n - dtw_cfg.query_seq_len + 1, dtw_cfg.slide_step))

    # ---- 5. 构建候选矩阵（固定用 dtw_max_len=6 分钟，DTW 弹性对齐匹配 4~6）----
    candidates: list[dict] = []
    cand_length = dtw_cfg.dtw_max_len  # 固定用最大长度，DTW 弹性对齐代替多长度

    # 查询序列位于 ref_window 末尾 query_seq_len 个点（行索引 [ref_n - query_seq_len, ref_n)）
    # 候选窗口不得与查询序列重叠：要求 end <= ref_n - query_seq_len
    query_region_start = ref_n - dtw_cfg.query_seq_len

    # 预计算整个 ref_window 的标准化+加权矩阵（一次性，候选直接切片复用）
    ref_features = ref_window[sim_feature_cols].values.astype(float)  # (ref_n, D)
    # 填充 NaN（用全量均值）
    if np.isnan(ref_features).any():
        inds = np.where(np.isnan(ref_features))
        ref_features[inds] = np.take(col_mean, inds[1])
    # z-score 标准化 + 加权
    ref_zw = ((ref_features - col_mean) / col_std) * w_sqrt  # (ref_n, D)

    # 候选窗口起点偏移量（ref_df_resid 中 t_start 之前的行数）
    global_offset = int((ref_df_resid[time_col] <= t_start).sum())

    for start in passed_starts:
        end = start + cand_length
        if end > query_region_start:
            continue  # 排除与查询序列重叠的候选

        # 直接从预计算的标准化矩阵切片（O(1)，无重复标准化）
        mat = ref_zw[start:end]  # (cand_length, D)
        global_start = global_offset + start
        candidates.append({
            "orig_start_idx": global_start,
            "orig_end_idx": global_start + cand_length,
            "length": cand_length,
            "matrix": mat,
        })

    if verbose:
        total_cands = len(candidates)
        print(f"[DTW] 候选序列数: {total_cands} "
              f"（预筛通过 {len(passed_starts)} 起点 × 1 长度（{cand_length}min，DTW弹性对齐4~6））")

    # ---- 6. DTW 对齐 + 覆盖帧检查 + cos 相似度均值（并行）----
    if verbose:
        print(f"[DTW] DTW 对齐（cos 距离）+ cos 相似度计算中（{n_workers} 线程并行）...")

    # 并行计算
    dtw_results = _parallel_dtw_align(
        query_matrix=query_matrix,
        candidates=candidates,
        min_coverage=dtw_cfg.dtw_min_len,
        n_workers=n_workers,
    )

    # 收集结果
    sim_scores = []
    for i, cand in enumerate(candidates):
        r = dtw_results[i]
        sim_scores.append(r["sim"])
        cand["dtw_cost"] = r["path_cost"]
        cand["path_length"] = r["path_length"]

    sim_scores = np.array(sim_scores, dtype=float)

    # ---- 7. Top-k ----
    top_k = min(dtw_cfg.top_k, len(sim_scores))
    top_indices = np.argsort(-sim_scores)[:top_k]  # 降序

    if verbose:
        print(f"[DTW] Top-{top_k} 相似度:")
        for rank, idx in enumerate(top_indices):
            cand = candidates[idx]
            print(f"[DTW]   #{rank + 1}: idx={idx}, start={cand['orig_start_idx']}, "
                  f"len={cand['length']}, sim={sim_scores[idx]:.4f}")

    # ---- 8. 生成规划中心 ----
    plan_center_cols = feat.plan_center_cols
    weights_dict = build_feature_weights(feat)

    # 加权均值：权重 = 相似度（截断防0）
    d_values = sim_scores[top_indices]
    d_values = np.clip(d_values, 0.001, None)
    d_weights = d_values / d_values.sum()

    # 从原始参考数据取 plan_center_cols（末帧值）
    raw_plan_center: dict[str, float] = {}
    for c in plan_center_cols:
        col_name = c
        if alias_map and c in alias_map:
            col_name = alias_map[c]

        vals = []
        for idx in top_indices:
            cand = candidates[idx]
            end_idx = cand["orig_end_idx"]
            if end_idx <= len(ref_df_resid) and col_name in ref_df_resid.columns:
                vals.append(float(ref_df_resid[col_name].iloc[end_idx - 1]))

        if vals:
            raw_plan_center[c] = float(np.average(vals, weights=d_weights))

    # ---- 8.5 计算 DTW 特有诊断信息 ----
    # DTW 路径代价（最佳候选）
    best_cand = candidates[top_indices[0]] if len(top_indices) > 0 else None
    dtw_path_cost_best = float(best_cand["dtw_cost"]) if best_cand and "dtw_cost" in best_cand else np.nan

    # 候选序列数量
    n_candidates = len(candidates)

    # 路径长度（最佳候选）
    dtw_path_length_best = int(best_cand["path_length"]) if best_cand and "path_length" in best_cand else 0

    # 时间偏移（最佳候选末帧时间 - 查询时间，单位：天）
    time_offset_days = np.nan
    if best_cand and time_col in ref_df_resid.columns:
        try:
            cand_end_idx = best_cand["orig_end_idx"]
            if cand_end_idx <= len(ref_df_resid):
                cand_end_time = pd.to_datetime(ref_df_resid[time_col].iloc[cand_end_idx - 1])
                time_offset_days = abs((cand_end_time - query_ts).total_seconds()) / 86400.0
        except Exception:
            time_offset_days = np.nan

    # ---- 9. 构建 PlanResult ----
    result = PlanResult(
        raw_plan_center=raw_plan_center,
        final_plan_center=dict(raw_plan_center),
        match_status="DTW时序匹配",
        topk_indices=[int(candidates[idx]["orig_start_idx"]) for idx in top_indices],
        best_index=int(candidates[top_indices[0]]["orig_start_idx"]) if len(top_indices) > 0 else None,
        topk_count=top_k,
        similarity_best=float(sim_scores[top_indices[0]]) if len(top_indices) > 0 else np.nan,
        similarity_topk_mean=float(np.mean(sim_scores[top_indices])) if len(top_indices) > 0 else np.nan,
        score_d_best=float(sim_scores[top_indices[0]]) if len(top_indices) > 0 else np.nan,
        score_d_topk_mean=float(np.mean(sim_scores[top_indices])) if len(top_indices) > 0 else np.nan,
        eff_score_best=np.nan,
        eff_topk_mean=np.nan,
        low_sim_fallback=False,
        fallback_threshold=np.nan,
        plan_center_source="DTW时序匹配",
        continuity_status="DTW（无连续性处理）",
        continuity_reset=False,
        rate_limited_features=[],
        smoothed_features=[],
    )

    # ---- 9.5 附加 DTW 特有诊断字段（动态属性） ----
    result._dtw_path_cost = dtw_path_cost_best
    result._n_candidates = n_candidates
    result._dtw_path_length = dtw_path_length_best
    result._time_offset_days = time_offset_days

    if verbose:
        print(f"\n[DTW] 规划中心（Top-{top_k} 加权均值）:")
        for c, v in raw_plan_center.items():
            print(f"[DTW]   {c}: {v:.2f}")
        print(f"[DTW] ===== DTW 查询完成 =====\n")

    return result


# ============================================================
# DTWQueryEngine 类
# ============================================================


@dataclass
class DTWQueryEngine:
    """
    DTW 时序查询引擎（加权 cos 相似度）。

    持有：
        - cfg: PlanningConfig（含 DTWQueryConfig）
        - ref_df_resid: 带残差特征的分钟级参考 DataFrame
        - models: 残差模型字典
        - norm_mean / norm_std: (D,) z-score 标准化参数（从 dtw_norm_stats.json 加载）
        - weight_sqrt: (D,) √w 加权向量
    """

    cfg: PlanningConfig
    ref_df_resid: pd.DataFrame | None = None
    models: dict[str, object] | None = None
    norm_mean: np.ndarray | None = None
    norm_std: np.ndarray | None = None
    weight_sqrt: np.ndarray | None = None
    n_workers: int = 4
    _cache_loaded: bool = field(default=False, init=False, repr=False)

    # ---- 初始化 / 加载 ----

    def _ensure_data_loaded(self) -> None:
        """懒加载：首次调用时加载参考数据 + 残差模型 + 标准化参数。"""
        if self._cache_loaded:
            return

        if self.cfg.dtw_query is None:
            raise RuntimeError("PlanningConfig 中缺少 dtw_query 配置段")

        dtw_cfg = self.cfg.dtw_query
        feat = self.cfg.features
        paths = self.cfg.paths

        # 1. 残差缓存
        cache_path = ensure_resid_cache(
            raw_parquet=paths.query_parquet,
            model_dir=paths.residual_model_dir,
            feat=feat,
            cache_parquet=dtw_cfg.resid_cache_parquet,
            feature_cols=feat.raw_features,
            alias_map=feat.column_aliases,
        )
        self.ref_df_resid = pd.read_parquet(cache_path)

        # 2. 加载残差模型
        self.models = load_residual_models(paths.residual_model_dir, feat.residual_targets)

        # 3. 标准化参数（z-score，从 dtw_norm_stats.json 加载）
        if paths.dtw_norm_stats_path and Path(paths.dtw_norm_stats_path).exists():
            import json
            with open(paths.dtw_norm_stats_path, "r", encoding="utf-8") as f:
                norm_data = json.load(f)
            sim_feature_cols = list(dict.fromkeys(
                feat.raw_features + [f"resid_{t}" for t in feat.residual_targets]
            ))
            self.norm_mean = np.array([norm_data["mean"][c] for c in sim_feature_cols], dtype=float)
            self.norm_std = np.array([norm_data["std"][c] for c in sim_feature_cols], dtype=float)
            print(f"[DTW] 标准化参数已加载: {paths.dtw_norm_stats_path}，维度 {self.norm_mean.shape[0]}")
        else:
            raise RuntimeError(
                f"未找到标准化参数文件: {paths.dtw_norm_stats_path}，请先运行 compute_dtw_norm_stats.py 生成"
            )

        # 4. √w 加权向量
        weights = build_feature_weights(feat)
        self.weight_sqrt = np.array([np.sqrt(max(weights.get(c, 0.0), 0.0)) for c in sim_feature_cols], dtype=float)
        n_nonzero = int(np.sum(self.weight_sqrt > 1e-12))
        print(f"[DTW] √w 加权向量已构建，非零维度: {n_nonzero}/{len(sim_feature_cols)}")

        self._cache_loaded = True

    def query_one(self, query_ts: str | pd.Timestamp, verbose: bool = True) -> PlanResult:
        """
        单次 DTW 查询（加权 cos 相似度）。

        参数：
            query_ts: 查询时间戳
            verbose: 是否打印进度

        返回：
            PlanResult
        """
        self._ensure_data_loaded()

        if self.ref_df_resid is None or self.models is None:
            raise RuntimeError("DTWQueryEngine 数据未正确加载")

        return query_dtw(
            query_ts=query_ts,
            ref_df_resid=self.ref_df_resid,
            feat=self.cfg.features,
            dtw_cfg=self.cfg.dtw_query,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            weight_sqrt=self.weight_sqrt,
            time_col=self.cfg.time_col,
            alias_map=self.cfg.features.column_aliases,
            verbose=verbose,
            n_workers=self.n_workers,
        )
