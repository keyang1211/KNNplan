# -*- coding: utf-8 -*-
"""
dtw_query.py — DTW 时序查询核心

新增独立查询路径，不修改现有 PlanningEngine / query.py。
相似度度量：加权马氏距离 + 柯西核（与稳定工况查询路径一致）
    - 标准化：通过加权协方差逆矩阵 M 隐式标准化
    - 加权：M = W^(1/2) Σ⁻¹ W^(1/2)（权重0的特征自动排除）
    - DTW 对齐代价 = 马氏距离 d=sqrt((q-x)ᵀM(q-x))
    - 对齐后相似度 = 柯西核 S=1/(1+d²) 均值，映射到 (0,1]
    - 协方差逆矩阵 M 复用 paths.covariance_path（与稳定工况路径共用）
依赖：
    - plan_center.features（add_residual_features / load_residual_models）
    - plan_center.config（DTWQueryConfig / PlanningConfig / build_feature_weights）
    - plan_center.schemas（PlanResult / plan_result_to_row / build_output_dataframe）
    - plan_center.batch（_resolve_time_col）

用法：
    from plan_center.dtw_query import DTWQueryEngine, query_dtw
    engine = DTWQueryEngine(cfg)
    # 调用方负责准备最近 ref_days 天的数据（含残差特征）
    # ref_df 末尾 query_seq_len 行是查询序列，前面部分是参考窗口
    ref_df = ...  # pd.DataFrame
    result = engine.query_one(ref_df)
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

sys.stdout.reconfigure(encoding="utf-8")

from .config import DTWQueryConfig, PlanningConfig, build_feature_weights
from .features import add_residual_features, load_residual_models
from .schemas import PlanResult
from .batch import _resolve_time_col


# ============================================================
# 工具：DTW 对齐（加权马氏距离代价，numpy 手动实现）
# ============================================================


def dtw_align_mahalanobis(
    query_seq: np.ndarray,
    cand_seq: np.ndarray,
    M: np.ndarray,
    sakoe_chiba_w: int = 0,
) -> tuple[list[tuple[int, int]], float, np.ndarray]:
    """
    标准 DTW 动态规划对齐（马氏距离代价），支持 Sakoe-Chiba 带约束。

    输入序列为原始特征（不做 z-score，不做 √w 加权），标准化和加权通过 M 矩阵隐式完成。

    参数：
        query_seq: (T_q, D) 查询序列（原始特征）
        cand_seq: (T_c, D) 候选序列（原始特征）
        M: (D, D) 加权协方差逆矩阵
        sakoe_chiba_w: Sakoe-Chiba 带宽（0=无约束全矩阵，>=1 限制 |i-j|<=w）

    返回：
        (aligned_pairs, path_cost, dist_matrix)
        aligned_pairs: [(i_q, i_c), ...]，正序
        path_cost: 最小累积代价（DTW 距离）
        dist_matrix: (T_q, T_c) 逐点马氏距离矩阵，供 dtw_mahalanobis_mean 复用
    """
    n, m = query_seq.shape[0], cand_seq.shape[0]

    # 预计算逐点马氏距离矩阵 (n, m)
    # d²[i,j] = (q[i]-x[j])ᵀ M (q[i]-x[j])
    # 向量化展开：d² = qMq - 2 qMx + xMx
    Mq = M @ query_seq.T                              # (D, n)
    Mx = M @ cand_seq.T                               # (D, m)
    qMq = np.sum(query_seq.T * Mq, axis=0)            # (n,)  qᵀMq
    xMx = np.sum(cand_seq.T * Mx, axis=0)             # (m,)  xᵀMx
    qMx = query_seq @ Mx                              # (n, m) qᵀMx
    d_sq = qMq[:, None] - 2.0 * qMx + xMx[None, :]    # (n, m)
    d_sq = np.maximum(d_sq, 0.0)                       # 数值保护
    dist_matrix = np.sqrt(d_sq)                        # (n, m) 马氏距离 d

    # DP 成本矩阵
    D = np.full((n, m), np.inf, dtype=float)
    D[0, 0] = dist_matrix[0, 0]

    if sakoe_chiba_w <= 0:
        # 无约束（全矩阵）
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
    else:
        # Sakoe-Chiba 带约束：只计算 |i-j| <= w 的单元
        # 安全保护：w 至少为 |n-m|，确保终点 (n-1, m-1) 可达
        w = max(sakoe_chiba_w, abs(n - m))
        # 首行：j ∈ [1, w]（|0-j|<=w → j<=w）
        for j in range(1, min(w + 1, m)):
            D[0, j] = D[0, j - 1] + dist_matrix[0, j]
        # 首列：i ∈ [1, w]
        for i in range(1, min(w + 1, n)):
            D[i, 0] = D[i - 1, 0] + dist_matrix[i, 0]
        # 内点：j ∈ [max(1, i-w), min(m, i+w+1)]
        for i in range(1, n):
            j_start = max(1, i - w)
            j_end = min(m, i + w + 1)
            for j in range(j_start, j_end):
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
    return aligned_pairs, float(D[-1, -1]), dist_matrix


def dtw_align_with_coverage_mahalanobis(
    query_seq: np.ndarray,
    cand_seq: np.ndarray,
    M: np.ndarray,
    min_coverage: int = 4,
    sakoe_chiba_w: int = 0,
) -> tuple[list[tuple[int, int]], float, bool, np.ndarray]:
    """
    DTW 对齐 + 候选覆盖帧检查（马氏距离代价）。

    比 dtw_align_mahalanobis() 多返回一个 bool: 路径中涉及的不重复候选帧数 >= min_coverage。

    参数：
        query_seq: (T_q, D) 查询序列（原始特征）
        cand_seq: (T_c, D) 候选序列（原始特征）
        M: (D, D) 加权协方差逆矩阵
        min_coverage: 最少不重复候选帧数（默认 4）
        sakoe_chiba_w: Sakoe-Chiba 带宽（0=无约束，>=1 限制 |i-j|<=w）

    返回：
        (aligned_pairs, path_cost, coverage_ok, dist_matrix)
    """
    aligned_pairs, path_cost, dist_matrix = dtw_align_mahalanobis(
        query_seq, cand_seq, M=M, sakoe_chiba_w=sakoe_chiba_w
    )
    # 计算路径覆盖的不重复候选帧数
    cand_indices = np.array([j for _, j in aligned_pairs])
    n_unique = int(np.unique(cand_indices).size)
    coverage_ok = n_unique >= min_coverage
    return aligned_pairs, path_cost, coverage_ok, dist_matrix


# ============================================================
# 工具：DTW 对齐后马氏距离相似度均值（柯西核，映射到 (0,1]）
# ============================================================


def dtw_mahalanobis_mean(
    dist_matrix: np.ndarray,
    aligned_pairs: list[tuple[int, int]],
) -> float:
    """
    对齐后从 dist_matrix 直接取值，柯西核 S=1/(1+d²) 映射到 (0,1]，取均值。

    参数：
        dist_matrix: (T_q, T_c) 逐点马氏距离矩阵（由 dtw_align_mahalanobis 预计算）
        aligned_pairs: DTW 对齐索引对 [(i_q, i_c), ...]

    返回：
        float，均值相似度 (0, 1]
    """
    if not aligned_pairs:
        return 0.0

    # 直接从 dist_matrix 取对齐点对的马氏距离
    d_vals = np.array([dist_matrix[i, j] for i, j in aligned_pairs])

    # 柯西核映射到 (0, 1] 后取均值
    sims = 1.0 / (1.0 + d_vals ** 2)
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
# 预筛辅助函数：负荷初筛 + 马氏距离预筛
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


def mahalanobis_prefilter(
    query_matrix: np.ndarray,       # (T_q, D) 查询序列（原始特征）
    ref_window: pd.DataFrame,        # 参考窗口
    segments: list[tuple[int, int]], # 负荷初筛通过的片段
    sim_feature_cols: list[str],     # 特征列名
    sim_threshold: float,            # 柯西核相似度阈值 S=1/(1+d²)
    slide_step: int,                 # 滑窗步长
    M: np.ndarray,                   # (D, D) 加权协方差逆矩阵
    verbose: bool = True,
) -> list[int]:
    """
    马氏距离预筛：在负荷初筛通过的片段上，等长滑窗 + 逐点马氏距离 + 柯西核相似度均值。

    向量化实现：整个片段一次性计算 (T_q, seg_len) 逐点马氏距离矩阵，
    所有滑窗只需取对角线均值（O(T_q) per window）。

    参数：
        query_matrix: (T_q, D) 查询序列（原始特征）
        ref_window: 参考窗口 DataFrame
        segments: 负荷初筛通过的片段 [(start, end), ...]
        sim_feature_cols: 特征列名列表
        sim_threshold: 柯西核相似度阈值，低于此值的窗口删掉
        slide_step: 滑窗步长（分钟）
        M: (D, D) 加权协方差逆矩阵
        verbose: 是否打印进度

    返回：
        通过马氏距离筛选的窗口起点索引列表（基于 ref_window 的行索引）
    """
    T_q, D = query_matrix.shape

    # 1. 预计算查询序列的 Mq 和 qMq（对所有片段复用）
    Mq = M @ query_matrix.T                    # (D, T_q)
    qMq = np.sum(query_matrix.T * Mq, axis=0)  # (T_q,) qᵀMq

    # 2. 在每个片段内批量计算所有滑窗的逐点马氏距离
    passed_starts: list[int] = []
    n_checked = 0
    n_passed = 0

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < T_q:
            continue

        # 提取片段特征矩阵（原始特征）
        seg_features = ref_window[sim_feature_cols].iloc[seg_start:seg_end].values.astype(float)
        # 填充 NaN 为 0（马氏距离用）
        seg_features = np.nan_to_num(seg_features, nan=0.0)

        # 向量化逐点马氏距离矩阵 (T_q, seg_len)
        # d²[i,j] = qMq[i] - 2 qMx[i,j] + xMx[j]
        Mx = M @ seg_features.T                          # (D, seg_len)
        xMx = np.sum(seg_features.T * Mx, axis=0)        # (seg_len,) xᵀMx
        qMx = query_matrix @ Mx                           # (T_q, seg_len) qᵀMx
        d_sq = qMq[:, None] - 2.0 * qMx + xMx[None, :]    # (T_q, seg_len)
        d_sq = np.maximum(d_sq, 0.0)                      # 数值保护

        # 柯西核相似度矩阵 (T_q, seg_len)
        sim_matrix = 1.0 / (1.0 + d_sq)  # S=1/(1+d²) ∈ (0, 1]

        # 滑窗：每个窗口取对角线均值
        n_windows = seg_len - T_q + 1
        for win_start in range(0, n_windows, slide_step):
            # 对角线元素 sim_matrix[t, win_start+t] for t in range(T_q)
            diag_idx = np.arange(T_q) + win_start
            sim_vals = sim_matrix[np.arange(T_q), diag_idx]  # (T_q,)
            sim_mean = float(np.mean(sim_vals))

            n_checked += 1
            if sim_mean >= sim_threshold:
                passed_starts.append(seg_start + win_start)
                n_passed += 1

    if verbose:
        print(f"[DTW] 马氏距离预筛: 检查 {n_checked} 个窗口，通过 {n_passed} 个（阈值 {sim_threshold}）")

    return passed_starts


# ============================================================
# 并行 DTW 对齐辅助函数
# ============================================================


def _dtw_process_one(args: tuple) -> dict:
    """
    模块级 worker 函数，供 ThreadPoolExecutor/ProcessPoolExecutor 调用。
    参数通过 tuple 传递以支持 pickle 序列化。
    """
    query_matrix, cand_matrix, M, min_coverage, sakoe_chiba_w = args
    aligned_pairs, path_cost, coverage_ok, dist_matrix = dtw_align_with_coverage_mahalanobis(
        query_matrix, cand_matrix, M=M, min_coverage=min_coverage, sakoe_chiba_w=sakoe_chiba_w
    )
    if not coverage_ok:
        sim = 0.0
    else:
        sim = dtw_mahalanobis_mean(dist_matrix, aligned_pairs)
    return {
        "sim": sim,
        "path_cost": path_cost,
        "path_length": len(aligned_pairs),
        "coverage_ok": coverage_ok,
    }


def _parallel_dtw_align(
    query_matrix: np.ndarray,
    candidates: list[dict],
    M: np.ndarray,
    min_coverage: int,
    sakoe_chiba_w: int = 0,
    n_workers: int = 4,
) -> list[dict]:
    """
    并行计算多个候选的 DTW 对齐 + 马氏距离相似度。

    使用 ThreadPoolExecutor，配合 numpy 矩阵运算释放 GIL 可部分并行。
    相比 ProcessPool 无 pickle 序列化开销，适合小任务。

    参数：
        query_matrix: (T_q, D) 查询序列（原始特征）
        candidates: 候选列表，每个含 "matrix" 字段（原始特征）
        M: (D, D) 加权协方差逆矩阵
        min_coverage: 最小覆盖帧数
        sakoe_chiba_w: Sakoe-Chiba 带宽（0=无约束，>=1 限制 |i-j|<=w）
        n_workers: 并行线程数

    返回：
        [{"sim", "path_cost", "path_length", "coverage_ok"}, ...]
    """
    from concurrent.futures import ThreadPoolExecutor

    args_list = [
        (query_matrix, cand["matrix"], M, min_coverage, sakoe_chiba_w)
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
    ref_df: pd.DataFrame,
    feat: Any,
    dtw_cfg: DTWQueryConfig,
    cov_inv_matrix: np.ndarray,
    query_ts: str | pd.Timestamp | None = None,
    time_col: str | None = None,
    alias_map: dict[str, str] | None = None,
    verbose: bool = True,
    n_workers: int = 4,
) -> PlanResult:
    """
    DTW 时序查询主函数（加权马氏距离 + 柯西核）。

    调用方负责准备 ref_df：最近 N 天数据（含残差特征），末尾 query_seq_len 行是查询序列。
    引擎不截取时间窗口，直接用 ref_df 前面部分作为参考窗口，末尾作为查询序列。

    流程：
        1. 解析 time_col，拆分 ref_df → ref_window（前面）+ query_df（末尾 query_seq_len 行）
        2. 接收加权协方差逆矩阵 M（从 covariance.json 加载，复用稳定工况路径）
        3. 负荷初筛 + 马氏距离预筛
        4. 构建候选矩阵（dtw_max_len min 滑窗）
        5. DTW 对齐（马氏距离代价）+ 柯西核相似度均值 → 序列相似度
        6. Top-k 排序 + 生成规划中心

    参数：
        ref_df: 完整窗口数据（参考+查询序列），末尾 query_seq_len 行是查询序列
        feat: FeatureConfig
        dtw_cfg: DTWQueryConfig
        cov_inv_matrix: (D, D) 加权协方差逆矩阵 M（从 covariance.json 加载）
        query_ts: 查询时间戳（可选，仅用于诊断；None 时用 ref_df 末尾时间）
        time_col: 时间列名（None=自动识别）
        alias_map: 列名别名映射
        verbose: 是否打印进度
        n_workers: 并行进程数

    返回：
        PlanResult
    """
    # ---- 1. 解析 time_col，拆分 ref_df ----
    time_col = _resolve_time_col("", time_col, ref_df)

    # 避免 SettingWithCopyWarning
    ref_df = ref_df.copy()
    ref_df[time_col] = pd.to_datetime(ref_df[time_col], errors="coerce")

    ref_n_total = len(ref_df)
    q_len = dtw_cfg.query_seq_len

    if ref_n_total < q_len + dtw_cfg.dtw_max_len:
        raise ValueError(
            f"ref_df 行数 ({ref_n_total}) 不足以支撑查询序列 ({q_len}) + 最小候选长度 ({dtw_cfg.dtw_max_len})"
        )

    # 查询序列 = ref_df 末尾 query_seq_len 行；参考窗口 = 前面部分
    ref_window = ref_df.iloc[:-q_len].copy().reset_index(drop=True)
    query_df = ref_df.iloc[-q_len:].copy().reset_index(drop=True)
    ref_n = len(ref_window)

    # query_ts 处理：None 时用 ref_df 末尾时间
    if query_ts is None:
        query_ts = ref_df[time_col].iloc[-1]
    else:
        query_ts = pd.to_datetime(query_ts)

    if verbose:
        print(f"\n[DTW] ===== DTW 查询 =====")
        print(f"[DTW] 查询时间戳: {query_ts}")
        print(f"[DTW] ref_df 行数: {ref_n_total}（参考窗口 {ref_n} + 查询序列 {q_len}）")
        print(f"[DTW] 参考窗口范围: {ref_window[time_col].min()} ~ {ref_window[time_col].max()}")

    # ---- 3. 加权协方差逆矩阵 M（从 covariance.json 加载，复用稳定工况路径）----
    all_feature_cols = feat.raw_features + [f"resid_{t}" for t in feat.residual_targets]
    sim_feature_cols = list(dict.fromkeys(all_feature_cols))  # 去重保序

    if cov_inv_matrix is None:
        raise ValueError("DTW 马氏距离需要 cov_inv_matrix，请先运行 train_residual.py 生成 covariance.json")

    M = np.asarray(cov_inv_matrix, dtype=float)

    if verbose:
        print(f"[DTW] 特征数: {len(sim_feature_cols)}，M 矩阵 shape: {M.shape}")

    # ---- 4. 提取查询序列特征矩阵 ----
    # query_df 已在 Step 1 从 ref_df 末尾拆分（query_seq_len 行）
    query_matrix = query_df[sim_feature_cols].values.astype(float)  # (T_q, D)

    # ---- 4.5 填充 NaN 值（用 0.0 填充，马氏距离用）----
    query_matrix = np.nan_to_num(query_matrix, nan=0.0)

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

    # ---- 4.7 马氏距离预筛 ----
    if pre_cfg.enable and segments != [(0, ref_n)]:
        passed_starts = mahalanobis_prefilter(
            query_matrix=query_matrix,    # 原始特征（已 NaN→0）
            ref_window=ref_window,
            segments=segments,
            sim_feature_cols=sim_feature_cols,
            sim_threshold=pre_cfg.sim_threshold,
            slide_step=pre_cfg.slide_step,
            M=M,
            verbose=verbose,
        )

        if not passed_starts:
            # 马氏距离预筛全部被筛掉，回退到负荷初筛结果（用片段起点）
            if verbose:
                print(f"[DTW] 马氏距离预筛无通过窗口，回退到负荷初筛片段起点")
            passed_starts = [s for s, e in segments if e - s >= dtw_cfg.query_seq_len]
    else:
        # 无预筛模式：用 slide_step 生成全部起点
        passed_starts = list(range(0, ref_n - dtw_cfg.query_seq_len + 1, dtw_cfg.slide_step))

    # ---- 5. 构建候选矩阵（固定用 dtw_max_len=6 分钟，DTW 弹性对齐匹配 4~6）----
    candidates: list[dict] = []
    cand_length = dtw_cfg.dtw_max_len  # 固定用最大长度，DTW 弹性对齐代替多长度

    # 查询序列已从 ref_window 拆分出去，候选窗口可在整个 ref_window 内滑动
    # 候选窗口 end 必须 <= ref_n（不进入查询序列区域）
    query_region_start = ref_n

    # 预计算整个 ref_window 的原始特征矩阵（一次性，候选直接切片复用）
    ref_features = ref_window[sim_feature_cols].values.astype(float)  # (ref_n, D)
    ref_features = np.nan_to_num(ref_features, nan=0.0)

    # orig_start_idx / orig_end_idx 直接用 ref_df 的局部行号（ref_window 是 ref_df 的前 ref_n 行，索引一致）
    for start in passed_starts:
        end = start + cand_length
        if end > query_region_start:
            continue  # 排除越界候选

        # 直接从预计算的原始特征矩阵切片（O(1)）
        mat = ref_features[start:end]  # (cand_length, D)
        candidates.append({
            "orig_start_idx": start,
            "orig_end_idx": end,
            "length": cand_length,
            "matrix": mat,
        })

    if verbose:
        total_cands = len(candidates)
        print(f"[DTW] 候选序列数: {total_cands} "
              f"（预筛通过 {len(passed_starts)} 起点 × 1 长度（{cand_length}min，DTW弹性对齐4~6））")

    # ---- 6. DTW 对齐 + 覆盖帧检查 + 马氏距离相似度均值（并行）----
    if verbose:
        print(f"[DTW] DTW 对齐（马氏距离）+ 柯西核相似度计算中（{n_workers} 线程并行，Sakoe-Chiba w={dtw_cfg.sakoe_chiba_w}）...")

    # 并行计算
    dtw_results = _parallel_dtw_align(
        query_matrix=query_matrix,
        candidates=candidates,
        M=M,
        min_coverage=dtw_cfg.min_coverage,
        sakoe_chiba_w=dtw_cfg.sakoe_chiba_w,
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
            if end_idx <= len(ref_df) and col_name in ref_df.columns:
                vals.append(float(ref_df[col_name].iloc[end_idx - 1]))

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
    if best_cand and time_col in ref_df.columns:
        try:
            cand_end_idx = best_cand["orig_end_idx"]
            if cand_end_idx <= len(ref_df):
                cand_end_time = pd.to_datetime(ref_df[time_col].iloc[cand_end_idx - 1])
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
    DTW 时序查询引擎（加权马氏距离 + 柯西核）。

    持有：
        - cfg: PlanningConfig（含 DTWQueryConfig）
        - cov_inv_matrix: (D, D) 加权协方差逆矩阵 M（从 covariance.json 加载，复用稳定工况路径）
        - models: 残差模型字典（供 ensure_resid_cache 工具函数用，query_one 不依赖）

    调用方负责准备 ref_df（最近 N 天数据，含残差特征），通过 query_one(ref_df) 传入。
    引擎不持有全量数据，不截取时间窗口。
    """

    cfg: PlanningConfig
    cov_inv_matrix: np.ndarray | None = None
    models: dict[str, object] | None = None
    n_workers: int = 4
    _cache_loaded: bool = field(default=False, init=False, repr=False)

    # ---- 初始化 / 加载 ----

    def _ensure_data_loaded(self) -> None:
        """懒加载：首次调用时加载加权协方差逆矩阵 M。"""
        if self._cache_loaded:
            return

        if self.cfg.dtw_query is None:
            raise RuntimeError("PlanningConfig 中缺少 dtw_query 配置段")

        paths = self.cfg.paths

        # 加权协方差逆矩阵 M（从 covariance.json 加载，复用稳定工况路径）
        if paths.covariance_path and Path(paths.covariance_path).exists():
            import json
            with open(paths.covariance_path, "r", encoding="utf-8") as f:
                cov_data = json.load(f)
            self.cov_inv_matrix = np.array(cov_data["cov_inv_matrix"], dtype=np.float64)
            print(f"[DTW] M 矩阵已加载: {paths.covariance_path}，shape: {self.cov_inv_matrix.shape}")
        else:
            raise RuntimeError(
                f"未找到协方差矩阵文件: {paths.covariance_path}，请先运行 train_residual.py 生成"
            )

        self._cache_loaded = True

    def query_one(
        self,
        ref_df: pd.DataFrame,
        query_ts: str | pd.Timestamp | None = None,
        verbose: bool = True,
    ) -> PlanResult:
        """
        单次 DTW 查询（加权马氏距离 + 柯西核）。

        参数：
            ref_df: 完整窗口数据（参考+查询序列），末尾 query_seq_len 行是查询序列
            query_ts: 查询时间戳（可选，仅用于诊断；None 时用 ref_df 末尾时间）
            verbose: 是否打印进度

        返回：
            PlanResult
        """
        self._ensure_data_loaded()

        if self.cov_inv_matrix is None:
            raise RuntimeError("DTWQueryEngine 数据未正确加载")

        return query_dtw(
            ref_df=ref_df,
            feat=self.cfg.features,
            dtw_cfg=self.cfg.dtw_query,
            cov_inv_matrix=self.cov_inv_matrix,
            query_ts=query_ts,
            time_col=self.cfg.time_col,
            alias_map=self.cfg.features.column_aliases,
            verbose=verbose,
            n_workers=self.n_workers,
        )
