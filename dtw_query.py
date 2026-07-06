# -*- coding: utf-8 -*-
"""
dtw_query.py — DTW 时序查询核心

新增独立查询路径，不修改现有 PlanningEngine / query.py。
依赖：
    - plan_center.similarity（robust_norm_stats / normalize_features / cosine01 等纯函数）
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
from .similarity import cosine01, robust_norm_stats, normalize_features
from .schemas import PlanResult
from .batch import _resolve_time_col


# ============================================================
# 工具：DTW 对齐（numpy 手动实现，scipy 1.15.2 无 dtw 模块）
# ============================================================


def dtw_align(
    query_seq: np.ndarray,
    cand_seq: np.ndarray,
    feature_weights: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """
    标准 DTW 动态规划对齐。

    参数：
        query_seq: (T_q, D) 查询序列
        cand_seq: (T_c, D) 候选序列
        feature_weights: (D,) 各特征的平方根权重（欧氏距离中），None=等权

    返回：
        (aligned_pairs, path_cost)
        aligned_pairs: [(i_q, i_c), ...]，正序
        path_cost: 最小累积代价（DTW 距离）
    """
    n, m, d = query_seq.shape[0], cand_seq.shape[0], query_seq.shape[1]

    if feature_weights is None:
        feature_weights = np.ones(d, dtype=float)
    else:
        feature_weights = np.asarray(feature_weights, dtype=float)

    # 预计算逐点欧氏距离矩阵 (n, m)
    diff = query_seq[:, np.newaxis, :] - cand_seq[np.newaxis, :, :]  # (n, m, d)
    dist_matrix = np.sqrt(np.sum(diff ** 2 * feature_weights[np.newaxis, np.newaxis, :], axis=2))  # (n, m)

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
    return aligned_pairs, float(D[-1, -1])


def dtw_align_with_coverage(
    query_seq: np.ndarray,
    cand_seq: np.ndarray,
    feature_weights: np.ndarray | None = None,
    min_coverage: int = 4,
) -> tuple[list[tuple[int, int]], float, bool]:
    """
    DTW 对齐 + 候选覆盖帧检查。

    比 dtw_align() 多返回一个 bool: 路径中涉及的不重复候选帧数 >= min_coverage。

    参数：
        query_seq: (T_q, D) 查询序列
        cand_seq: (T_c, D) 候选序列
        feature_weights: (D,) 各特征的平方根权重
        min_coverage: 最少不重复候选帧数（默认 4）

    返回：
        (aligned_pairs, path_cost, coverage_ok)
    """
    aligned_pairs, path_cost = dtw_align(query_seq, cand_seq, feature_weights)
    # 计算路径覆盖的不重复候选帧数
    cand_indices = np.array([j for _, j in aligned_pairs])
    n_unique = int(np.unique(cand_indices).size)
    coverage_ok = n_unique >= min_coverage
    return aligned_pairs, path_cost, coverage_ok


# ============================================================
# 工具：DTW 对齐后加权 cosine 均值
# ============================================================


def dtw_weighted_cosine_mean(
    query_seq: np.ndarray,
    cand_seq: np.ndarray,
    aligned_pairs: list[tuple[int, int]],
    feature_weights: np.ndarray | None = None,
) -> float:
    """
    对齐后逐对点计算加权余弦相似度，取均值。

    参数：
        query_seq: (T_q, D) 查询序列（已加权）
        cand_seq: (T_c, D) 候选序列（已加权）
        aligned_pairs: DTW 对齐索引对 [(i_q, i_c), ...]
        feature_weights: 未使用，传 None 即可

    返回：
        float，均值相似度 [0, 1]
    """
    if not aligned_pairs:
        return 0.0

    # 提取对齐点
    q_aligned = np.stack([query_seq[i] for i, _ in aligned_pairs], axis=0)  # (L, D)
    c_aligned = np.stack([cand_seq[j] for _, j in aligned_pairs], axis=0)  # (L, D)

    # 逐对 cosine（逐对计算，因为 cosine01 需要 (1,D)×(1,D)）
    sims = []
    for k in range(len(aligned_pairs)):
        a = q_aligned[k].reshape(1, -1)
        b = c_aligned[k].reshape(1, -1)
        s = float(cosine01(a, b)[0, 0])
        sims.append(s)

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
# DTW 查询主函数
# ============================================================


def query_dtw(
    query_ts: str | pd.Timestamp,
    ref_df_resid: pd.DataFrame,
    feat: Any,
    dtw_cfg: DTWQueryConfig,
    norm_stats: dict | None = None,
    time_col: str | None = None,
    alias_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> PlanResult:
    """
    DTW 时序查询主函数。

    流程：
        1. 解析查询时间戳，定位 ref_df 中的行号
        2. 截取参考窗口（t - ref_days天 ~ t）
        3. 参考窗口全局 robust 归一化
        4. 提取查询序列（ref_window 末尾 query_seq_len 个点）和候选序列（4/5/6 min 滑窗）
        5. DTW 对齐 + 加权 cosine 均值 → 序列相似度
        6. Top-k 排序
        7. 生成规划中心

    参数：
        query_ts: 查询时间戳（字符串或 pd.Timestamp）
        ref_df_resid: 带残差特征的参考 DataFrame（分钟级，时间有序）
        feat: FeatureConfig
        dtw_cfg: DTWQueryConfig
        norm_stats: 归一化参数（可选）
        time_col: 时间列名（None=自动识别）
        alias_map: 列名别名映射
        verbose: 是否打印进度

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

    # ---- 3. 全局 robust 归一化参数 ----
    all_feature_cols = feat.raw_features + [f"resid_{t}" for t in feat.residual_targets]
    sim_feature_cols = list(dict.fromkeys(all_feature_cols))  # 去重保序

    if norm_stats is None:
        if verbose:
            print("[DTW] 计算全局 robust 归一化参数 ...")
        norm_stats = robust_norm_stats(ref_window, sim_feature_cols)

    if verbose:
        print(f"[DTW] 归一化特征数: {len(sim_feature_cols)}")

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

    # ---- 5. 构建候选矩阵（固定 dtw_max_len 分钟，滑动步长 1 分钟）----
    # 候选序列滑动范围: [ref_window 开头, ref_window 末尾 - dtw_max_len)
    # 固定用 dtw_max_len 长度切分，不再 4/5/6 三种
    cand_length = dtw_cfg.dtw_max_len
    cand_pool_start = 0
    cand_pool_end = ref_n - dtw_cfg.query_seq_len  # 不包含 query_seq 部分
    if cand_pool_end < cand_pool_start + cand_length:
        cand_pool_end = ref_n  # 回退：候选池不够时用全部 ref_window

    candidates: list[dict] = []
    for start in range(cand_pool_start, cand_pool_end - cand_length + 1, dtw_cfg.slide_step):
        end = start + cand_length
        mat = ref_window[sim_feature_cols].iloc[start:end].values.astype(float)
        # 映射回全局索引
        global_start = int((ref_df_resid[time_col] <= t_start).sum()) + start
        candidates.append({
            "orig_start_idx": global_start,
            "orig_end_idx": global_start + cand_length,
            "length": cand_length,
            "matrix": mat,
        })

    if verbose:
        total_cands = len(candidates)
        print(f"[DTW] 候选序列数: {total_cands} "
              f"（范围 {dtw_cfg.dtw_min_len}~{dtw_cfg.dtw_max_len} min，步长 {dtw_cfg.slide_step} min）")

    # ---- 6. DTW 对齐 + 覆盖帧检查 + 加权 cosine 均值 ----
    if verbose:
        print(f"[DTW] DTW 对齐 + 覆盖帧检查 + 加权 cosine 计算中 ...")

    sim_scores: list[float] = []
    coverage_ok_list: list[bool] = []
    for cand in candidates:
        aligned_pairs, path_cost, coverage_ok = dtw_align_with_coverage(
            query_matrix, cand["matrix"],
            min_coverage=dtw_cfg.dtw_min_len   # 默认 4
        )
        # 覆盖帧不足的候选相似度置 0（不参与 Top-k）
        if not coverage_ok:
            sim_scores.append(0.0)
        else:
            sim = dtw_weighted_cosine_mean(query_matrix, cand["matrix"], aligned_pairs)
            sim_scores.append(sim)
        cand["dtw_cost"] = path_cost
        cand["path_length"] = len(aligned_pairs)
        coverage_ok_list.append(coverage_ok)

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
    DTW 时序查询引擎。

    持有：
        - cfg: PlanningConfig（含 DTWQueryConfig）
        - ref_df_resid: 带残差特征的分钟级参考 DataFrame
        - models: 残差模型字典
        - norm_stats: 归一化参数（可选）
    """

    cfg: PlanningConfig
    ref_df_resid: pd.DataFrame | None = None
    models: dict[str, object] | None = None
    norm_stats: dict | None = None
    _cache_loaded: bool = field(default=False, init=False, repr=False)

    # ---- 初始化 / 加载 ----

    def _ensure_data_loaded(self) -> None:
        """懒加载：首次调用时加载参考数据 + 残差模型 + 归一化参数。"""
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

        # 3. 归一化参数
        if paths.norm_stats_path and Path(paths.norm_stats_path).exists():
            import json
            with open(paths.norm_stats_path, "r", encoding="utf-8") as f:
                self.norm_stats = json.load(f)
            print(f"[DTW] 归一化参数已加载: {paths.norm_stats_path}")
        else:
            self.norm_stats = None
            print("[DTW] 未提供 norm_stats.json，将动态计算")

        self._cache_loaded = True

    def query_one(self, query_ts: str | pd.Timestamp, verbose: bool = True) -> PlanResult:
        """
        单次 DTW 查询。

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
            norm_stats=self.norm_stats,
            time_col=self.cfg.time_col,
            alias_map=self.cfg.features.column_aliases,
            verbose=verbose,
        )
