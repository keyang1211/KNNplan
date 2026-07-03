# -*- coding: utf-8 -*-
"""
query.py — 单次查询核心：query_one() → PlanResult
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PlanningConfig, build_feature_weights
from .continuity import apply_output_continuity, has_valid_center, should_reset_continuity
from .features import make_query_vector_15d
from .schemas import PlanResult
from .similarity import candidate_similarity, compute_and_normalize_candidates, cosine01, flow_gate_keep_mask, weighted_vector_1d
from .standard_store import StandardStore


def _nan_safe(v: float) -> float:
    """NaN 安全返回：确保 float 返回（np.nan → float('nan')）。"""
    return float(v)


def query_one(
    raw_features: dict[str, float] | pd.Series,
    store: StandardStore,
    cfg: PlanningConfig,
    models: dict[str, object],
) -> PlanResult:
    """
    不含连续性的单次查询。

    参数：
        raw_features: 原始特征字典/Series（需含 load_col + raw_features 的所有列）
        store: StandardStore
        cfg: PlanningConfig
        models: {target: model} 残差模型字典

    返回：
        PlanResult（raw_plan_center 已填写，final_plan_center 与 raw 相同）
    """
    result = PlanResult()
    feat = cfg.features
    match_cfg = cfg.matching
    gate_cfg = cfg.flow_gate

    # 2. 构建查询向量
    try:
        q_15d = make_query_vector_15d(raw_features, models, feat)
    except Exception as e:
        result.match_status = f"查询特征异常: {e}"
        return result

    # 3. 硬门控先筛选候选
    raw_dict = raw_features if isinstance(raw_features, dict) else dict(raw_features)
    q_load = float(raw_dict.get(feat.load_col, 0.0))
    keep_mask = flow_gate_keep_mask(q_load, store.loads_standard, gate_cfg)
    valid_pos = np.where(keep_mask)[0]

    if len(valid_pos) == 0:
        if not match_cfg.allow_fallback_nearest_load:
            result.match_status = "无样本通过负荷硬门控"
            return result
        valid_pos = np.argsort(np.abs(store.loads_standard - q_load))[:min(match_cfg.top_k, len(store.loads_standard))]
        result.match_status = "负荷硬门控未命中，按最近负荷兜底"
    else:
        result.match_status = "正常匹配"

    # 4. 提取候选子集，动态计算候选集归一化参数并做相似度
    df_candidates = store.df_standard.iloc[valid_pos]
    weights = build_feature_weights(feat)
    global_norm_stats = getattr(store, 'norm_stats', None)
    s_candidates, effective_norm_stats, norm_source = compute_and_normalize_candidates(
        df_candidates, q_15d, store.sim_feature_cols, weights, global_norm_stats
    )

    # NaN 检查：如有 NaN，用均值填充
    if np.any(np.isnan(s_candidates)):
        nan_mask = np.isnan(s_candidates)
        mean_val = np.nanmean(s_candidates)
        s_candidates = np.where(nan_mask, mean_val, s_candidates)

    # 5. D = a*S + b*E（仅候选子集）
    d_candidates = match_cfg.d_weight_s * s_candidates + match_cfg.d_weight_e * store.eff_score_all[valid_pos]

    # 6. Top-k by D
    order = np.argsort(d_candidates)[::-1]
    top_pos_local = order[:min(match_cfg.top_k, len(order))]
    top_pos = valid_pos[top_pos_local]

    if len(top_pos) == 0:
        result.match_status += "；无有效候选样本"
        return result

    result.topk_indices = top_pos.tolist()
    result.topk_count = len(top_pos)
    result.best_index = int(top_pos[0])
    result.similarity_best = _nan_safe(s_candidates[top_pos_local[0]])
    result.similarity_topk_mean = _nan_safe(np.nanmean(s_candidates[top_pos_local]))
    result.score_d_best = _nan_safe(d_candidates[top_pos_local[0]])
    result.score_d_topk_mean = _nan_safe(np.nanmean(d_candidates[top_pos_local]))
    result.eff_score_best = _nan_safe(store.eff_score_all[top_pos[0]])
    result.eff_topk_mean = _nan_safe(
        store.df_standard.iloc[top_pos][feat.eff_col].astype(float).mean()
    )

    # 7. 低相似度回退判定
    if match_cfg.enable_low_sim_fallback:
        result.low_sim_fallback = bool(
            result.similarity_topk_mean < match_cfg.low_sim_fallback_threshold
        )
        result.fallback_threshold = float(match_cfg.low_sim_fallback_threshold)

    if result.low_sim_fallback:
        # 回退：当前真实工况作为原始规划中心
        center = {c: _nan_safe(raw_dict.get(c, np.nan)) for c in feat.plan_center_cols}
        result.raw_plan_center = center
        result.plan_center_source = "当前真实工况"
        result.match_status += "；低相似度回退"
    else:
        result.plan_center_source = "历史TopK标准样本"

        if match_cfg.plan_center_mode == 1:
            # 最佳单样本
            best_df = store.df_standard.iloc[top_pos[0]]
            result.raw_plan_center = {
                c: _nan_safe(float(best_df[c])) for c in feat.plan_center_cols
            }
        elif match_cfg.plan_center_mode == 2:
            # Top-k 加权/算术平均
            top_df = store.df_standard.iloc[top_pos]
            if match_cfg.topk_avg_method == "weighted":
                weights_d = np.clip(d_candidates[top_pos_local], 0.001, None)
                center_vals = {}
                for c in feat.plan_center_cols:
                    vals = top_df[c].astype(float).values
                    center_vals[c] = _nan_safe(float(np.average(vals, weights=weights_d)))
                result.raw_plan_center = center_vals
            elif match_cfg.topk_avg_method == "mean":
                result.raw_plan_center = {
                    c: _nan_safe(float(top_df[c].astype(float).mean())) for c in feat.plan_center_cols
                }
            else:
                raise ValueError(f"未知的 topk_avg_method: {match_cfg.topk_avg_method}")
        else:
            raise ValueError(f"未知的 plan_center_mode: {match_cfg.plan_center_mode}")

    # final 先等于 raw（连续性处理在 query_one_full 中做）
    result.final_plan_center = result.raw_plan_center.copy()
    return result


def query_one_full(
    raw_features: dict[str, float] | pd.Series,
    store: StandardStore,
    cfg: PlanningConfig,
    models: dict[str, object],
    prev_center: dict[str, float] | None = None,
    prev_time: object = None,
    current_time: object = None,
) -> PlanResult:
    """
    完整单次调用：query_one + apply_output_continuity。

    参数：
        raw_features: 原始特征字典/Series
        store: StandardStore
        cfg: PlanningConfig
        models: {target: model} 残差模型字典
        prev_center: 上一分钟最终中心（None=首点）
        prev_time: 上一分钟时间戳（用于判断时间间隔重置）
        current_time: 当前时间戳

    返回：
        PlanResult（含连续性诊断）
    """
    result = query_one(raw_features, store, cfg, models)

    # 连续性重置判定
    reset = should_reset_continuity(prev_time, current_time, cfg.continuity)
    result.continuity_reset = reset

    # 连续性处理
    final_center, diag = apply_output_continuity(
        raw_center=result.raw_plan_center,
        prev_center=prev_center,
        cfg=cfg.continuity,
        reset_happened=reset,
        is_low_sim_fallback=result.low_sim_fallback,
    )

    result.final_plan_center = final_center
    result.continuity_status = diag["status"]
    result.rate_limited_features = diag["rate_limited"]
    result.smoothed_features = diag.get("smoothed", [])

    return result
