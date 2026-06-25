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
from .similarity import cosine01, flow_gate_keep_mask, weighted_vector_1d
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

    # 1. 15维查询向量
    try:
        q_15d = make_query_vector_15d(raw_features, models, feat)
    except Exception as e:
        result.match_status = f"查询特征异常: {e}"
        return result

    # 2. 加权（用 store 的 norm_stats）
    weights = build_feature_weights(feat)
    try:
        q_xw, _ = weighted_vector_1d(q_15d, store.sim_feature_cols, store.norm_stats, weights)
    except Exception as e:
        result.match_status = f"加权失败: {e}"
        return result

    # 3. 余弦相似度
    q_xw_2d = q_xw.reshape(1, -1)
    s_all = cosine01(q_xw_2d, store.xw_standard)[0]

    # 4. 硬门控
    raw_dict = raw_features if isinstance(raw_features, dict) else dict(raw_features)
    q_load = float(raw_dict.get(feat.load_col, 0.0))
    keep_mask = flow_gate_keep_mask(q_load, store.loads_standard, gate_cfg)
    valid_pos = np.where(keep_mask)[0]

    if len(valid_pos) == 0:
        if not match_cfg.allow_fallback_nearest_load:
            result.match_status = "无样本通过负荷硬门控"
            return result
        # 兜底：按最近负荷取 Top-k
        valid_pos = np.argsort(np.abs(store.loads_standard - q_load))[:min(match_cfg.top_k, len(store.loads_standard))]
        s_all = s_all.copy()
        s_all[valid_pos] = 0.0  # 兜底样本相似度置0，避免误判
        result.match_status = "负荷硬门控未命中，按最近负荷兜底"
    else:
        result.match_status = "正常匹配"

    # 5. D = a*S + b*E（无 F）
    d_all = match_cfg.d_weight_s * s_all + match_cfg.d_weight_e * store.eff_score_all

    # 6. Top-k by D
    d_valid = d_all[valid_pos]
    order = np.argsort(d_valid)[::-1]
    top_pos = valid_pos[order[:min(match_cfg.top_k, len(order))]]

    if len(top_pos) == 0:
        result.match_status += "；无有效候选样本"
        return result

    result.topk_indices = top_pos.tolist()
    result.topk_count = len(top_pos)
    result.best_index = int(top_pos[0])
    result.similarity_best = _nan_safe(s_all[top_pos[0]])
    result.similarity_topk_mean = _nan_safe(np.nanmean(s_all[top_pos]))
    result.score_d_best = _nan_safe(d_all[top_pos[0]])
    result.score_d_topk_mean = _nan_safe(np.nanmean(d_all[top_pos]))
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
                weights_d = np.clip(d_all[top_pos], 0.001, None)
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
