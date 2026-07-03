# -*- coding: utf-8 -*-
"""
schemas.py — 列名前缀常量、PlanResult 数据结构、结果→DataFrame 装配工具
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# =========================
# 列名前缀
# =========================
PLAN_COL_PREFIX = "规划中心_"
RAW_PLAN_COL_PREFIX = "原始规划中心_"


# =========================
# 单次查询结果
# =========================
@dataclass
class PlanResult:
    """单次查询的完整结果（不含连续性时也可用，字段留空）。"""

    # ---- 原始规划中心（搜索结果或低相似度回退） ----
    raw_plan_center: dict[str, float] = field(default_factory=dict)
    # ---- 连续性处理后的最终规划中心 ----
    final_plan_center: dict[str, float] = field(default_factory=dict)

    # ---- 匹配诊断 ----
    match_status: str = "未计算"
    topk_indices: list[int] = field(default_factory=list)
    best_index: int | None = None
    topk_count: int = 0
    similarity_best: float = np.nan
    similarity_topk_mean: float = np.nan
    score_d_best: float = np.nan       # D = a*S + b*E
    score_d_topk_mean: float = np.nan

    # ---- 效率诊断 ----
    eff_score_best: float = np.nan     # E（效率分位数）
    eff_topk_mean: float = np.nan      # Top-k 平均真实锅炉效率

    # ---- 回退诊断 ----
    low_sim_fallback: bool = False
    fallback_threshold: float = np.nan
    plan_center_source: str = ""

    # ---- 连续性诊断（query_one_full 填写） ----
    continuity_status: str = ""
    continuity_reset: bool = False
    rate_limited_features: list[str] = field(default_factory=list)
    smoothed_features: list[str] = field(default_factory=list)


def plan_result_to_row(result: PlanResult, plan_center_cols: list[str]) -> dict:
    """将 PlanResult 展平为一行 dict，便于拼接到 DataFrame。"""
    row: dict = {}

    # 原始规划中心列
    for c in plan_center_cols:
        row[RAW_PLAN_COL_PREFIX + c] = result.raw_plan_center.get(c, np.nan)

    # 最终规划中心列
    for c in plan_center_cols:
        row[PLAN_COL_PREFIX + c] = result.final_plan_center.get(c, np.nan)

    # 匹配诊断
    row["规划匹配状态"] = result.match_status
    row["TopK数量"] = result.topk_count
    row["匹配标准样本index"] = result.best_index if result.best_index is not None else np.nan
    row["匹配标准样本TopK"] = ",".join(str(int(i)) for i in result.topk_indices)
    row["相似度S"] = result.similarity_best
    row["TopK_S均值"] = result.similarity_topk_mean
    row["匹配度D"] = result.score_d_best
    row["TopK_D均值"] = result.score_d_topk_mean

    # 效率诊断
    row["效率得分E"] = result.eff_score_best
    row["TopK_效率均值"] = result.eff_topk_mean

    # 连续性诊断
    row["连续性处理状态"] = result.continuity_status
    row["连续性重置"] = result.continuity_reset
    row["限幅触发数"] = len(result.rate_limited_features)
    row["限幅触发特征"] = ",".join(result.rate_limited_features)

    # 回退诊断
    row["低相似度回退"] = result.low_sim_fallback
    row["低相似度回退阈值"] = result.fallback_threshold
    row["规划中心来源"] = result.plan_center_source

    # 时间差诊断（动态附加属性，可选）
    row["Top1时间差_天"] = getattr(result, '_top1_time_diff', np.nan)
    row["Top5平均时间差_天"] = getattr(result, '_top5_mean_time_diff', np.nan)

    return row


def build_output_dataframe(
    raw_df: pd.DataFrame,
    results: list[PlanResult],
    plan_center_cols: list[str],
) -> pd.DataFrame:
    """将原始 DataFrame 与多条 PlanResult 拼接为最终输出 DataFrame。"""
    diag_rows = [plan_result_to_row(r, plan_center_cols) for r in results]
    diag_df = pd.DataFrame(diag_rows, index=raw_df.index)

    return pd.concat([raw_df.reset_index(drop=True), diag_df.reset_index(drop=True)], axis=1)
