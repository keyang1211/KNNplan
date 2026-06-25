# -*- coding: utf-8 -*-
"""
continuity.py — 输出端连续性处理（时间间隔重置 + 变化率限幅）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ContinuityConfig


def should_reset_continuity(
    prev_time: object,
    current_time: object,
    cfg: ContinuityConfig,
) -> bool:
    """
    判断是否因时间间隔过大而重置连续性状态。

    参数：
        prev_time: 上一分钟的时间戳
        current_time: 当前时间戳
        cfg: ContinuityConfig

    返回：
        True 表示应重置（不继承上一分钟中心）
    """
    if not cfg.reset_on_time_gap:
        return False
    if prev_time is None or current_time is None:
        return False

    try:
        prev_ts = pd.Timestamp(prev_time)
        curr_ts = pd.Timestamp(current_time)
    except Exception:
        return False

    if pd.isna(prev_ts) or pd.isna(curr_ts):
        return False

    gap_min = abs((curr_ts - prev_ts).total_seconds()) / 60.0
    return gap_min > float(cfg.max_gap_minutes)


def has_valid_center(center: dict[str, float] | None) -> bool:
    """检查中心字典是否含至少一个非 NaN 值。"""
    if center is None:
        return False
    vals = [center.get(c, np.nan) for c in center]
    return any(pd.notna(v) for v in vals)


def apply_output_continuity(
    raw_center: dict[str, float],
    prev_center: dict[str, float] | None,
    cfg: ContinuityConfig,
    reset_happened: bool = False,
    is_low_sim_fallback: bool = False,
) -> tuple[dict[str, float], dict]:
    """
    对原始规划中心执行输出端连续性处理：变化率限幅。

    参数：
        raw_center: 原始规划中心 {col: value}
        prev_center: 上一分钟最终中心（None=首点）
        cfg: ContinuityConfig
        reset_happened: 是否已发生时间间隔重置
        is_low_sim_fallback: 是否处于低相似度回退

    返回：
        (final_center, diag)
        final_center: {col: value} 连续性处理后的最终中心
        diag: {"status": str, "reset": bool, "rate_limited": list[str]}
    """
    if not has_valid_center(raw_center):
        return raw_center.copy(), {
            "status": "无有效原始中心",
            "reset": bool(reset_happened),
            "rate_limited": [],
        }

    final_center = raw_center.copy()

    # 时间间隔重置 → 不继承上一分钟
    if reset_happened or prev_center is None or not has_valid_center(prev_center):
        return final_center, {
            "status": "首个有效点/无上一分钟中心，未处理" if not reset_happened else "时间间隔过大，重置",
            "reset": bool(reset_happened),
            "rate_limited": [],
        }

    # 变化率限幅
    limited_features = []

    if cfg.enable_rate_limit:
        for c in cfg.rate_limit_features:
            if c not in final_center or c not in prev_center:
                continue
            if c not in cfg.rate_limit_abs:
                continue

            cur = final_center.get(c, np.nan)
            pre = prev_center.get(c, np.nan)
            limit = cfg.rate_limit_abs.get(c)

            if limit is None or pd.isna(cur) or pd.isna(pre):
                continue

            limit = float(limit)
            if limit < 0:
                raise ValueError(f"rate_limit_abs['{c}'] 不能为负数")

            diff = float(cur) - float(pre)
            if abs(diff) > limit:
                final_center[c] = float(pre) + np.sign(diff) * limit
                limited_features.append(c)

    status = "已连续性处理" if limited_features else "未触发连续性处理"

    return final_center, {
        "status": status,
        "reset": bool(reset_happened),
        "rate_limited": limited_features,
    }
