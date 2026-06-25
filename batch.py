# -*- coding: utf-8 -*-
"""
batch.py — 批量驱动：run_batch() 逐行查询 + BatchState + parquet 输出
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PlanningConfig
from .engine import PlanningEngine
from .continuity import has_valid_center
from .schemas import PlanResult, build_output_dataframe
from .standard_store import deduplicate_columns_keep_first


# =========================
# 时间列候选
# =========================
TIME_COL_CANDIDATES = [
    "时间", "采样时间", "数据时间", "日期时间", "统计时间", "Time", "time",
    "Timestamp", "timestamp", "DateTime", "datetime", "date", "日期",
]


# =========================
# 批量状态
# =========================

@dataclass
class BatchState:
    """批量处理中的状态（逐行维护）。"""
    prev_center: dict[str, float] | None = None
    prev_time: object = None


# =========================
# 时间解析
# =========================

def _parse_datetime_bound(value: str | None, is_end: bool = False) -> pd.Timestamp | None:
    """解析用户输入的起止时间。只填日期时，结束日期自动扩展到当天末尾。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    dt = pd.to_datetime(text, errors="raise")
    is_date_only = re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text) is not None
    if is_end and is_date_only:
        dt = dt + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return dt


def _resolve_time_col(path: str, time_col: str | None, df: pd.DataFrame = None) -> str:
    """解析时间列名。优先 time_col，其次自动识别。"""
    cols = list(df.columns) if df is not None else []

    if time_col:
        if cols and time_col not in cols:
            raise ValueError(f"time_col='{time_col}' 不在 parquet 列名中")
        return time_col

    for c in TIME_COL_CANDIDATES:
        if c in cols:
            return c

    raise ValueError(
        f"无法自动识别时间列。请在 config 中设置 time_col。可用列名: {cols[:30]}"
    )


def _build_time_masks(
    time_series: pd.Series,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
) -> tuple[pd.Series, pd.Series]:
    """
    构建时间掩码。

    返回：
        (calc_mask, out_mask)
        calc_mask: 含预读范围（用于滚动均值计算）
        out_mask: 仅用户指定范围（用于最终输出）
    """
    t = pd.to_datetime(time_series, errors="coerce")
    calc_mask = t.notna()
    out_mask = t.notna()

    if start_dt is not None:
        calc_mask &= t >= start_dt
        out_mask &= t >= start_dt

    if end_dt is not None:
        calc_mask &= t <= end_dt
        out_mask &= t <= end_dt

    return calc_mask, out_mask


# =========================
# 主批量函数
# =========================

def run_batch(
    engine: PlanningEngine,
    query_parquet: str | Path | None = None,
    output_parquet: str | Path | None = None,
    time_col: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    lookback_minutes: int = 120,
) -> pd.DataFrame:
    """
    批量处理：读取查询 parquet，逐行调用 plan_one，拼接输出。

    参数：
        engine: PlanningEngine 实例
        query_parquet: 查询数据 parquet 路径（或从 config 读取）
        output_parquet: 输出 parquet 路径（None=不写文件）
        time_col: 时间列名（None=从 config 或自动识别）
        start_time: 开始时间（含）
        end_time: 结束时间（含）
        lookback_minutes: 预读分钟数（用于滚动均值计算，如吨煤产汽）

    返回：
        完整输出 DataFrame
    """
    if query_parquet is None:
        query_parquet = engine.cfg.paths.query_parquet
    if query_parquet is None:
        raise ValueError("必须指定 query_parquet 路径（通过参数或 config.paths.query_parquet）")

    # 解析时间范围
    start_dt = _parse_datetime_bound(start_time, is_end=False)
    end_dt = _parse_datetime_bound(end_time, is_end=True)

    if start_dt and end_dt and start_dt > end_dt:
        raise ValueError(f"start_time({start_dt}) 晚于 end_time({end_dt})")

    # 读取 parquet
    raw = pd.read_parquet(str(query_parquet))
    raw = deduplicate_columns_keep_first(raw)

    # 应用列别名映射（将查询数据的列名对齐到配置中的特征名）
    aliases = engine.cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in raw.columns:
                if new in raw.columns:
                    # 目标列已存在，直接删除源列（避免重复列）
                    raw = raw.drop(columns=[old])
                    print(f"目标列 '{new}' 已存在，已删除源列 '{old}'")
                else:
                    raw = raw.rename(columns={old: new})
                    print(f"已应用列别名映射: {old} → {new}")

    # 解析时间列
    if time_col is None:
        time_col = engine.cfg.time_col
    time_col = _resolve_time_col(str(query_parquet), time_col, raw)

    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")

    # 时间范围筛选
    calc_mask, out_mask = _build_time_masks(raw[time_col], start_dt, end_dt)
    raw_calc = raw.loc[calc_mask].sort_values(time_col).reset_index(drop=True)

    print(f"查询数据读取: {raw_calc.shape}")

    # 逐行查询
    results: list[PlanResult] = []
    state = BatchState()

    for idx, row in raw_calc.iterrows():
        current_time = row.get(time_col)

        # 向量构建（从行中提取原始特征）
        raw_features = {}
        for c in engine.cfg.features.raw_features:
            raw_features[c] = float(row.get(c, 0.0))
        for c in engine.cfg.features.residual_inputs:
            if c not in raw_features:
                raw_features[c] = float(row.get(c, 0.0))
        if engine.cfg.features.load_col not in raw_features:
            raw_features[engine.cfg.features.load_col] = float(row.get(engine.cfg.features.load_col, 0.0))

        # 调用查询
        result = engine.plan_one(
            raw_features=raw_features,
            prev_center=state.prev_center,
            prev_time=state.prev_time,
            current_time=current_time,
        )

        results.append(result)

        # 更新状态
        if result.final_plan_center and has_valid_center(result.final_plan_center):
            state.prev_center = result.final_plan_center.copy()
            state.prev_time = current_time

    # 构建输出 DataFrame
    df_out = build_output_dataframe(
        raw_df=raw_calc,
        results=results,
        plan_center_cols=engine.cfg.features.plan_center_cols,
    )

    # 写 parquet
    if output_parquet:
        df_out.to_parquet(str(output_parquet), index=False)
        print(f"已输出: {output_parquet}，形状: {df_out.shape}")

    return df_out
