# -*- coding: utf-8 -*-
"""
run_10day_sample.py — 批量随机采样查询：最后30天随机10天×每天10个时间点，
输出 Top-5 匹配详情（稳定工况ID、相似度S、匹配度D、所有匹配特征值 + 查询输入特征值）。

用法:
    python plan_center/run_10day_sample.py                              # 默认配置
    python plan_center/run_10day_sample.py --days 10 --points-per-day 10 --random-seed 42
    python plan_center/run_10day_sample.py --output result.csv --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# =========================
# 数据结构
# =========================

@dataclass
class Top5Detail:
    """单条查询点的 Top-5 完整详情。"""
    query_index: int           # 查询序号（0-based）
    query_time: str            # 查询时间字符串
    query_features: dict[str, float]  # 查询输入特征（9维原始）
    top5: list[dict[str, Any]]  # Top-5 匹配详情列表


# =========================
# 参数解析
# =========================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量随机采样查询：最后30天随机10天×每天10个时间点，输出 Top-5 详情到 CSV"
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认 defaults.yaml）")
    parser.add_argument("--query-parquet", type=str, default=None, help="查询数据 parquet 路径")
    parser.add_argument("--days", type=int, default=10, help="随机采样天数（默认 10）")
    parser.add_argument("--points-per-day", type=int, default=10, help="每天采样点数（默认 10）")
    parser.add_argument("--random-seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K（默认 5）")
    parser.add_argument("--output", type=str, default="plan_10day_top5.csv", help="输出 CSV 路径")
    return parser.parse_args()


# =========================
# 随机采样
# =========================

def _sample_query_points(
    df_query: pd.DataFrame,
    time_col: str,
    days: int,
    points_per_day: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    从查询数据的最后30天中，随机选 `days` 天，每天随机取 `points_per_day` 个时间点。

    参数：
        df_query: 查询数据 DataFrame（含时间列）
        time_col: 时间列名
        days: 随机天数
        points_per_day: 每天采样点数
        random_seed: 随机种子

    返回：
        DataFrame，含 query_index, query_time, 原始特征列
    """
    rng = np.random.RandomState(random_seed)

    # 确保时间列是 datetime
    df_query = df_query.copy()
    df_query[time_col] = pd.to_datetime(df_query[time_col], errors="coerce")
    df_query = df_query.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    # 筛选最后30天
    max_time = df_query[time_col].max()
    start_time = max_time - pd.Timedelta(days=30)
    mask = df_query[time_col] >= start_time
    df_last30 = df_query[mask].reset_index(drop=True)

    print(f"最后30天数据量: {len(df_last30)} 行，时间范围: {df_last30[time_col].min()} ~ {df_last30[time_col].max()}")

    # 按日期分组
    df_last30["_date"] = df_last30[time_col].dt.date
    unique_dates = df_last30["_date"].unique()

    if len(unique_dates) < days:
        print(f"警告：最后30天仅 {len(unique_dates)} 天，不足 {days} 天，将使用全部日期")
        days = len(unique_dates)

    # 随机选 days 天
    selected_dates = rng.choice(unique_dates, size=days, replace=False)

    sampled_rows = []
    for i, date in enumerate(selected_dates):
        day_df = df_last30[df_last30["_date"] == date]
        n_sample = min(points_per_day, len(day_df))
        sampled = day_df.sample(n=n_sample, random_state=int(rng.randint(0, 2**31)))
        sampled_rows.append(sampled)

    result = pd.concat(sampled_rows, ignore_index=True)
    result["query_index"] = np.arange(len(result))
    print(f"采样完成: {len(result)} 个查询点，{days} 天 × {points_per_day} 点/天")
    return result


# =========================
# 单次详细查询（Top-5 完整信息）
# =========================

def _query_one_detailed(
    raw_features: dict[str, float],
    store: Any,
    cfg: Any,
    models: dict[str, object],
    top_k: int,
) -> dict[str, Any]:
    """
    单次查询，返回 Top-5 完整详情（含稳定工况ID、相似度S、匹配度D、所有匹配特征值）。

    参数：
        raw_features: 查询输入特征（9维）
        store: StandardStore
        cfg: PlanningConfig
        models: 残差模型字典
        top_k: 取 Top-K

    返回：
        {
            "query_features": dict,     # 查询输入特征
            "match_status": str,         # 匹配状态
            "top5": [                    # Top-K 列表
                {
                    "rank": int,
                    "stable_id": int/str,
                    "window_time": str,
                    "similarity_S": float,
                    "match_D": float,
                    "eff_score_E": float,
                    "features": dict,     # 所有参与匹配的特征值（原始+残差）
                },
                ...
            ]
        }
    """
    from plan_center.features import make_query_vector_15d
    from plan_center.config import build_feature_weights
    from plan_center.similarity import weighted_vector_1d, cosine01, flow_gate_keep_mask, pct_rank

    feat = cfg.features
    match_cfg = cfg.matching
    gate_cfg = cfg.flow_gate

    # 1. 构建15维查询向量
    q_15d = make_query_vector_15d(raw_features, models, feat)

    # 2. 加权归一化
    weights = build_feature_weights(feat)
    q_xw, _ = weighted_vector_1d(q_15d, store.sim_feature_cols, store.norm_stats, weights)

    # NaN保护：用0填充（中位数归一化后0=中位数水平）
    if np.any(np.isnan(q_xw)):
        q_xw = np.nan_to_num(q_xw, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. 计算相似度（全库）
    s_all = cosine01(q_xw.reshape(1, -1), store.xw_standard)[0]

    # NaN检查：如有NaN，用均值填充
    if np.any(np.isnan(s_all)):
        nan_mask = np.isnan(s_all)
        mean_val = np.nanmean(s_all)
        s_all = np.where(nan_mask, mean_val, s_all)

    # 4. 硬门控
    q_load = float(raw_features.get(feat.load_col, 0.0))
    keep_mask = flow_gate_keep_mask(q_load, store.loads_standard, gate_cfg)
    valid_pos = np.where(keep_mask)[0]

    if len(valid_pos) == 0:
        # 无样本通过硬门控，按最近负荷兜底
        if match_cfg.allow_fallback_nearest_load:
            valid_pos = np.argsort(np.abs(store.loads_standard - q_load))[:min(top_k, len(store.loads_standard))]
            s_all = s_all.copy()
            s_all[valid_pos] = 0.0
            match_status = "负荷硬门控未命中，按最近负荷兜底"
        else:
            return {
                "query_features": raw_features,
                "match_status": "无样本通过负荷硬门控",
                "top5": [],
            }
    else:
        match_status = "正常匹配"

    # 5. D = a*S + b*E
    d_all = match_cfg.d_weight_s * s_all + match_cfg.d_weight_e * store.eff_score_all

    # 6. Top-K 排序
    d_valid = d_all[valid_pos]
    order = np.argsort(d_valid)[::-1]
    top_pos = valid_pos[order[:min(top_k, len(order))]]

    if len(top_pos) == 0:
        return {
            "query_features": raw_features,
            "match_status": match_status + "；无有效候选样本",
            "top5": [],
        }

    # 7. 构建 Top-K 详情
    top_indices = top_pos.tolist()
    top_s = s_all[top_indices]
    top_d = d_all[top_indices]
    top_e = store.eff_score_all[top_indices]
    top_df = store.df_standard.iloc[top_indices]

    top5 = []
    for rank, (idx, s_val, d_val, e_val) in enumerate(zip(top_indices, top_s, top_d, top_e), start=1):
        match_row = top_df.iloc[rank - 1]

        # 所有参与匹配的特征值（原始特征 + 主汽流量 + 残差特征）
        features = {}
        for c in feat.raw_features:
            features[c] = float(match_row.get(c, float("nan")))
        # 主汽流量（硬门控变量）
        features[feat.load_col] = float(match_row.get(feat.load_col, float("nan")))
        for target in feat.residual_targets:
            resid_col = f"resid_{target}"
            features[resid_col] = float(match_row.get(resid_col, float("nan")))

        top5.append({
            "rank": rank,
            "stable_id": match_row.get("稳定工况ID", idx),
            "window_time": str(match_row.get("稳定窗口时间范围", "")),
            "similarity_S": float(s_val),
            "match_D": float(d_val),
            "eff_score_E": float(e_val),
            "features": features,
        })

    return {
        "query_features": raw_features,
        "match_status": match_status,
        "top5": top5,
    }


# =========================
# 构建输出行
# =========================

def _build_output_rows(details: list[Top5Detail], cfg: Any) -> list[dict[str, Any]]:
    """
    将 Top5Detail 列表扁平化为 CSV 行列表。

    每个查询点展开为 top_k 行（Top1~Top5）。
    """
    rows = []
    feat = cfg.features

    for detail in details:
        q_feat = detail.query_features
        for match in detail.top5:
            row: dict[str, Any] = {}

            # ---- 查询输入信息 ----
            row["查询序号"] = detail.query_index
            row["查询时间"] = detail.query_time

            for c in feat.raw_features:
                row[f"查询_{c}"] = q_feat.get(c, float("nan"))
            # 主汽流量（硬门控变量，单独输出）
            row[f"查询_{feat.load_col}"] = q_feat.get(feat.load_col, float("nan"))

            # ---- 匹配信息 ----
            row["匹配排名"] = match["rank"]
            row["匹配ID"] = match["stable_id"]
            row["匹配窗口时间"] = match["window_time"]
            row["相似度S"] = match["similarity_S"]
            row["匹配度D"] = match["match_D"]
            row["效率分位数E"] = match["eff_score_E"]

            # ---- 匹配特征值（原始 + 主汽流量 + 残差）----
            for c in feat.raw_features:
                row[f"匹配_{c}"] = match["features"].get(c, float("nan"))
            # 主汽流量（硬门控变量，单独输出）
            row[f"匹配_{feat.load_col}"] = match["features"].get(feat.load_col, float("nan"))
            for target in feat.residual_targets:
                resid_col = f"resid_{target}"
                row[f"匹配_{resid_col}"] = match["features"].get(resid_col, float("nan"))

            rows.append(row)

    return rows


# =========================
# 主流程
# =========================

def main() -> None:
    args = _parse_args()

    # 1. 加载引擎
    print("[1] 加载 PlanningEngine...")
    from plan_center import PlanningEngine
    engine = PlanningEngine(args.config)
    print(f"    {engine}")

    # 2. 读取查询数据
    query_parquet = args.query_parquet or engine.cfg.paths.query_parquet
    if not query_parquet or not Path(query_parquet).exists():
        print(f"错误: 查询数据路径不存在 {query_parquet}")
        sys.exit(1)

    print(f"\n[2] 读取查询数据: {query_parquet}")
    df_query = pd.read_parquet(query_parquet)
    print(f"    总行数: {len(df_query)}")

    # 解析时间列
    time_col = engine.cfg.time_col or "时间"
    if time_col not in df_query.columns:
        raise ValueError(f"时间列 '{time_col}' 不在查询数据中。可用列: {list(df_query.columns)[:10]}")

    # 3. 随机采样
    print(f"\n[3] 随机采样（最后30天，{args.days}天×{args.points_per_day}点）...")
    df_sample = _sample_query_points(
        df_query,
        time_col=time_col,
        days=args.days,
        points_per_day=args.points_per_day,
        random_seed=args.random_seed,
    )

    # 4. 逐点查询
    print(f"\n[4] 执行 {len(df_sample)} 次查询...")

    # 应用列别名映射
    aliases = engine.cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in df_sample.columns and new not in df_sample.columns:
                df_sample[new] = df_sample[old]
            elif old in df_sample.columns and new in df_sample.columns:
                df_sample = df_sample.drop(columns=[old])

    details: list[Top5Detail] = []
    for idx, row in df_sample.iterrows():
        # 提取查询特征
        raw_features = {}
        for c in engine.cfg.features.raw_features:
            raw_features[c] = float(row.get(c, 0.0))
        for c in engine.cfg.features.residual_inputs:
            if c not in raw_features:
                raw_features[c] = float(row.get(c, 0.0))
        raw_features[engine.cfg.features.load_col] = float(row.get(engine.cfg.features.load_col, 0.0))

        query_time = row.get(time_col, None)
        query_time_str = str(query_time) if query_time is not None else ""

        # 调用查询
        result = _query_one_detailed(
            raw_features=raw_features,
            store=engine.store,
            cfg=engine.cfg,
            models=engine.models,
            top_k=args.top_k,
        )

        detail = Top5Detail(
            query_index=int(row.get("query_index", idx)),
            query_time=query_time_str,
            query_features=raw_features,
            top5=result["top5"],
        )
        details.append(detail)

        if (len(details) % 10 == 0) or (len(details) == len(df_sample)):
            print(f"    进度: {len(details)}/{len(df_sample)}")

    # 5. 构建输出
    print(f"\n[5] 构建输出 CSV...")
    rows = _build_output_rows(details, engine.cfg)
    out_df = pd.DataFrame(rows)
    print(f"    输出行数: {len(out_df)}（{len(details)} 查询点 × {args.top_k} Top-K）")
    print(f"    输出列数: {len(out_df.columns)}")

    # 6. 保存
    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[6] 已保存: {out_path.absolute()}")


if __name__ == "__main__":
    main()
