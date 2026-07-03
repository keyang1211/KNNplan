# -*- coding: utf-8 -*-
"""
weight_comparison_eval_dtw.py — DTW 查询评估脚本

与 weight_comparison_eval.py 功能对等，但使用 DTWQueryEngine 替代 PlanningEngine。
评估 DTW 时序查询在真实数据上的表现：
  - DTW 加权余弦相似度 S
  - DTW 路径代价
  - 候选池规模
  - 时间偏移（候选末帧与查询时刻的时间差）
  - 规划中心 Loss（各维 + 总 Loss）

用法：
    python plan_center/weight_comparison_eval_dtw.py
    python plan_center/weight_comparison_eval_dtw.py --start 2025-06-01 --end 2025-06-02
    python plan_center/weight_comparison_eval_dtw.py --months 2026-01
    python plan_center/weight_comparison_eval_dtw.py --output dtw_eval_report.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


# =========================
# 数据结构
# =========================

@dataclass
class DTWEvalResult:
    """DTW 评估结果。"""
    query_ts: str                           # 查询时间戳
    similarity_best: float = np.nan         # DTW 最佳相似度 S
    dtw_path_cost: float = np.nan           # DTW 路径代价
    n_candidates: int = 0                   # 候选序列数量
    dtw_path_length: int = 0                # DTW 路径长度（对齐点数）
    time_offset_days: float = np.nan        # 时间偏移（天）
    # 规划中心 Loss
    plan_center: dict[str, float] = field(default_factory=dict)      # 规划中心各维值
    actual: dict[str, float] = field(default_factory=dict)            # 实际值各维值
    loss_total: float = np.nan             # 总 Loss
    per_dim_loss: dict[str, float] = field(default_factory=dict)      # 各维 Loss


# =========================
# Loss 计算（复用现有逻辑）
# =========================

LOSS_FEATURE_WEIGHTS = {
    "主汽压力": 1.0, "炉膛差压": 1.0, "一次风流量": 1.0,
    "床温": 1.0, "料层差压": 1.0, "锅炉出口氧量": 0.6, "二次风风量": 0.6,
}


def compute_dtw_loss(plan_center: dict, actual: dict) -> tuple[float, dict]:
    """
    计算 DTW 规划中心的 Loss。

    参数：
        plan_center: 规划中心各维值
        actual: 实际值各维值

    返回：
        (total_loss, per_dim_loss)
    """
    loss_cols = [c for c in LOSS_FEATURE_WEIGHTS if LOSS_FEATURE_WEIGHTS[c] > 0]
    loss_w = np.array([LOSS_FEATURE_WEIGHTS[c] for c in loss_cols], dtype=np.float64)

    errs = []
    per_dim_loss = {}
    for c in loss_cols:
        if c in plan_center and c in actual:
            # 使用 IQR 量纲归一化（与 weight_comparison_eval.py 一致）
            # 这里简化：直接用绝对误差（因为 DTW 每点只有一个候选，无法计算全局 IQR）
            err = abs(plan_center[c] - actual[c])
            per_dim_loss[c] = float(err)
            errs.append(err * loss_w[loss_cols.index(c)])

    total_loss = float(np.mean(errs)) if errs else np.nan
    return total_loss, per_dim_loss


# =========================
# 评估主函数
# =========================

def evaluate_dtw(
    cfg,
    engine,
    df_query: pd.DataFrame,
    time_col: str,
    sample_step: int = 1,
    verbose: bool = True,
) -> list[DTWEvalResult]:
    """
    逐时间戳调用 DTWQueryEngine，收集评估结果。

    参数：
        cfg: PlanningConfig
        engine: DTWQueryEngine（已初始化）
        df_query: 查询数据 DataFrame
        time_col: 时间列名
        sample_step: 采样步长（分钟）
        verbose: 是否打印进度

    返回：
        List[DTWEvalResult]
    """
    feat = cfg.features
    plan_center_cols = list(feat.plan_center_cols)

    # 采样：按 sample_step 每隔 N 个点取一个
    indices = range(0, len(df_query), sample_step)
    total = len(list(indices))

    results = []
    t0 = time.perf_counter()

    for i, row_idx in enumerate(indices):
        row = df_query.iloc[row_idx]
        query_ts = row[time_col]

        try:
            result = engine.query_one(query_ts, verbose=False)
        except Exception as e:
            if verbose and i == 0:
                print(f"[警告] 查询 {query_ts} 失败: {e}")
            continue

        # 提取实际值
        actual = {c: float(row[c]) if c in row.index else np.nan for c in plan_center_cols}

        # 计算规划中心
        plan_center = result.final_plan_center

        # 计算 Loss
        total_loss, per_dim_loss = compute_dtw_loss(plan_center, actual)

        # 构建结果
        eval_result = DTWEvalResult(
            query_ts=str(query_ts),
            similarity_best=result.similarity_best,
            dtw_path_cost=getattr(result, "_dtw_path_cost", np.nan),
            n_candidates=getattr(result, "_n_candidates", 0),
            dtw_path_length=getattr(result, "_dtw_path_length", 0),
            time_offset_days=getattr(result, "_time_offset_days", np.nan),
            plan_center=plan_center,
            actual=actual,
            loss_total=total_loss,
            per_dim_loss=per_dim_loss,
        )
        results.append(eval_result)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            avg = elapsed / (i + 1)
            eta = avg * (total - i - 1)
            print(f"  进度: {i+1}/{total}，均速 {avg:.1f}s/点，剩余 {eta:.0f}s")

    return results


# =========================
# 报告生成
# =========================

def print_dtw_report(results: list[DTWEvalResult], range_label: str):
    """打印评估报告。"""
    if not results:
        print("无有效结果")
        return

    sims = np.array([r.similarity_best for r in results if not np.isnan(r.similarity_best)])
    costs = np.array([r.dtw_path_cost for r in results if not np.isnan(r.dtw_path_cost)])
    offsets = np.array([r.time_offset_days for r in results if not np.isnan(r.time_offset_days)])
    losses = np.array([r.loss_total for r in results if not np.isnan(r.loss_total)])

    print(f"\n{'=' * 70}")
    print(f"DTW 查询评估报告 — 时间段: {range_label}")
    print(f"{'=' * 70}")
    print(f"评估样本数: {len(results)}")

    if len(sims) > 0:
        print(f"\n[DTW 相似度 S]")
        print(f"  均值={sims.mean():.4f}  最大={sims.max():.4f}  最小={sims.min():.4f}  标准差={sims.std():.4f}")

    if len(costs) > 0:
        print(f"\n[DTW 路径代价]")
        print(f"  均值={costs.mean():.4f}  最大={costs.max():.4f}  最小={costs.min():.4f}")

    if len(offsets) > 0:
        print(f"\n[时间偏移（天）]")
        print(f"  均值={offsets.mean():.4f}  最大={offsets.max():.4f}  最小={offsets.min():.4f}")

    if len(losses) > 0:
        print(f"\n[规划中心 Loss]")
        print(f"  总Loss均值={losses.mean():.6f}  最大={losses.max():.6f}  最小={losses.min():.6f}")

        # 各维 Loss
        all_dim_losses: dict[str, list] = {}
        for r in results:
            for c, loss in r.per_dim_loss.items():
                if c not in all_dim_losses:
                    all_dim_losses[c] = []
                all_dim_losses[c].append(loss)

        if all_dim_losses:
            print(f"\n  各维度 Loss:")
            for c, losses_c in sorted(all_dim_losses.items()):
                losses_arr = np.array([l for l in losses_c if not np.isnan(l)])
                if len(losses_arr) > 0:
                    print(f"    {c}: 均值={losses_arr.mean():.4f}  最大={losses_arr.max():.4f}")

    print(f"\n{'=' * 70}")


def save_dtw_report(results: list[DTWEvalResult], output_path: str):
    """保存详细报告到 CSV。"""
    rows = []
    for r in results:
        row = {
            "query_ts": r.query_ts,
            "similarity_best": r.similarity_best,
            "dtw_path_cost": r.dtw_path_cost,
            "n_candidates": r.n_candidates,
            "dtw_path_length": r.dtw_path_length,
            "time_offset_days": r.time_offset_days,
            "loss_total": r.loss_total,
        }
        # 规划中心
        for c, v in r.plan_center.items():
            row[f"plan_center_{c}"] = v
        # 实际值
        for c, v in r.actual.items():
            row[f"actual_{c}"] = v
        # 各维 Loss
        for c, loss in r.per_dim_loss.items():
            row[f"loss_{c}"] = loss
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n报告已保存: {output_path}")
    return df


# =========================
# 主流程
# =========================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DTW 时序查询评估脚本")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--start", type=str, default=None, help="起始日期（如 2025-06-01）")
    parser.add_argument("--end", type=str, default=None, help="终止日期（如 2025-06-02）")
    parser.add_argument("--months", type=str, nargs="+", default=None,
                        help="要验证的月份列表，如 2026-01 2026-03")
    parser.add_argument("--output", type=str, default="dtw_eval_report.csv", help="输出报告路径")
    parser.add_argument("--sample-step", type=int, default=1, help="采样步长（分钟），默认1")
    return parser.parse_args()


def main():
    args = _parse_args()

    print("=== DTW 时序查询评估 ===\n")
    from plan_center.config import load_config
    from plan_center.dtw_query import DTWQueryEngine

    cfg = load_config(args.config)
    print(f"配置文件: {args.config or 'defaults.yaml'}\n")

    # 初始化 DTW 查询引擎
    print("[1] 初始化 DTWQueryEngine（懒加载，首次会生成残差缓存）...")
    engine = DTWQueryEngine(cfg)

    # 读取查询数据
    print("\n[2] 读取查询数据...")
    query_parquet = cfg.paths.query_parquet
    if not query_parquet or not Path(query_parquet).exists():
        raise FileNotFoundError(f"查询数据不存在: {query_parquet}")

    df_query = pd.read_parquet(query_parquet)

    # 列别名映射
    aliases = cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in df_query.columns and new not in df_query.columns:
                df_query[new] = df_query[old]
            elif old in df_query.columns and new in df_query.columns:
                df_query = df_query.drop(columns=[old])

    time_col = cfg.time_col or "时间"
    if time_col not in df_query.columns:
        raise ValueError(f"时间列 '{time_col}' 不在查询数据中")

    df_query[time_col] = pd.to_datetime(df_query[time_col], errors="coerce")
    df_query = df_query.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    print(f"    查询数据: {df_query.shape}，时间范围 {df_query[time_col].min()} ~ {df_query[time_col].max()}")

    # 按日期范围或月份筛选
    if args.start and args.end:
        start_time = pd.Timestamp(args.start)
        end_time = pd.Timestamp(args.end)
        # 如果 end 没指定时刻（即 00:00:00），包含当天 → +1天
        if end_time.hour == 0 and end_time.minute == 0 and end_time.second == 0:
            end_time += pd.Timedelta(days=1)
        mask = (df_query[time_col] >= start_time) & (df_query[time_col] < end_time)
        df_filtered = df_query[mask].reset_index(drop=True)
        range_label = f"{args.start}_to_{args.end}"
        print(f"\n[3] 筛选 {args.start} ~ {args.end}: {len(df_filtered)} 行")
    elif args.months:
        # 只处理第一个月份
        parts = args.months[0].split("-")
        year = int(parts[0])
        month = int(parts[1])
        mask = (df_query[time_col].dt.year == year) & (df_query[time_col].dt.month == month)
        df_filtered = df_query[mask].reset_index(drop=True)
        range_label = args.months[0]
        print(f"\n[3] 筛选 {args.months[0]}: {len(df_filtered)} 行")
    else:
        max_time = df_query[time_col].max()
        start_time = max_time - pd.Timedelta(days=30)
        mask = df_query[time_col] >= start_time
        df_filtered = df_query[mask].reset_index(drop=True)
        range_label = "last_30days"
        print(f"\n[3] 筛选最后30天: {len(df_filtered)} 行")

    print(f"\n[4] 开始评估（sample_step={args.sample_step}）...")
    results = evaluate_dtw(
        cfg=cfg,
        engine=engine,
        df_query=df_filtered,
        time_col=time_col,
        sample_step=args.sample_step,
        verbose=True,
    )

    # 打印报告
    print_dtw_report(results, range_label)

    # 保存报告
    save_dtw_report(results, args.output)

    print("\n=== DTW 评估完成 ===")
    return results


if __name__ == "__main__":
    main()
