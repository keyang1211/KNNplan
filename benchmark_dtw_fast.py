# -*- coding: utf-8 -*-
"""
benchmark_dtw_fast.py — DTW Baseline vs Fast 性能对比

测试两者的：
1. 单次查询耗时
2. Top-1 候选一致性
3. 相似度偏差

用法：
    python plan_center/benchmark_dtw_fast.py
    python plan_center/benchmark_dtw_fast.py --n-samples 10 --warmup
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


def run_benchmark(
    cfg,
    test_timestamps: list[str],
    warmup: bool = True,
    verbose: bool = True,
) -> dict:
    """运行 Baseline vs Fast 对比测试。"""
    from plan_center.config import load_config
    from plan_center.dtw_query import DTWQueryEngine

    engine = DTWQueryEngine(cfg)
    engine._ensure_data_loaded()

    # 预热 JIT（仅影响 Fast 模式）
    if warmup:
        if verbose:
            print("[预热] JIT 编译中...")
        t0 = time.perf_counter()
        engine.warmup(n_iter=3)
        warmup_time = time.perf_counter() - t0
        if verbose:
            print(f"[预热] 完成，耗时 {warmup_time:.2f}s\n")

    # Baseline 结果
    baseline_results = []
    if verbose:
        print("=" * 60)
        print("Baseline (Python) 模式")
        print("=" * 60)

    t0 = time.perf_counter()
    for ts in test_timestamps:
        start = time.perf_counter()
        result = engine.query_one(ts, verbose=False)
        elapsed = time.perf_counter() - start
        baseline_results.append({
            "ts": ts,
            "elapsed": elapsed,
            "best_idx": result.topk_indices[0] if result.topk_indices else None,
            "similarity_best": result.similarity_best,
            "plan_center": result.final_plan_center.copy(),
        })
        if verbose:
            print(f"  {ts}: {elapsed:.3f}s, sim={result.similarity_best:.4f}, best_idx={result.topk_indices[0] if result.topk_indices else 'N/A'}")

    baseline_total = time.perf_counter() - t0

    # Fast 结果
    fast_results = []
    if verbose:
        print("\n" + "=" * 60)
        print("Fast (Numba JIT) 模式")
        print("=" * 60)

    t0 = time.perf_counter()
    for ts in test_timestamps:
        start = time.perf_counter()
        result = engine.query_one_fast(ts, verbose=False)
        elapsed = time.perf_counter() - start
        fast_results.append({
            "ts": ts,
            "elapsed": elapsed,
            "best_idx": result.topk_indices[0] if result.topk_indices else None,
            "similarity_best": result.similarity_best,
            "plan_center": result.final_plan_center.copy(),
        })
        if verbose:
            print(f"  {ts}: {elapsed:.3f}s, sim={result.similarity_best:.4f}, best_idx={result.topk_indices[0] if result.topk_indices else 'N/A'}")

    fast_total = time.perf_counter() - t0

    # 汇总统计
    baseline_times = [r["elapsed"] for r in baseline_results]
    fast_times = [r["elapsed"] for r in fast_results]

    n_consistent = sum(
        1 for b, f in zip(baseline_results, fast_results)
        if b["best_idx"] == f["best_idx"]
    )

    sim_diffs = [
        abs(b["similarity_best"] - f["similarity_best"])
        for b, f in zip(baseline_results, fast_results)
    ]

    # 输出汇总
    if verbose:
        print("\n" + "=" * 60)
        print("性能对比汇总")
        print("=" * 60)
        print(f"测试样本数: {len(test_timestamps)}")
        print(f"预热耗时: {warmup_time:.2f}s")
        print()
        print(f"[Baseline] 总耗时: {baseline_total:.2f}s, 均值: {np.mean(baseline_times):.3f}s, "
              f"最大: {np.max(baseline_times):.3f}s, 最小: {np.min(baseline_times):.3f}s")
        print(f"[Fast]     总耗时: {fast_total:.2f}s, 均值: {np.mean(fast_times):.3f}s, "
              f"最大: {np.max(fast_times):.3f}s, 最小: {np.min(fast_times):.3f}s")
        print()
        if np.mean(fast_times) > 0:
            speedup = np.mean(baseline_times) / np.mean(fast_times)
            print(f"[加速比] {speedup:.1f}x")
        print()
        print(f"[Top-1 一致性] {n_consistent}/{len(test_timestamps)} ({100*n_consistent/len(test_timestamps):.1f}%)")
        print(f"[相似度偏差] 均值: {np.mean(sim_diffs):.6f}, 最大: {np.max(sim_diffs):.6f}")

    return {
        "baseline_times": baseline_times,
        "fast_times": fast_times,
        "n_consistent": n_consistent,
        "sim_diffs": sim_diffs,
        "baseline_total": baseline_total,
        "fast_total": fast_total,
        "speedup": np.mean(baseline_times) / np.mean(fast_times) if np.mean(fast_times) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="DTW Baseline vs Fast 性能对比")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--n-samples", type=int, default=5, help="测试样本数")
    parser.add_argument("--warmup", action="store_true", help="是否预热 JIT")
    parser.add_argument("--output", type=str, default=None, help="结果 CSV 输出路径")
    args = parser.parse_args()

    from plan_center.config import load_config

    cfg = load_config(args.config)

    print("=== DTW Baseline vs Fast 性能对比 ===\n")

    # 自动生成测试时间点（从数据末尾往前选取）
    query_parquet = cfg.paths.query_parquet
    if not query_parquet or not Path(query_parquet).exists():
        print(f"[错误] 查询数据不存在: {query_parquet}")
        return

    df_query = pd.read_parquet(query_parquet)
    time_col = cfg.time_col or "时间"

    # 确保时间列存在
    aliases = cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in df_query.columns and new not in df_query.columns:
                df_query[new] = df_query[old]

    if time_col not in df_query.columns:
        print(f"[错误] 时间列 '{time_col}' 不在数据中")
        return

    df_query[time_col] = pd.to_datetime(df_query[time_col], errors="coerce")
    df_query = df_query.dropna(subset=[time_col]).sort_values(time_col)

    # 从数据末尾选取测试时间点
    end_idx = len(df_query) - 1
    step = max(1, len(df_query) // (args.n_samples * 10))  # 间隔一定距离避免太近
    test_timestamps = []
    for i in range(args.n_samples):
        idx = max(0, end_idx - (i + 1) * step)
        ts = df_query.iloc[idx][time_col]
        test_timestamps.append(str(ts))

    print(f"选取 {len(test_timestamps)} 个测试时间点")
    for ts in test_timestamps:
        print(f"  - {ts}")
    print()

    # 运行 benchmark
    results = run_benchmark(cfg, test_timestamps, warmup=args.warmup)

    # 保存 CSV（可选）
    if args.output:
        rows = []
        for i, ts in enumerate(test_timestamps):
            rows.append({
                "timestamp": ts,
                "baseline_time": results["baseline_times"][i],
                "fast_time": results["fast_times"][i],
                "baseline_best_idx": None,  # 可从结果中获取
                "fast_best_idx": None,
                "similarity_diff": results["sim_diffs"][i],
            })
        df_out = pd.DataFrame(rows)
        df_out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存: {args.output}")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()