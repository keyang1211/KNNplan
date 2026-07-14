# -*- coding: utf-8 -*-
"""
compute_dtw_norm_stats.py — 预计算 DTW cos 标准化参数（mean/std）并保存到文件

从全量分钟级残差缓存（#4_df_all_1min_with_resid.parquet）计算 sim_feature_cols
的 mean / std（z-score 参数），保存为 dtw_norm_stats.json。
查询时 DTWQueryEngine 直接加载，不再动态计算。

用法：
    python -m plan_center.compute_dtw_norm_stats
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from .config import load_config
from .dtw_query import ensure_resid_cache


def main():
    cfg = load_config()

    # 特征列：9 原始 + 6 残差 = 15 维
    all_feature_cols = cfg.features.raw_features + [f"resid_{t}" for t in cfg.features.residual_targets]
    sim_feature_cols = list(dict.fromkeys(all_feature_cols))  # 去重保序
    print(f"[DTW-norm] 特征列 ({len(sim_feature_cols)} 维): {sim_feature_cols}")

    # 1. 确保残差缓存存在
    alias_map = cfg.features.column_aliases or None
    cache_parquet = cfg.dtw_query.resid_cache_parquet
    cache_path = ensure_resid_cache(
        raw_parquet=cfg.paths.query_parquet,
        model_dir=cfg.paths.residual_model_dir,
        feat=cfg.features,
        cache_parquet=cache_parquet,
        feature_cols=sim_feature_cols,
        alias_map=alias_map,
    )

    # 2. 加载全量残差数据
    print(f"[DTW-norm] 加载全量残差数据: {cache_path}")
    t0 = time.time()
    df = pd.read_parquet(cache_path)
    print(f"[DTW-norm]   shape: {df.shape}，耗时 {time.time() - t0:.1f}s")

    # 列名别名映射
    if alias_map:
        for old, new in alias_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]

    # 3. 检查特征列是否存在
    missing = [c for c in sim_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"残差缓存中缺少特征列: {missing}")

    # 4. 计算 mean / std（z-score 参数，全量数据）
    print(f"[DTW-norm] 计算全量 mean / std ...")
    feat_matrix = df[sim_feature_cols].values.astype(float)

    col_mean = np.nanmean(feat_matrix, axis=0)
    # 填充 NaN 用于 std 计算
    inds = np.where(np.isnan(feat_matrix))
    feat_matrix_filled = feat_matrix.copy()
    feat_matrix_filled[inds] = np.take(col_mean, inds[1])
    col_std = np.nanstd(feat_matrix_filled, axis=0)
    col_std = np.where(col_std < 1e-10, 1.0, col_std)  # 防除0

    # 5. 保存到 JSON
    output_path = Path(cfg.paths.dtw_norm_stats_path) if cfg.paths.dtw_norm_stats_path else Path("plan_center/output/dtw_norm_stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    norm_data = {
        "mean": {col: float(col_mean[i]) for i, col in enumerate(sim_feature_cols)},
        "std": {col: float(col_std[i]) for i, col in enumerate(sim_feature_cols)},
        "feature_cols": sim_feature_cols,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(norm_data, f, ensure_ascii=False, indent=2)

    print(f"\n[DTW-norm] 标准化参数已保存: {output_path}")
    print(f"[DTW-norm] 各特征 mean / std:")
    for i, col in enumerate(sim_feature_cols):
        print(f"  {col}: mean={col_mean[i]:.4f}, std={col_std[i]:.4f}")


if __name__ == "__main__":
    main()
