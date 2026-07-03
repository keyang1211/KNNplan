# -*- coding: utf-8 -*-
"""
run_once.py — 单次查询示例：读取一行数据，输出 Top-5 结果到 CSV

用法:
    python plan_center/run_once.py                     # 使用默认配置
    python plan_center/run_once.py --row-index 100     # 指定查询数据的行号
    python plan_center/run_once.py --output result.csv # 指定输出路径
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="单次查询：输出 Top-5 结果到 CSV")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认 defaults.yaml）")
    parser.add_argument("--query-parquet", type=str, default=None, help="查询数据 parquet 路径")
    parser.add_argument("--row-index", type=int, default=0, help="查询数据的行号（默认 0）")
    parser.add_argument("--top-k", type=int, default=5, help="输出 Top-K（默认 5）")
    parser.add_argument("--output", type=str, default="plan_top5.csv", help="输出 CSV 路径")
    args = parser.parse_args()

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

    if args.row_index >= len(df_query):
        print(f"错误: 行号 {args.row_index} 超出范围（0 ~ {len(df_query)-1}）")
        sys.exit(1)

    # 3. 提取查询行
    row = df_query.iloc[args.row_index]
    print(f"\n[3] 查询行号: {args.row_index}")

    # 应用列别名映射
    aliases = engine.cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in df_query.columns and new not in df_query.columns:
                df_query[new] = df_query[old]
            elif old in df_query.columns and new in df_query.columns:
                df_query = df_query.drop(columns=[old])
        row = df_query.iloc[args.row_index]

    # 提取时间（用于输出）
    time_col = engine.cfg.time_col or "时间"
    input_time = row.get(time_col, None)

    raw_features = {}
    for c in engine.cfg.features.raw_features:
        raw_features[c] = float(row.get(c, 0.0))
    for c in engine.cfg.features.residual_inputs:
        if c not in raw_features:
            raw_features[c] = float(row.get(c, 0.0))
    raw_features[engine.cfg.features.load_col] = float(row.get(engine.cfg.features.load_col, 0.0))

    print("    查询特征:")
    for k, v in raw_features.items():
        print(f"      {k}: {v:.4f}")
    if input_time is not None:
        print(f"    查询时间: {input_time}")

    # 4. 单次查询（不含连续性）
    print(f"\n[4] 执行查询 (Top-{args.top_k})...")
    result = engine.plan_one_no_continuity(raw_features)

    print(f"    匹配状态: {result.match_status}")
    print(f"    Top-K 数量: {result.topk_count}")

    # 5. 构建 Top-K 输出
    print(f"\n[5] 构建 Top-{args.top_k} 输出...")

    # 重新计算 S 和 D（候选子集动态归一化方式）
    from plan_center.features import make_query_vector_15d
    from plan_center.config import build_feature_weights
    from plan_center.similarity import candidate_similarity, compute_and_normalize_candidates, flow_gate_keep_mask

    q_15d = make_query_vector_15d(raw_features, engine.models, engine.cfg.features)
    weights = build_feature_weights(engine.cfg.features)

    # 硬门控先筛选候选
    q_load = float(raw_features.get(engine.cfg.features.load_col, 0.0))
    keep_mask = flow_gate_keep_mask(q_load, engine.store.loads_standard, engine.cfg.flow_gate)
    valid_pos = np.where(keep_mask)[0]

    if len(valid_pos) == 0:
        print("无样本通过硬门控，回退到最近负荷")
        valid_pos = np.argsort(np.abs(engine.store.loads_standard - q_load))[:args.top_k]

    df_candidates = engine.store.df_standard.iloc[valid_pos]
    global_norm_stats = getattr(engine.store, 'norm_stats', None)
    s_candidates, effective_norm_stats, norm_source = compute_and_normalize_candidates(
        df_candidates, q_15d, engine.store.sim_feature_cols, weights, global_norm_stats
    )
    if np.any(np.isnan(s_candidates)):
        nan_mask = np.isnan(s_candidates)
        mean_val = np.nanmean(s_candidates)
        s_candidates = np.where(nan_mask, mean_val, s_candidates)

    d_candidates = engine.cfg.matching.d_weight_s * s_candidates + engine.cfg.matching.d_weight_e * engine.store.eff_score_all[valid_pos]
    d_valid = d_candidates
    order = np.argsort(d_valid)[::-1][:args.top_k]
    top_pos = valid_pos[order]

    top_indices = top_pos.tolist()
    top_s = s_candidates[order][:args.top_k]
    top_d = d_candidates[order][:args.top_k]
    top_e = engine.store.eff_score_all[top_pos]
    top_df = engine.store.df_standard.iloc[top_indices].copy()

    # 选择输出列
    output_cols = (
        ["稳定工况ID", "稳定窗口时间范围"]
        + [engine.cfg.features.load_col]
        + engine.cfg.features.raw_features
        + [f"resid_{t}" for t in engine.cfg.features.residual_targets]
        + [engine.cfg.features.eff_col]
    )
    output_cols = [c for c in output_cols if c in top_df.columns]

    out_df = top_df[output_cols].copy()
    out_df.insert(0, "排名", range(1, len(out_df) + 1))
    out_df["相似度S"] = top_s
    out_df["匹配度D"] = top_d
    out_df["效率分位数E"] = top_e

    # 在每行前面插入本次输入的详细信息（重复 Top-K 行数次，保持表格矩形）
    out_df.insert(0, "输入_行号", args.row_index)
    out_df.insert(1, "输入_时间", str(input_time) if input_time is not None else "")
    for i, (feat_name, feat_val) in enumerate(raw_features.items()):
        out_df.insert(2 + i, f"输入_{feat_name}", feat_val)

    # 6. 保存
    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[6] 已保存: {out_path}")

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"查询行: {args.row_index}")
    if input_time is not None:
        print(f"查询时间: {input_time}")
    print(f"匹配状态: {result.match_status}")
    print(f"Top-{args.top_k} 结果:")
    print(f"{'='*60}")
    for _, r in out_df.iterrows():
        print(
            f"  排名{int(r['排名'])}: "
            f"ID={r.get('稳定工况ID', 'N/A')}, "
            f"S={r['相似度S']:.4f}, "
            f"D={r['匹配度D']:.4f}, "
            f"效率={r.get(engine.cfg.features.eff_col, 'N/A'):.2f}"
        )


if __name__ == "__main__":
    main()
