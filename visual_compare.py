# -*- coding: utf-8 -*-
"""
visual_compare.py -- 三套权重配置可视化对比

对同一批查询数据分别用三套权重运行，生成 HTML 文件。

用法：
    python plan_center/visual_compare.py --start 2026-05-22 --end 2026-05-22
    python plan_center/visual_compare.py --dates 2026-05-22 2025-02-11
"""

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import yaml

from plan_center.validate_visual import (
    select_random_days,
    run_validation,
    build_plotly_figure,
    plot_validation,
    select_time_range,
)
from plan_center.config import load_config, build_feature_weights
from plan_center import PlanningEngine
from plan_center.schemas import build_output_dataframe
from plan_center.similarity import weighted_matrix


# ============================================================
# 三套权重配置
# ============================================================

ORIGINAL_WEIGHTS = {
    "主汽流量": 0.0, "主汽压力": 0.4, "炉膛差压": 0.5, "一次风流量": 0.4,
    "床温": 0.25, "料层差压": 0.30, "锅炉出口氧量": 0.20, "二次风风量": 0.20,
    "吨煤产气量": 0.35, "热值": 0.0,
}

CURRENT_WEIGHTS = {
    "主汽流量": 0.00, "吨煤产气量": 0.00, "主汽压力": 0.98, "炉膛差压": 0.50,
    "一次风流量": 0.49, "床温": 0.41, "料层差压": 0.57, "锅炉出口氧量": 0.42,
    "二次风风量": 0.30, "热值": 0.00,
}


def make_config_with_weights(base_config_path, new_weights):
    """
    基于 defaults.yaml 创建一个临时配置文件，其中 features.weights 被替换。
    返回临时文件路径，用完需手动删除。
    """
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg_data = yaml.safe_load(f)

    cfg_data["features"]["weights"] = dict(new_weights)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(cfg_data, tmp, allow_unicode=True, default_flow_style=False)
    tmp.close()
    return tmp.name


def get_weight_presets():
    """Return three weight configs: (name, weights_dict)."""
    new_w = dict(CURRENT_WEIGHTS)
    new_w["吨煤产气量"] = 0.1

    return [
        ("原始一套", dict(ORIGINAL_WEIGHTS)),
        ("现在一套", dict(CURRENT_WEIGHTS)),
        ("新方案_吨煤0.1", new_w),
    ]


def run_single_preset(engine, df_selected, time_col, output_dir, preset_name, weights_dict):
    """
    Run the validation pipeline for one weight preset.
    Creates a temp config, builds a fresh engine, runs validation, saves HTML.
    """
    print(f"\n{'='*60}")
    print(f"[{preset_name}]")
    print(f"  weights: {weights_dict}")
    print(f"{'='*60}")

    # Write temp config
    base_cfg_path = str(Path(__file__).parent / "defaults.yaml")
    tmp_cfg_path = make_config_with_weights(base_cfg_path, weights_dict)

    try:
        # Build fresh engine with temp config
        sub_engine = PlanningEngine(tmp_cfg_path)

        feat_cfg = sub_engine.cfg.features

        # Run validation
        results = run_validation(sub_engine, df_selected, time_col)

        # Build output DataFrame
        df_out = build_output_dataframe(
            raw_df=df_selected,
            results=results,
            plan_center_cols=feat_cfg.plan_center_cols,
        )

        # Save HTML
        safe_name = preset_name.replace("(", "_").replace(")", "").replace("=", "_").replace(",", "_").replace(" ", "_")
        output_path = output_dir / f"validate_{safe_name}.html"
        title = f"validate: {preset_name}"
        plot_validation(df_out, time_col, str(output_path), title=title)

        return output_path

    finally:
        Path(tmp_cfg_path).unlink(missing_ok=True)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="三套权重配置可视化对比")
    parser.add_argument("--start", type=str, default=None, help="起始时间（如 2026-05-22）")
    parser.add_argument("--end", type=str, default=None, help="终止时间（如 2026-05-22）")
    parser.add_argument("--dates", nargs="+", type=str, default=None,
                        help="批量日期，如 --dates 2026-05-22 2025-02-11")
    parser.add_argument("--days", type=int, default=1, help="随机模式下选取天数（默认1）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    # 1. Load base config + query data (read once)
    print("=== 三套权重可视化对比 ===\n")
    config_path = Path(args.config) if args.config else Path(__file__).parent / "defaults.yaml"
    base_cfg = load_config(str(config_path))

    query_parquet = base_cfg.paths.query_parquet
    df_query = pd.read_parquet(query_parquet)
    time_col = base_cfg.time_col or "时间"

    aliases = base_cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in df_query.columns and new not in df_query.columns:
                df_query[new] = df_query[old]
            elif old in df_query.columns and new in df_query.columns:
                df_query = df_query.drop(columns=[old])

    # 2. Output dir
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Determine batches
    batches = []  # [(start_str, end_str)]
    if args.dates:
        for d in args.dates:
            batches.append((f"{d} 00:00", f"{d} 23:59"))
    elif args.start and args.end:
        # ── 单次指定范围 ──
        # If bare date, expand to full day
        if " " not in args.start and ":" not in args.start:
            s_full = f"{args.start} 00:00"
            e_full = f"{args.end} 23:59"
        else:
            s_full = args.start
            e_full = args.end
        batches.append((s_full, e_full))
    else:
        if args.seed is not None:
            np.random.seed(args.seed)
        df_sel = select_random_days(df_query, time_col, days=args.days)
        t0 = df_sel[time_col].min()
        t1 = df_sel[time_col].max()
        batches.append(
            (t0.strftime("%Y-%m-%d %H:%M"), t1.strftime("%Y-%m-%d %H:%M"))
        )

    weight_presets = get_weight_presets()
    all_runs = []

    for start_str, end_str in batches:
        df_selected = select_time_range(df_query, time_col, start_str, end_str)
        range_str = f"{start_str}_to_{end_str}".replace(" ", "_").replace(":", "")
        print(f"\n批次: {range_str}，共 {len(df_selected)} 条")

        batch_dir = output_dir / range_str
        batch_dir.mkdir(parents=True, exist_ok=True)

        outputs = []
        for name, weights in weight_presets:
            try:
                out_path = run_single_preset(
                    engine=None,  # unused; each preset builds its own engine
                    df_selected=df_selected,
                    time_col=time_col,
                    output_dir=batch_dir,
                    preset_name=name,
                    weights_dict=weights,
                )
                print(f"  [OK] {name} -> {out_path}")
                outputs.append((name, str(out_path)))
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                import traceback; traceback.print_exc()

        all_runs.append((range_str, len(df_selected), outputs))

    # 4. Summary
    print(f"\n{'='*60}")
    print("三套权重可视化输出汇总：")
    for r_str, n_rows, outs in all_runs:
        print(f"\n  {r_str} ({n_rows} rows):")
        for nm, p in outs:
            print(f"    {nm}: {p}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
