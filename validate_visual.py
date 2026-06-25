# -*- coding: utf-8 -*-
"""
validate_visual.py — 验证脚本：7天连续数据可视化

从分钟级查询数据中随机选取连续7天，逐分钟调用规划中心，
将实际值与规划值对比可视化。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def select_random_7days(df: pd.DataFrame, time_col: str, days: int = 1, max_retries: int = 20) -> pd.DataFrame:
    """
    从 df 中随机选取连续N天数据。

    参数：
        df: 分钟级查询数据
        time_col: 时间列名
        days: 天数（默认1天）
        max_retries: 最大重试次数（避免选到空区间）

    返回：
        连续N天的子集 DataFrame
    """
    df = df.sort_values(time_col).reset_index(drop=True)

    t_min = df[time_col].min()
    t_max = df[time_col].max()
    total_minutes = (t_max - t_min).total_seconds() / 60

    # N天 = N * 24 * 60 分钟
    target_minutes = days * 24 * 60

    if total_minutes < target_minutes:
        raise ValueError(f"数据不足{days}天，只有 {total_minutes / 60 / 24:.1f} 天")

    # 随机选起始时间，重试直到选到非空区间
    max_start_minutes = total_minutes - target_minutes
    for attempt in range(max_retries):
        start_offset = np.random.randint(0, int(max_start_minutes))
        start_time = t_min + pd.Timedelta(minutes=start_offset)
        end_time = start_time + pd.Timedelta(minutes=target_minutes)

        mask = (df[time_col] >= start_time) & (df[time_col] < end_time)
        df_selected = df.loc[mask].reset_index(drop=True)

        if len(df_selected) > 0:
            print(f"选取的{days}天时间范围: {start_time} ~ {end_time}")
            print(f"总分钟数: {len(df_selected)}")
            return df_selected

    # 所有重试都为空，回退：取数据最密集的连续N天窗口
    print(f"警告: 随机选取{max_retries}次均为空区间，改用最密集窗口")
    # 按小时分桶，找数据量最大的起始小时
    df_h = df.set_index(time_col).resample("1h").size()
    if df_h.sum() == 0:
        raise ValueError("数据全为空，无法选取")
    best_hour = df_h.rolling(window=max(1, int(target_minutes / 60)), min_periods=1).sum().idxmax()
    start_time = pd.Timestamp(best_hour)
    end_time = start_time + pd.Timedelta(minutes=target_minutes)
    mask = (df[time_col] >= start_time) & (df[time_col] < end_time)
    df_selected = df.loc[mask].reset_index(drop=True)
    print(f"选取的{days}天时间范围: {start_time} ~ {end_time}")
    print(f"总分钟数: {len(df_selected)}")
    return df_selected


def run_validation(engine, df_7days: pd.DataFrame, time_col: str) -> list:
    """
    逐分钟调用 plan_one，收集结果。

    参数：
        engine: PlanningEngine 实例
        df_7days: 连续7天的数据
        time_col: 时间列名

    返回：
        PlanResult 列表
    """
    results = []
    prev_center = None
    prev_time = None

    total = len(df_7days)
    print(f"开始逐分钟调用，共 {total} 条数据...")

    for idx, row in df_7days.iterrows():
        current_time = row[time_col]

        # 提取原始特征
        raw_features = {}
        for c in engine.cfg.features.raw_features:
            raw_features[c] = float(row.get(c, 0.0))
        raw_features[engine.cfg.features.load_col] = float(row.get(engine.cfg.features.load_col, 0.0))

        # 调用
        result = engine.plan_one(
            raw_features=raw_features,
            prev_center=prev_center,
            prev_time=prev_time,
            current_time=current_time,
        )

        results.append(result)

        # 更新状态
        if result.final_plan_center:
            prev_center = result.final_plan_center.copy()
            prev_time = current_time

        # 进度提示
        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1}/{total} ({(idx + 1) / total * 100:.1f}%)")

    # 统计匹配状态
    status_counts = pd.Series([r.match_status for r in results]).value_counts()
    print(f"\n匹配状态统计:")
    for status, count in status_counts.items():
        print(f"  {status}: {count} ({count / total * 100:.1f}%)")

    return results


def plot_validation(df_out: pd.DataFrame, time_col: str, output_path: str):
    """
    画8个子图：实际值 vs 规划值 + S/D 对比。

    参数：
        df_out: 输出 DataFrame（含原始数据 + 规划中心 + 诊断）
        time_col: 时间列名
        output_path: HTML 输出路径
    """
    if len(df_out) == 0:
        print("数据为空，跳过绘图")
        return

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # 8个子图：实际值 vs 规划值
    features = [
        ("主汽流量（负荷）", "主汽流量"),
        ("床温", "床温"),
        ("一次风流量", "一次风流量"),
        ("料层差压", "料层差压"),
        ("炉膛差压", "炉膛差压"),
        ("锅炉出口氧量", "锅炉出口氧量"),
    ]

    fig = make_subplots(
        rows=8, cols=1,
        subplot_titles=[f[0] for f in features] + ["相似度S / 匹配度D"],
        vertical_spacing=0.04,
    )

    for i, (name, col) in enumerate(features, start=1):
        actual = df_out[col]
        planned = df_out[f"规划中心_{col}"]

        # 实际值（蓝色实线）
        fig.add_trace(
            go.Scatter(
                x=df_out[time_col],
                y=actual,
                mode="lines",
                name="实际值" if i == 1 else None,
                line=dict(color="blue", width=1),
                legendgroup="actual",
                showlegend=(i == 1),
            ),
            row=i, col=1,
        )

        # 规划值（红色虚线）
        fig.add_trace(
            go.Scatter(
                x=df_out[time_col],
                y=planned,
                mode="lines",
                name="规划值" if i == 1 else None,
                line=dict(color="red", width=1, dash="dash"),
                legendgroup="planned",
                showlegend=(i == 1),
            ),
            row=i, col=1,
        )

        fig.update_yaxes(title_text=name, row=i, col=1)

    # 第8个子图：S 和 D
    row_sd = len(features) + 1
    fig.add_trace(
        go.Scatter(
            x=df_out[time_col],
            y=df_out["相似度S"],
            mode="lines",
            name="相似度S（Best）",
            line=dict(color="green", width=1),
            legendgroup="sim",
        ),
        row=row_sd, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_out[time_col],
            y=df_out["TopK_S均值"],
            mode="lines",
            name="相似度S（TopK均值）",
            line=dict(color="green", width=1, dash="dot"),
            legendgroup="sim",
        ),
        row=row_sd, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_out[time_col],
            y=df_out["匹配度D"],
            mode="lines",
            name="匹配度D（Best）",
            line=dict(color="orange", width=1),
            legendgroup="score_d",
        ),
        row=row_sd, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_out[time_col],
            y=df_out["TopK_D均值"],
            mode="lines",
            name="匹配度D（TopK均值）",
            line=dict(color="orange", width=1, dash="dot"),
            legendgroup="score_d",
        ),
        row=row_sd, col=1,
    )
    fig.update_yaxes(title_text="S / D", row=row_sd, col=1)

    fig.update_xaxes(title_text="时间", row=row_sd, col=1)
    fig.update_layout(
        title="规划中心验证：实际值 vs 规划值（连续1天）",
        height=2000,
        width=1400,
        legend=dict(x=0.01, y=0.99),
        font=dict(family="SimHei, Microsoft YaHei, Arial"),
    )

    fig.write_html(output_path)
    print(f"\n可视化已保存: {output_path}")


def main():
    print("=== 验证脚本：7天连续数据可视化 ===\n")

    # 1. 加载配置 + 构建引擎
    print("[1] 加载 PlanningEngine...")
    from plan_center import PlanningEngine

    # 使用默认配置（需要用户修改 defaults.yaml 中的路径）
    config_path = Path(__file__).parent / "defaults.yaml"
    if not config_path.exists():
        print(f"错误: 配置文件不存在 {config_path}")
        print("请先修改 defaults.yaml 中的路径配置")
        sys.exit(1)

    engine = PlanningEngine(str(config_path))
    print(f"    {engine}")

    # 2. 读取分钟级查询 parquet
    print("\n[2] 读取分钟级查询数据...")
    query_parquet = engine.cfg.paths.query_parquet
    if not query_parquet or not Path(query_parquet).exists():
        print(f"错误: 查询数据路径不存在 {query_parquet}")
        print("请在 defaults.yaml 中设置 paths.query_parquet")
        sys.exit(1)

    df_query = pd.read_parquet(query_parquet)
    print(f"    查询数据形状: {df_query.shape}")

    # 应用列别名映射
    aliases = engine.cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in df_query.columns and new not in df_query.columns:
                df_query[new] = df_query[old]
            elif old in df_query.columns and new in df_query.columns:
                df_query = df_query.drop(columns=[old])

    # 3. 随机选取连续1天数据
    print("\n[3] 随机选取连续1天数据...")
    time_col = engine.cfg.time_col or "时间"
    df_7days = select_random_7days(df_query, time_col, days=1)

    # 4. 逐分钟调用 plan_one
    print("\n[4] 逐分钟调用 plan_one...")
    results = run_validation(engine, df_7days, time_col)

    # 5. 构建输出 DataFrame
    print("\n[5] 构建输出 DataFrame...")
    from plan_center.schemas import build_output_dataframe

    df_out = build_output_dataframe(
        raw_df=df_7days,
        results=results,
        plan_center_cols=engine.cfg.features.plan_center_cols,
    )
    print(f"    输出形状: {df_out.shape}")

    if len(df_out) == 0:
        print("\n警告: 选取的数据为空，跳过可视化")
        return

    # 6. 可视化
    print("\n[6] 生成可视化...")
    output_path = Path(__file__).parent / "validate_visual_1day.html"
    plot_validation(df_out, time_col, str(output_path))

    print("\n=== 验证完成 ===")


if __name__ == "__main__":
    main()
