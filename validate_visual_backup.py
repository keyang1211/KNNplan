# -*- coding: utf-8 -*-
"""
validate_visual.py — 验证脚本：连续数据可视化

从分钟级查询数据中随机选取或按指定时间范围选取连续若干天，
逐分钟调用规划中心，将实际值与规划值对比可视化。

用法:
    python -m plan_center.validate_visual                         # 随机1天
    python -m plan_center.validate_visual --days 3                # 随机3天
    python -m plan_center.validate_visual --seed 42               # 随机1天（固定种子，可复现）
    python -m plan_center.validate_visual --start 2025-03-01 --end 2025-03-02   # 指定范围
    python -m plan_center.validate_visual --output result.html    # 指定输出路径
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def select_time_range(df: pd.DataFrame, time_col: str, start_str: str, end_str: str) -> pd.DataFrame:
    """
    按指定起止时间切取数据。

    参数：
        df: 分钟级查询数据
        time_col: 时间列名
        start_str: 起始时间字符串（如 "2025-03-01" 或 "2025-03-01 10:00"）
        end_str: 终止时间字符串（含，精确到分钟）

    返回：
        指定时间范围的子集 DataFrame
    """
    start_time = pd.Timestamp(start_str)
    end_time = pd.Timestamp(end_str)

    mask = (df[time_col] >= start_time) & (df[time_col] <= end_time)
    df_selected = df.loc[mask].sort_values(time_col).reset_index(drop=True)

    if len(df_selected) == 0:
        raise ValueError(f"指定时间范围内无数据: {start_time} ~ {end_time}")

    print(f"指定时间范围: {start_time} ~ {end_time}")
    print(f"总分钟数: {len(df_selected)}")
    return df_selected


def select_random_days(df: pd.DataFrame, time_col: str, days: int = 1, max_retries: int = 20) -> pd.DataFrame:
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
            print(f"随机选取的{days}天时间范围: {start_time} ~ {end_time}")
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
    print(f"随机选取的{days}天时间范围: {start_time} ~ {end_time}")
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


def plot_validation(df_out: pd.DataFrame, time_col: str, output_path: str, title: str = "规划中心验证：实际值 vs 规划值"):
    """
    画8个子图：实际值 vs 规划值 + S/D 对比。

    参数：
        df_out: 输出 DataFrame（含原始数据 + 规划中心 + 诊断）
        time_col: 时间列名
        output_path: HTML 输出路径
        title: 图标题
    """
    if len(df_out) == 0:
        print("数据为空，跳过绘图")
        return

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # 8个子图：实际值 vs 规划值（新增二次风风量）
    features = [
        ("主汽流量（负荷）", "主汽流量"),
        ("床温", "床温"),
        ("一次风流量", "一次风流量"),
        ("料层差压", "料层差压"),
        ("炉膛差压", "炉膛差压"),
        ("锅炉出口氧量", "锅炉出口氧量"),
        ("二次风风量", "二次风风量"),
    ]

    # 计算每个特征的默认纵轴范围（实际值 ± 5%）
    default_y_ranges = {}
    for name, col in features:
        actual = df_out[col].dropna()
        if len(actual) > 0:
            y_min, y_max = actual.min(), actual.max()
            margin = (y_max - y_min) * 0.1
            default_y_ranges[col] = [y_min - margin, y_max + margin]

    n_features = len(features)
    fig = make_subplots(
        rows=n_features + 1, cols=1,
        subplot_titles=[f[0] for f in features] + ["相似度S / 匹配度D"],
        vertical_spacing=0.04,
    )

    # 将时间列转为毫秒时间戳（避免base64二进制编码）
    time_values = pd.to_datetime(df_out[time_col]).values.astype('datetime64[ms]').astype(int)

    # 设置每个 xaxis 的 type 为 'date' 以正确显示时间
    for i in range(1, n_features + 2):
        fig.update_xaxes(type='date', row=i, col=1)

    for i, (name, col) in enumerate(features, start=1):
        actual = df_out[col]
        planned = df_out[f"规划中心_{col}"]

        # 实际值（蓝色实线）
        fig.add_trace(
            go.Scatter(
                x=time_values,
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
                x=time_values,
                y=planned,
                mode="lines",
                name="规划值" if i == 1 else None,
                line=dict(color="red", width=1, dash="dash"),
                legendgroup="planned",
                showlegend=(i == 1),
            ),
            row=i, col=1,
        )

        # 设置默认纵轴范围
        if col in default_y_ranges:
            y_range = default_y_ranges[col]
            fig.update_yaxes(range=y_range, row=i, col=1)

    # 第9个子图：S 和 D
    row_sd = n_features + 1
    fig.add_trace(
        go.Scatter(
            x=time_values,
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
            x=time_values,
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
            x=time_values,
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
            x=time_values,
            y=df_out["TopK_D均值"],
            mode="lines",
            name="匹配度D（TopK均值）",
            line=dict(color="orange", width=1, dash="dot"),
            legendgroup="score_d",
        ),
        row=row_sd, col=1,
    )
    # S/D图默认范围 0~1
    fig.update_yaxes(range=[0, 1.05], row=row_sd, col=1)

    fig.update_xaxes(title_text="时间", row=row_sd, col=1)
    fig.update_layout(
        title=title,
        height=2400,
        width=1400,
        legend=dict(x=0.01, y=0.99),
        font=dict(family="SimHei, Microsoft YaHei, Arial"),
    )

    # 生成features配置JSON
    import json

    def convert_to_native(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # 对每个 yaxis 设置 constrain: 'range'，使改变 range 时不改变 subplot 垂直位置
    yaxis_list = ['yaxis'] + [f'yaxis{i}' for i in range(2, n_features + 2)]
    for ya in yaxis_list:
        if ya in fig['layout']:
            fig['layout'][ya]['constrain'] = 'range'

    feat_json_list = []
    for i, (name, col) in enumerate(features):
        if col in default_y_ranges:
            y_min, y_max = default_y_ranges[col]
            y_min_f = float(y_min)
            y_max_f = float(y_max)
            margin = float((y_max_f - y_min_f) * 0.05)
            feat_json_list.append({
                "name": name,
                "col": col,
                "default_range": [y_min_f - margin, y_max_f + margin],
                "slider_min": float((y_min_f - margin) * 0.9),
                "slider_max": float((y_max_f + margin) * 1.1),
                "slider_step": float((y_max_f - y_min_f) / 100)
            })

    features_json = json.dumps(feat_json_list, ensure_ascii=False)
    plotly_json = json.dumps(fig.to_plotly_json(), ensure_ascii=False, allow_nan=True, default=convert_to_native)

    # HTML模板（使用JS模板字面量而非Python f-string）
    html_template = r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>__TITLE__</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: 'SimHei', 'Microsoft YaHei', Arial, sans-serif; margin: 10px; }
        .controls {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .control-row {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 10px;
            align-items: center;
        }
        .control-row input[type="range"] { width: 150px; }
        .btn-reset {
            padding: 5px 15px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        .btn-reset:hover { background: #45a049; }
        .feature-label { display: inline-block; min-width: 90px; }
    </style>
</head>
<body>
    <div class="controls">
        <div class="control-row">
            <strong>纵轴范围设置：</strong>
            <button class="btn-reset" onclick="resetYRanges()">重置默认范围</button>
        </div>
        <div class="control-row" id="sliders-container"></div>
    </div>
    <div id="plotly-chart"></div>
    <script>
    const featureConfig = __FEATURES_JSON__;
    const plotlyData = __PLOTLY_JSON__;
    // 记录每个子图的原始 domain 锁定垂直位置
    let fixedDomains = {};

    // 初始化图表
    Plotly.newPlot('plotly-chart', plotlyData.data, plotlyData.layout, {responsive: true});

    // 等待图表渲染完成后，记录 domain 并初始化滑块
    Plotly.relayout('plotly-chart', {}).then(function() {
        // 从渲染后的 layout 中获取 domain 并锁定
        Plotly.relayout('plotly-chart', {}).then(function(gd) {
            return Plotly.relayout(gd, {}).then(function(gd2) {
                // 另一种方式：直接用 initial layout 中的 domain
                const layout = plotlyData.layout;
                // feature 子图的 yaxis domain
                featureConfig.forEach(function(feat, idx) {
                    var yaxisName = idx === 0 ? 'yaxis' : 'yaxis' + (idx + 1);
                    if (layout[yaxisName] && layout[yaxisName].domain) {
                        fixedDomains[yaxisName] = layout[yaxisName].domain.slice();
                    }
                });
                // S/D 子图
                var nF = featureConfig.length;
                var sdYa = nF === 0 ? 'yaxis' : 'yaxis' + (nF + 1);
                if (layout[sdYa] && layout[sdYa].domain) {
                    fixedDomains[sdYa] = layout[sdYa].domain.slice();
                }
                initSlidersFromLayout();
            });
        });
    });

    function initSlidersFromLayout() {
        const container = document.getElementById('sliders-container');
        container.innerHTML = '';

        // 从 plotlyData.layout 中获取所有 yaxis 信息
        const layout = plotlyData.layout;

        featureConfig.forEach((feat, idx) => {
            // subplot 的 yaxis 命名：第一个是 'yaxis'，后续是 'yaxis2', 'yaxis3', ...
            let yaxisName = idx === 0 ? 'yaxis' : 'yaxis' + (idx + 1);

            const range = layout[yaxisName] && layout[yaxisName].range ? layout[yaxisName].range : feat.default_range;

            const div = document.createElement('div');
            div.style.cssText = 'display: flex; align-items: center; gap: 5px; margin-bottom: 5px;';
            div.innerHTML = `
                <span class="feature-label">${feat.name}：</span>
                <input type="range" id="min_${yaxisName}" min="${feat.slider_min}" max="${feat.slider_max}"
                    step="${feat.slider_step}" value="${range[0].toFixed(2)}"
                    oninput="updateYRange('${yaxisName}')">
                <span id="min_${yaxisName}_val">${range[0].toFixed(2)}</span>
                <span>~</span>
                <input type="range" id="max_${yaxisName}" min="${feat.slider_min}" max="${feat.slider_max}"
                    step="${feat.slider_step}" value="${range[1].toFixed(2)}"
                    oninput="updateYRange('${yaxisName}')">
                <span id="max_${yaxisName}_val">${range[1].toFixed(2)}</span>
            `;
            container.appendChild(div);
        });

        // S/D滑块 (最后一个是第 n_features + 1 个)
        const nFeatures = featureConfig.length;
        const sdYaxis = nFeatures === 0 ? 'yaxis' : 'yaxis' + (nFeatures + 1);
        const sdRange = layout[sdYaxis] && layout[sdYaxis].range
            ? layout[sdYaxis].range : [0, 1.05];
        const sdDiv = document.createElement('div');
        sdDiv.style.cssText = 'display: flex; align-items: center; gap: 5px; margin-bottom: 5px;';
        sdDiv.innerHTML = `
            <span class="feature-label">S/D 匹配度：</span>
            <input type="range" id="min_${sdYaxis}" min="0" max="1"
                step="0.01" value="${sdRange[0].toFixed(2)}"
                oninput="updateYRange('${sdYaxis}')">
            <span id="min_${sdYaxis}_val">${sdRange[0].toFixed(2)}</span>
            <span>~</span>
            <input type="range" id="max_${sdYaxis}" min="0" max="1"
                step="0.01" value="${sdRange[1].toFixed(2)}"
                oninput="updateYRange('${sdYaxis}')">
            <span id="max_${sdYaxis}_val">${sdRange[1].toFixed(2)}</span>
        `;
        container.appendChild(sdDiv);
    }

    function updateYRange(yaxisName) {
        const minInput = document.getElementById('min_' + yaxisName);
        const maxInput = document.getElementById('max_' + yaxisName);
        if (!minInput || !maxInput) {
            console.log('Slider not found for:', yaxisName);
            return;
        }

        const minVal = parseFloat(minInput.value);
        const maxVal = parseFloat(maxInput.value);

        // 更新显示值
        document.getElementById('min_' + yaxisName + '_val').textContent = minVal.toFixed(2);
        document.getElementById('max_' + yaxisName + '_val').textContent = maxVal.toFixed(2);

        // 更新图表：设置 range 同时锁定 domain，防止子图位置漂移
        const update = {};
        update[yaxisName] = {range: [minVal, maxVal]};
        if (fixedDomains[yaxisName]) {
            update[yaxisName].domain = fixedDomains[yaxisName];
        }
        Plotly.relayout('plotly-chart', update);
    }

    function resetYRanges() {
        featureConfig.forEach((feat, idx) => {
            const yaxisName = idx === 0 ? 'yaxis' : 'yaxis' + (idx + 1);
            updateSliderInputs(yaxisName, feat.default_range);
        });
        // S/D 重置
        const nFeatures = featureConfig.length;
        const sdYaxis = nFeatures === 0 ? 'yaxis' : 'yaxis' + (nFeatures + 1);
        updateSliderInputs(sdYaxis, [0, 1.05]);
    }

    function updateSliderInputs(yaxisName, range) {
        const minInput = document.getElementById('min_' + yaxisName);
        const maxInput = document.getElementById('max_' + yaxisName);
        if (minInput && maxInput) {
            minInput.value = range[0].toFixed(2);
            maxInput.value = range[1].toFixed(2);
            document.getElementById('min_' + yaxisName + '_val').textContent = range[0].toFixed(2);
            document.getElementById('max_' + yaxisName + '_val').textContent = range[1].toFixed(2);
        }
        // 更新图表：设置 range 同时锁定 domain
        const update = {};
        update[yaxisName] = {range: range};
        if (fixedDomains[yaxisName]) {
            update[yaxisName].domain = fixedDomains[yaxisName];
        }
        Plotly.relayout('plotly-chart', update);
    }
    </script>
</body>
</html>"""

    html_content = html_template.replace("__TITLE__", title)
    html_content = html_content.replace("__FEATURES_JSON__", features_json)
    html_content = html_content.replace("__PLOTLY_JSON__", plotly_json)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n可视化已保存: {output_path}")
    print(f"包含二次风风量 + 交互式纵轴范围滑块")


def main():
    parser = argparse.ArgumentParser(description="规划中心验证：实际值 vs 规划值可视化")
    parser.add_argument("--start", type=str, default=None,
                        help="起始时间（如 '2025-03-01' 或 '2025-03-01 10:00'），与 --end 配合使用")
    parser.add_argument("--end", type=str, default=None,
                        help="终止时间（如 '2025-03-02' 或 '2025-03-02 10:00'），与 --start 配合使用")
    parser.add_argument("--days", type=int, default=1,
                        help="随机模式下选取的天数（默认 1；--start/--end 指定时忽略此参数）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，固定后可复现（仅随机模式有效）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 HTML 路径（默认按时间范围自动命名）")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（默认 defaults.yaml）")
    args = parser.parse_args()

    use_specified_range = bool(args.start and args.end)

    print("=== 验证脚本：连续数据可视化 ===\n")

    # 1. 加载配置 + 构建引擎
    print("[1] 加载 PlanningEngine...")
    from plan_center import PlanningEngine

    config_path = Path(args.config) if args.config else Path(__file__).parent / "defaults.yaml"
    if not config_path.exists():
        print(f"错误: 配置文件不存在 {config_path}")
        sys.exit(1)

    engine = PlanningEngine(str(config_path))
    print(f"    {engine}")

    # 2. 读取分钟级查询 parquet
    print("\n[2] 读取分钟级查询数据...")
    query_parquet = engine.cfg.paths.query_parquet
    if not query_parquet or not Path(query_parquet).exists():
        print(f"错误: 查询数据路径不存在 {query_parquet}")
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

    time_col = engine.cfg.time_col or "时间"

    # 3. 选取数据
    if use_specified_range:
        print(f"\n[3] 按指定时间范围选取数据: {args.start} ~ {args.end}")
        df_selected = select_time_range(df_query, time_col, args.start, args.end)
        start_label = pd.Timestamp(args.start).strftime("%Y-%m-%d")
        end_label = pd.Timestamp(args.end).strftime("%Y-%m-%d")
        range_str = f"{start_label}_to_{end_label}"
        title = f"规划中心验证：实际值 vs 规划值（{args.start} ~ {args.end}）"
    else:
        print(f"\n[3] 随机选取连续{args.days}天数据...")
        if args.seed is not None:
            np.random.seed(args.seed)
            print(f"    随机种子: {args.seed}")
        df_selected = select_random_days(df_query, time_col, days=args.days)
        t_start = df_selected[time_col].min()
        t_end = df_selected[time_col].max()
        range_str = f"{t_start.strftime('%Y-%m-%d')}_{args.days}days"
        title = f"规划中心验证：实际值 vs 规划值（随机{args.days}天，{t_start.strftime('%Y-%m-%d %H:%M')} ~ {t_end.strftime('%Y-%m-%d %H:%M')}）"

    # 4. 逐分钟调用 plan_one
    print("\n[4] 逐分钟调用 plan_one...")
    results = run_validation(engine, df_selected, time_col)

    # 5. 构建输出 DataFrame
    print("\n[5] 构建输出 DataFrame...")
    from plan_center.schemas import build_output_dataframe

    df_out = build_output_dataframe(
        raw_df=df_selected,
        results=results,
        plan_center_cols=engine.cfg.features.plan_center_cols,
    )
    print(f"    输出形状: {df_out.shape}")

    if len(df_out) == 0:
        print("\n警告: 选取的数据为空，跳过可视化")
        return

    # 6. 可视化
    print("\n[6] 生成可视化...")
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / f"validate_{range_str}.html"

    plot_validation(df_out, time_col, str(output_path), title=title)

    print("\n=== 验证完成 ===")
    print(f"时间范围: {df_selected[time_col].min()} ~ {df_selected[time_col].max()}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
