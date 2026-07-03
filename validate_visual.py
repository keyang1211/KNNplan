# -*- coding: utf-8 -*-
"""
validate_visual.py — 验证脚本：连续数据可视化（重写版）

从分钟级查询数据中随机选取或按指定时间范围选取连续若干天，
逐分钟调用规划中心，将实际值与规划值对比可视化。

修复:
- f-string {} 转义问题 → 用 str.replace() 模板替换
- xaxis 命名混乱 → 用 row=i, col=1 让 Plotly 自动管理
- subplot 漂移 → relayout 同时固定 domain
- JSON 序列化 → NumpyEncoder
- 时间轴显示 → type='date' + ms 时间戳

用法:
    python -m plan_center.validate_visual
    python -m plan_center.validate_visual --days 3
    python -m plan_center.validate_visual --start 2025-03-01 --end 2025-03-02
    python -m plan_center.validate_visual --output result.html
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────
# Numpy JSON encoder
# ──────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ──────────────────────────────────────────────
# 数据选取
# ──────────────────────────────────────────────
def select_time_range(df, time_col, start_str, end_str):
    start_time = pd.Timestamp(start_str)
    end_time = pd.Timestamp(end_str)
    mask = (df[time_col] >= start_time) & (df[time_col] <= end_time)
    df_selected = df.loc[mask].sort_values(time_col).reset_index(drop=True)
    if len(df_selected) == 0:
        raise ValueError(f"指定时间范围内无数据: {start_time} ~ {end_time}")
    print(f"指定时间范围: {start_time} ~ {end_time}")
    print(f"总分钟数: {len(df_selected)}")
    return df_selected


def select_random_days(df, time_col, days=1, max_retries=20):
    df = df.sort_values(time_col).reset_index(drop=True)
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    total_minutes = int((t_max - t_min).total_seconds() / 60)
    target_minutes = days * 24 * 60
    if total_minutes < target_minutes:
        raise ValueError(f"数据不足{days}天，只有 {total_minutes / 60 / 24:.1f} 天")
    max_start_minutes = total_minutes - target_minutes
    for _ in range(max_retries):
        start_offset = np.random.randint(0, max_start_minutes)
        start_time = t_min + pd.Timedelta(minutes=start_offset)
        end_time = start_time + pd.Timedelta(minutes=target_minutes)
        mask = (df[time_col] >= start_time) & (df[time_col] < end_time)
        df_selected = df.loc[mask].reset_index(drop=True)
        if len(df_selected) > 0:
            print(f"随机选取的{days}天时间范围: {start_time} ~ {end_time}")
            print(f"总分钟数: {len(df_selected)}")
            return df_selected
    # fallback
    print(f"警告: 随机选取{max_retries}次均为空区间，改用最密集窗口")
    df_h = df.set_index(time_col).resample("1h").size()
    best_hour = df_h.rolling(max(1, target_minutes // 60), min_periods=1).sum().idxmax()
    start_time = pd.Timestamp(best_hour)
    end_time = start_time + pd.Timedelta(minutes=target_minutes)
    mask = (df[time_col] >= start_time) & (df[time_col] < end_time)
    df_selected = df.loc[mask].reset_index(drop=True)
    print(f"随机选取的{days}天时间范围: {start_time} ~ {end_time}")
    print(f"总分钟数: {len(df_selected)}")
    return df_selected


# ──────────────────────────────────────────────
# 逐分钟调用
# ──────────────────────────────────────────────
def run_validation(engine, df, time_col):
    results = []
    prev_center = None
    prev_time = None
    total = len(df)
    print(f"开始逐分钟调用，共 {total} 条数据...")
    for idx, row in df.iterrows():
        current_time = row[time_col]
        raw_features = {}
        for c in engine.cfg.features.raw_features:
            raw_features[c] = float(row.get(c, 0.0))
        raw_features[engine.cfg.features.load_col] = float(row.get(engine.cfg.features.load_col, 0.0))
        result = engine.plan_one(
            raw_features=raw_features,
            prev_center=prev_center,
            prev_time=prev_time,
            current_time=current_time,
        )

        # ── 时间差计算（天）──
        query_time = pd.Timestamp(current_time)
        top1_diff = np.nan
        top5_diffs = []

        if result.best_index is not None and hasattr(engine, 'store'):
            # Top-1: 匹配窗口中点 → 时间差
            try:
                best_row = engine.store.df_standard.iloc[result.best_index]
                win_raw = best_row.get("稳定窗口时间范围", "")
                if win_raw:
                    parts = str(win_raw).split("~")
                    if len(parts) == 2:
                        t_s = pd.Timestamp(parts[0].strip())
                        t_e = pd.Timestamp(parts[1].strip())
                        t_mid = t_s + (t_e - t_s) / 2
                        top1_diff = abs((query_time - t_mid).total_seconds()) / 86400.0
            except Exception:
                top1_diff = np.nan

        # Top-5: 各匹配窗口中点 → 平均时间差
        for ti in result.topk_indices:
            try:
                if hasattr(engine, 'store') and ti < len(engine.store.df_standard):
                    k_row = engine.store.df_standard.iloc[ti]
                    k_win = k_row.get("稳定窗口时间范围", "")
                    if k_win:
                        parts = str(k_win).split("~")
                        if len(parts) == 2:
                            t_s = pd.Timestamp(parts[0].strip())
                            t_e = pd.Timestamp(parts[1].strip())
                            t_mid = t_s + (t_e - t_s) / 2
                            top5_diffs.append(abs((query_time - t_mid).total_seconds()) / 86400.0)
            except Exception:
                pass

        top5_mean_diff = float(np.mean(top5_diffs)) if top5_diffs else np.nan
        result._top1_time_diff = top1_diff
        result._top5_mean_time_diff = top5_mean_diff

        results.append(result)
        if result.final_plan_center:
            prev_center = result.final_plan_center.copy()
            prev_time = current_time
        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1}/{total} ({(idx + 1) / total * 100:.1f}%)")
    status_counts = pd.Series([r.match_status for r in results]).value_counts()
    print(f"\n匹配状态统计:")
    for status, count in status_counts.items():
        print(f"  {status}: {count} ({count / total * 100:.1f}%)")
    return results


# ──────────────────────────────────────────────
# 构建 Plotly Figure
# ──────────────────────────────────────────────
def build_plotly_figure(df_out, time_col, plan_center_cols):
    """构造 10 行 subplot 的 Plotly Figure（新增时间差子图）"""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # 特征列表（排除负荷/热值，选 plan_center_cols 中有规划中心值的）
    features = [
        ("主汽流量（负荷）", "主汽流量"),
        ("床温", "床温"),
        ("一次风流量", "一次风流量"),
        ("料层差压", "料层差压"),
        ("炉膛差压", "炉膛差压"),
        ("锅炉出口氧量", "锅炉出口氧量"),
        ("二次风风量", "二次风风量"),
    ]

    # 计算默认纵轴范围
    default_y_ranges = {}
    for name, col in features:
        actual = df_out[col].dropna()
        if len(actual) > 0:
            y_min, y_max = float(actual.min()), float(actual.max())
            margin = (y_max - y_min) * 0.1
            default_y_ranges[col] = [y_min - margin, y_max + margin]

    n_features = len(features)

    # ── 提取时间差数据（可能全 NaN，若无对应列则跳过） ──
    top1_time_diffs = df_out.get("Top1时间差_天", pd.Series(dtype=float)).values.astype(float).tolist()
    top5_time_diffs = df_out.get("Top5平均时间差_天", pd.Series(dtype=float)).values.astype(float).tolist()
    has_time_diff = any(not np.isnan(v) for v in top1_time_diffs + top5_time_diffs)

    # 子图：7 特征 + 时间差（可选） + S/D
    extra_rows = 1 if has_time_diff else 0
    n_rows = n_features + extra_rows + 1  # 特征 + 时间差 + S/D

    subplot_titles = [f[0] for f in features]
    if has_time_diff:
        subplot_titles.append("匹配时间差（天）")
    subplot_titles.append("相似度S / 匹配度D")

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
    )

    # 时间值（ms时间戳 + type='date'）
    time_values = pd.to_datetime(df_out[time_col]).values.astype("datetime64[ms]").astype(int).tolist()

    # ── 7 个特征子图 ──
    for i, (name, col) in enumerate(features, start=1):
        actual = df_out[col].values.astype(float).tolist()
        planned = df_out[f"规划中心_{col}"].values.astype(float).tolist()
        fig.add_trace(
            go.Scatter(x=time_values, y=actual, mode="lines",
                       name="实际值" if i == 1 else None,
                       line=dict(color="blue", width=1),
                       legendgroup="actual", showlegend=(i == 1)),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(x=time_values, y=planned, mode="lines",
                       name="规划值" if i == 1 else None,
                       line=dict(color="red", width=1, dash="dash"),
                       legendgroup="planned", showlegend=(i == 1)),
            row=i, col=1,
        )
        if col in default_y_ranges:
            fig.update_yaxes(range=default_y_ranges[col], row=i, col=1)

    # ── 时间差子图 ──
    row_time = n_features + 1  # 第8行（若有时间差）
    if has_time_diff:
        fig.add_trace(
            go.Scatter(x=time_values, y=top1_time_diffs, mode="lines",
                       name="Top-1时间差", line=dict(color="purple", width=1)),
            row=row_time, col=1,
        )
        fig.add_trace(
            go.Scatter(x=time_values, y=top5_time_diffs, mode="lines",
                       name="Top-5平均时间差", line=dict(color="brown", width=1, dash="dot")),
            row=row_time, col=1,
        )
        fig.update_yaxes(title_text="天", row=row_time, col=1)
        row_sd = n_features + 2   # S/D 子图移至第9行
    else:
        row_sd = n_features + 1   # S/D 子图仍在第8行

    # ── S/D 子图 ──
    for trace_data in [
        (df_out["相似度S"].values.astype(float).tolist(), "相似度S（Best）", "green", None),
        (df_out["TopK_S均值"].values.astype(float).tolist(), "相似度S（TopK均值）", "green", "dot"),
        (df_out["匹配度D"].values.astype(float).tolist(), "匹配度D（Best）", "orange", None),
        (df_out["TopK_D均值"].values.astype(float).tolist(), "匹配度D（TopK均值）", "orange", "dot"),
    ]:
        y_data, name, color, dash = trace_data
        fig.add_trace(
            go.Scatter(x=time_values, y=y_data, mode="lines",
                       name=name, line=dict(color=color, width=1, dash=dash)),
            row=row_sd, col=1,
        )
    fig.update_yaxes(range=[0, 1.05], row=row_sd, col=1)

    # 所有 xaxis 设为 date 类型
    for i in range(1, n_rows + 1):
        fig.update_xaxes(type="date", row=i, col=1)

    # 锁定每个 yaxis 的 domain（防漂移）
    yaxis_keys = ["yaxis"] + [f"yaxis{j}" for j in range(2, n_rows + 1)]
    for ya in yaxis_keys:
        if ya in fig["layout"]:
            fig["layout"][ya]["constrain"] = "range"

    fig.update_layout(
        title="",
        height=2400, width=1400,
        legend=dict(x=0.01, y=0.99),
        font=dict(family="SimHei, Microsoft YaHei, Arial"),
    )
    return fig, features, default_y_ranges


# ──────────────────────────────────────────────
# 生成 HTML
# ──────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>__TITLE__</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: 'SimHei', 'Microsoft YaHei', Arial, sans-serif; margin: 10px; }
        .controls { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .control-row { display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 8px; align-items: center; }
        .control-row input[type="range"] { width: 150px; }
        .btn-reset { padding: 5px 15px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; }
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
    const plotlyData = __PLOTLY_JSON__;
    const featureConfig = __FEATURES_JSON__;
    const fixedDomains = {};

    Plotly.newPlot('plotly-chart', plotlyData.data, plotlyData.layout, {responsive: true});
    Plotly.relayout('plotly-chart', {}).then(function() {
        lockDomains();
        initSliders();
    });

    function lockDomains() {
        const layout = plotlyData.layout;
        featureConfig.forEach(function(feat, idx) {
            var ya = idx === 0 ? 'yaxis' : 'yaxis' + (idx + 1);
            if (layout[ya] && layout[ya].domain) {
                fixedDomains[ya] = layout[ya].domain.slice();
            }
        });
        var n = featureConfig.length;
        var sdYa = n === 0 ? 'yaxis' : 'yaxis' + (n + 1);
        if (layout[sdYa] && layout[sdYa].domain) {
            fixedDomains[sdYa] = layout[sdYa].domain.slice();
        }
    }

    function initSliders() {
        var container = document.getElementById('sliders-container');
        container.innerHTML = '';
        featureConfig.forEach(function(feat, idx) {
            var ya = idx === 0 ? 'yaxis' : 'yaxis' + (idx + 1);
            var range = plotlyData.layout[ya] && plotlyData.layout[ya].range
                ? plotlyData.layout[ya].range : feat.default_range;
            addSlider(container, feat.name, ya, range, feat);
        });
        // S/D 滑块
        var n = featureConfig.length;
        var sdYa = n === 0 ? 'yaxis' : 'yaxis' + (n + 1);
        var sdRange = plotlyData.layout[sdYa] && plotlyData.layout[sdYa].range
            ? plotlyData.layout[sdYa].range : [0, 1.05];
        addSlider(container, 'S/D 匹配度', sdYa, sdRange, null);
    }

    function addSlider(container, name, ya, range, feat) {
        var smin = feat ? feat.slider_min : 0;
        var smax = feat ? feat.slider_max : 1;
        var sstep = feat ? feat.slider_step : 0.01;
        var div = document.createElement('div');
        div.style.cssText = 'display: flex; align-items: center; gap: 5px; margin-bottom: 5px;';
        div.innerHTML =
            '<span class="feature-label">' + name + '：</span>' +
            '<input type="range" id="min_' + ya + '" min="' + smin + '" max="' + smax + '"' +
            ' step="' + sstep + '" value="' + range[0].toFixed(2) + '"' +
            ' oninput="updateYRange(\'' + ya + '\')">' +
            '<span id="min_' + ya + '_val">' + range[0].toFixed(2) + '</span>' +
            '<span>~</span>' +
            '<input type="range" id="max_' + ya + '" min="' + smin + '" max="' + smax + '"' +
            ' step="' + sstep + '" value="' + range[1].toFixed(2) + '"' +
            ' oninput="updateYRange(\'' + ya + '\')">' +
            '<span id="max_' + ya + '_val">' + range[1].toFixed(2) + '</span>';
        container.appendChild(div);
    }

    function updateYRange(ya) {
        var minInput = document.getElementById('min_' + ya);
        var maxInput = document.getElementById('max_' + ya);
        if (!minInput || !maxInput) return;
        var minVal = parseFloat(minInput.value);
        var maxVal = parseFloat(maxInput.value);
        document.getElementById('min_' + ya + '_val').textContent = minVal.toFixed(2);
        document.getElementById('max_' + ya + '_val').textContent = maxVal.toFixed(2);
        var update = {};
        update[ya] = { range: [minVal, maxVal], domain: fixedDomains[ya] };
        Plotly.relayout('plotly-chart', update);
    }

    function resetYRanges() {
        featureConfig.forEach(function(feat, idx) {
            var ya = idx === 0 ? 'yaxis' : 'yaxis' + (idx + 1);
            applyRange(ya, feat.default_range);
        });
        var n = featureConfig.length;
        var sdYa = n === 0 ? 'yaxis' : 'yaxis' + (n + 1);
        applyRange(sdYa, [0, 1.05]);
    }

    function applyRange(ya, range) {
        var minInput = document.getElementById('min_' + ya);
        var maxInput = document.getElementById('max_' + ya);
        if (minInput && maxInput) {
            minInput.value = range[0].toFixed(2);
            maxInput.value = range[1].toFixed(2);
            document.getElementById('min_' + ya + '_val').textContent = range[0].toFixed(2);
            document.getElementById('max_' + ya + '_val').textContent = range[1].toFixed(2);
        }
        var update = {};
        update[ya] = { range: range, domain: fixedDomains[ya] };
        Plotly.relayout('plotly-chart', update);
    }
    </script>
</body>
</html>"""


def make_html_template(title, plotly_json_str, feat_json_str):
    html = HTML_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__PLOTLY_JSON__", plotly_json_str)
    html = html.replace("__FEATURES_JSON__", feat_json_str)
    return html


# ──────────────────────────────────────────────
# plot_validation
# ──────────────────────────────────────────────
def plot_validation(df_out, time_col, output_path, title="规划中心验证：实际值 vs 规划值"):
    if len(df_out) == 0:
        print("数据为空，跳过绘图")
        return

    fig, features, default_y_ranges = build_plotly_figure(df_out, time_col, [])

    # 构造 featureConfig JSON
    feat_list = []
    for name, col in features:
        if col in default_y_ranges:
            y_min, y_max = default_y_ranges[col]
            margin = (y_max - y_min) * 0.05
            feat_list.append({
                "name": name,
                "default_range": [y_min - margin, y_max + margin],
                "slider_min": float((y_min - margin) * 0.9),
                "slider_max": float((y_max + margin) * 1.1),
                "slider_step": float((y_max - y_min) / 100),
            })

    plotly_json = json.dumps(fig.to_plotly_json(), cls=NumpyEncoder, ensure_ascii=False)
    feat_json = json.dumps(feat_list, ensure_ascii=False)

    html = make_html_template(title, plotly_json, feat_json)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n可视化已保存: {output_path}")
    print(f"包含 {len(features)} 个特征子图 + S/D 子图 + 交互式纵轴范围滑块")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
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
