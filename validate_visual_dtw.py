# -*- coding: utf-8 -*-
"""
validate_visual_dtw.py — DTW 验证脚本：连续数据可视化

与 validate_visual.py 功能对等，但使用 DTWQueryEngine 替代 PlanningEngine。
从分钟级查询数据中随机选取或按指定时间范围选取连续数据，
逐点调用 DTW 查询引擎，将实际值与规划值对比可视化。

包含 DTW 特有子图：
  - DTW 相似度 S（绿实线）
  - 时间偏移（天）（橙点线：候选末帧 vs 查询时刻）
  - 候选池规模（棕柱状）

用法：
    python -m plan_center.validate_visual_dtw
    python -m plan_center.validate_visual_dtw --days 3
    python -m plan_center.validate_visual_dtw --start 2025-03-01 --end 2025-03-02
    python -m plan_center.validate_visual_dtw --sample-step 5 --output result.html
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


# =========================
# Numpy JSON encoder
# =========================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =========================
# 数据选取
# =========================

def select_time_range(df, time_col, start_str, end_str):
    start_time = pd.Timestamp(start_str)
    end_time = pd.Timestamp(end_str)
    # 如果 end 没指定时刻（即 00:00:00），包含当天 → +1天
    if end_time.hour == 0 and end_time.minute == 0 and end_time.second == 0:
        end_time += pd.Timedelta(days=1)
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


# =========================
# DTW 特有诊断信息提取
# =========================

def _extract_dtw_diagnostics(plan_center: dict, actual: dict, result: Any) -> dict:
    """
    从 PlanResult 和实际值中提取 DTW 特有诊断字段。

    参数：
        plan_center: 规划中心字典
        actual: 实际值字典
        result: DTWQueryEngine.query_one() 返回的 PlanResult

    返回：
        dict: 包含 DTW 特有诊断字段
    """
    diag = {
        "dtw_similarity_best": result.similarity_best if hasattr(result, 'similarity_best') else np.nan,
        "dtw_d_topk_mean": result.score_d_topk_mean if hasattr(result, 'score_d_topk_mean') else np.nan,
        "dtw_path_cost": np.nan,
        "dtw_n_candidates": 0,
        "dtw_path_length": 0,
        "dtw_time_offset_days": np.nan,
    }

    # DTW 特有字段（通过 _query_dtw_detail 动态附加）
    if hasattr(result, '_dtw_path_cost'):
        diag["dtw_path_cost"] = result._dtw_path_cost
    if hasattr(result, '_n_candidates'):
        diag["dtw_n_candidates"] = result._n_candidates
    if hasattr(result, '_dtw_path_length'):
        diag["dtw_path_length"] = result._dtw_path_length
    if hasattr(result, '_time_offset_days'):
        diag["dtw_time_offset_days"] = result._time_offset_days

    return diag


# =========================
# 逐点 DTW 查询
# =========================

def run_validation_dtw(engine, df, time_col, sample_step=1, verbose=True, use_fast=False):
    """
    逐点调用 DTW 查询引擎，收集结果和诊断信息。

    参数：
        engine: DTWQueryEngine
        df: 查询数据 DataFrame
        time_col: 时间列名
        sample_step: 采样步长（分钟）
        verbose: 是否打印进度
        use_fast: 是否使用 Numba JIT 加速模式

    返回：
        (results, df_out) -> (List[PlanResult], pd.DataFrame)
    """
    from plan_center.schemas import build_output_dataframe, plan_result_to_row

    plan_center_cols = list(engine.cfg.features.plan_center_cols)

    # 采样
    indices = range(0, len(df), sample_step)
    total = len(list(indices))

    results = []
    t0 = time.perf_counter()

    # 选择查询方法
    query_method = "query_one_fast" if use_fast else "query_one"

    for i, row_idx in enumerate(indices):
        row = df.iloc[row_idx]
        query_ts = row[time_col]

        try:
            if use_fast:
                result = engine.query_one_fast(query_ts, verbose=False)
            else:
                result = engine.query_one(query_ts, verbose=False)
            results.append(result)
        except Exception as e:
            if verbose and i == 0:
                print(f"[警告] DTW 查询 {query_ts} 失败: {e}")
            # 空结果占位
            from plan_center.schemas import PlanResult
            empty = PlanResult(match_status=f"查询失败: {e}")
            results.append(empty)

        if verbose and (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t0
            avg = elapsed / (i + 1)
            eta = avg * (total - i - 1)
            print(f"  进度: {i+1}/{total}，均速 {avg:.1f}s/点，剩余 {eta:.0f}s")

    elapsed_total = time.perf_counter() - t0
    print(f"\nDTW 查询完成: {len(results)}/{total} 条，总耗时 {elapsed_total:.1f}s，均速 {elapsed_total/len(results):.1f}s/点")

    # 构建输出 DataFrame
    df_out = build_output_dataframe(
        raw_df=df.iloc[list(indices)].reset_index(drop=True),
        results=results,
        plan_center_cols=plan_center_cols,
    )

    return results, df_out


# =========================
# 构建 Plotly Figure（10 子图）
# =========================

def build_plotly_figure_dtw(df_out, time_col, plan_center_cols):
    """构造 10 行 subplot 的 Plotly Figure，包含 DTW 特有子图。"""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # 特征列表（从 plan_center_cols 中筛选有实际值的）
    feature_cols = [c for c in plan_center_cols if c in df_out.columns]

    # 特征显示名称映射
    feature_labels = {
        "主汽流量": "主汽流量（负荷）",
        "主汽压力": "主汽压力",
        "炉膛差压": "炉膛差压",
        "一次风流量": "一次风流量",
        "床温": "床温",
        "料层差压": "料层差压",
        "锅炉出口氧量": "锅炉出口氧量",
        "二次风风量": "二次风风量",
        "热值": "热值",
        "吨煤产气量": "吨煤产气量",
    }

    features = []
    for col in feature_cols:
        label = feature_labels.get(col, col)
        features.append((label, col))

    # 计算默认纵轴范围
    default_y_ranges = {}
    for name, col in features:
        actual = df_out[col].dropna()
        if len(actual) > 0:
            y_min, y_max = float(actual.min()), float(actual.max())
            if y_max > y_min:
                margin = (y_max - y_min) * 0.1
            else:
                margin = abs(y_max) * 0.1 if y_max != 0 else 0.1
            default_y_ranges[col] = [y_min - margin, y_max + margin]

    n_features = len(features)
    n_rows = n_features + 3  # 7 特征 + DTW相似度 + 时间偏移 + 候选规模

    subplot_titles = [f[0] for f in features]
    subplot_titles.append("DTW 相似度 S")
    subplot_titles.append("时间偏移（天）")
    subplot_titles.append("候选池规模")

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.03,
    )

    # 时间值
    time_values = pd.to_datetime(df_out[time_col]).values.astype("datetime64[ms]").astype(int).tolist()

    # ── 特征子图 ──
    for i, (name, col) in enumerate(features, start=1):
        actual = df_out[col].values.astype(float).tolist()
        planned_key = f"规划中心_{col}"
        planned = df_out[planned_key].values.astype(float).tolist() if planned_key in df_out.columns else [np.nan] * len(df_out)

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

    # ── DTW 相似度 S ──
    row_sim = n_features + 1
    sim_vals = df_out.get("相似度S", pd.Series(dtype=float)).values.astype(float).tolist()
    topk_sim_vals = df_out.get("TopK_S均值", pd.Series(dtype=float)).values.astype(float).tolist()

    fig.add_trace(
        go.Scatter(x=time_values, y=sim_vals, mode="lines",
                   name="相似度S（Best）", line=dict(color="green", width=1)),
        row=row_sim, col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_values, y=topk_sim_vals, mode="lines",
                   name="相似度S（TopK均值）", line=dict(color="lightgreen", width=1, dash="dot")),
        row=row_sim, col=1,
    )
    fig.update_yaxes(range=[0, 1.05], row=row_sim, col=1)

    # ── 时间偏移 ──
    row_offset = n_features + 2
    offset_vals = df_out.get("dtw_time_offset_days", pd.Series(dtype=float))
    if offset_vals.empty or offset_vals.isna().all():
        offset_vals = pd.Series([np.nan] * len(df_out), dtype=float)
    else:
        offset_vals = offset_vals.values.astype(float).tolist()

    fig.add_trace(
        go.Scatter(x=time_values, y=offset_vals, mode="lines",
                   name="候选末帧时间偏移", line=dict(color="orange", width=1)),
        row=row_offset, col=1,
    )
    fig.update_yaxes(title_text="天", row=row_offset, col=1)

    # ── 候选池规模 ──
    row_cand = n_features + 3
    cand_vals = df_out.get("dtw_n_candidates", pd.Series(dtype=int))
    if cand_vals.empty:
        cand_vals = pd.Series([0] * len(df_out), dtype=int)
    else:
        cand_vals = cand_vals.values.astype(int).tolist()

    fig.add_trace(
        go.Bar(x=time_values, y=cand_vals,
               name="候选序列数", marker=dict(color="brown", opacity=0.7)),
        row=row_cand, col=1,
    )
    fig.update_yaxes(title_text="数量", row=row_cand, col=1)

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


# =========================
# 生成 HTML
# =========================

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


def make_html_template_dtw(title, plotly_json_str, feat_json_str):
    html = HTML_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__PLOTLY_JSON__", plotly_json_str)
    html = html.replace("__FEATURES_JSON__", feat_json_str)
    return html


# =========================
# plot_validation_dtw
# =========================

def plot_validation_dtw(df_out, time_col, output_path, title="DTW 验证：实际值 vs 规划值"):
    if len(df_out) == 0:
        print("数据为空，跳过绘图")
        return

    # 提取 plan_center_cols
    plan_cols = [c.replace("规划中心_", "") for c in df_out.columns if c.startswith("规划中心_")]
    feat_cols = [c for c in plan_cols if c in df_out.columns]

    fig, features, default_y_ranges = build_plotly_figure_dtw(df_out, time_col, feat_cols)

    # 构造 featureConfig JSON
    feat_list = []
    for name, col in features:
        if col in default_y_ranges:
            y_min, y_max = default_y_ranges[col]
            margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
            feat_list.append({
                "name": name,
                "default_range": [y_min - margin, y_max + margin],
                "slider_min": float((y_min - margin) * 0.9),
                "slider_max": float((y_max + margin) * 1.1),
                "slider_step": float(max((y_max - y_min) / 100, 0.01)),
            })

    plotly_json = json.dumps(fig.to_plotly_json(), cls=NumpyEncoder, ensure_ascii=False)
    feat_json = json.dumps(feat_list, ensure_ascii=False)

    html = make_html_template_dtw(title, plotly_json, feat_json)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n可视化已保存: {output_path}")
    print(f"包含 {len(features)} 个特征子图 + DTW相似度S + 时间偏移 + 候选池规模 + 交互式纵轴范围滑块")


# =========================
# main
# =========================

def main():
    import time as time_mod

    parser = argparse.ArgumentParser(description="DTW 验证：实际值 vs 规划值可视化")
    parser.add_argument("--start", type=str, default=None,
                        help="起始时间（如 '2025-03-01'），与 --end 配合使用")
    parser.add_argument("--end", type=str, default=None,
                        help="终止时间（如 '2025-03-02'），与 --start 配合使用")
    parser.add_argument("--days", type=int, default=1,
                        help="随机模式下选取的天数（默认 1）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（仅随机模式有效）")
    parser.add_argument("--sample-step", type=int, default=1,
                        help="采样步长（分钟），默认 1（每分钟查询）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 HTML 路径（默认按时间范围自动命名）")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（默认 defaults.yaml）")
    parser.add_argument("--fast", action="store_true",
                        help="使用 Numba JIT 加速模式（需预热）")
    args = parser.parse_args()

    use_specified_range = bool(args.start and args.end)

    print("=== DTW 验证脚本：连续数据可视化 ===\n")
    print(f"模式: {'Fast (JIT)' if args.fast else 'Baseline (Python)'}\n")

    # 1. 加载配置 + 初始化 DTW 引擎
    print("[1] 加载 DTWQueryEngine...")
    from plan_center.config import load_config
    from plan_center.dtw_query import DTWQueryEngine

    cfg = load_config(args.config)
    engine = DTWQueryEngine(cfg)
    print(f"    DTW 配置: ref_days={cfg.dtw_query.ref_days}, query_seq_len={cfg.dtw_query.query_seq_len}, "
          f"dtw_min_len={cfg.dtw_query.dtw_min_len}, top_k={cfg.dtw_query.top_k}")

    # 2. 读取分钟级查询 parquet
    print("\n[2] 读取分钟级查询数据...")
    query_parquet = cfg.paths.query_parquet
    if not query_parquet or not Path(query_parquet).exists():
        raise FileNotFoundError(f"查询数据路径不存在: {query_parquet}")

    df_query = pd.read_parquet(query_parquet)

    # 应用列别名映射
    aliases = cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in df_query.columns and new not in df_query.columns:
                df_query[new] = df_query[old]
            elif old in df_query.columns and new in df_query.columns:
                df_query = df_query.drop(columns=[old])

    time_col = cfg.time_col or "时间"
    df_query[time_col] = pd.to_datetime(df_query[time_col], errors="coerce")
    df_query = df_query.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    print(f"    查询数据: {df_query.shape}，时间范围 {df_query[time_col].min()} ~ {df_query[time_col].max()}")

    # 3. 选取数据（扩展开头以确保 DTW 有足够参考窗口）
    dtw_cfg = cfg.dtw_query
    if use_specified_range:
        print(f"\n[3] 按指定时间范围选取: {args.start} ~ {args.end}")
        # 为了让可视化范围起点也能正确匹配，需往前扩展 ref_days 天
        t_user_start = pd.Timestamp(args.start)
        t_actual_start = t_user_start - pd.Timedelta(days=dtw_cfg.ref_days)
        # 从扩展起点选取到用户指定的结束时间（含边界）
        t_end = pd.Timestamp(args.end)
        if t_end.hour == 0 and t_end.minute == 0 and t_end.second == 0:
            t_end += pd.Timedelta(days=1)
        mask = (df_query[time_col] >= t_actual_start) & (df_query[time_col] <= t_end)
        df_selected = df_query[mask].sort_values(time_col).reset_index(drop=True)
        print(f"    DTW 参考窗口扩展: {t_actual_start} ~ {t_end}")
        print(f"    实际选取: {df_selected[time_col].min()} ~ {df_selected[time_col].max()} ({len(df_selected)} 行)")
        print(f"    可视化范围: {args.start} ~ {args.end}")
        start_label = pd.Timestamp(args.start).strftime("%Y-%m-%d")
        end_label = pd.Timestamp(args.end).strftime("%Y-%m-%d")
        range_str = f"{start_label}_to_{end_label}"
        title = f"DTW 验证：实际值 vs 规划值（{args.start} ~ {args.end}）"
    else:
        print(f"\n[3] 随机选取连续 {args.days} 天...")
        if args.seed is not None:
            np.random.seed(args.seed)
            print(f"    随机种子: {args.seed}")
        df_selected = select_random_days(df_query, time_col, days=args.days)
        # 随机模式也需往前扩展 ref_days 天
        t_min = df_selected[time_col].min()
        t_actual_start = t_min - pd.Timedelta(days=dtw_cfg.ref_days)
        if t_actual_start >= df_query[time_col].min():
            mask = (df_query[time_col] >= t_actual_start) & (df_query[time_col] <= df_selected[time_col].max())
            df_selected = df_query[mask].sort_values(time_col).reset_index(drop=True)
            print(f"    DTW 参考窗口扩展: {t_actual_start} ~ {df_selected[time_col].max()}")
            print(f"    实际选取: {df_selected[time_col].min()} ~ {df_selected[time_col].max()} ({len(df_selected)} 行)")
        t_start = df_selected[time_col].min()
        t_end = df_selected[time_col].max()
        range_str = f"{t_start.strftime('%Y-%m-%d')}_{args.days}days"
        title = f"DTW 验证：实际值 vs 规划值（随机{args.days}天，{t_start.strftime('%Y-%m-%d %H:%M')} ~ {t_end.strftime('%Y-%m-%d %H:%M')}）"

    # 4. 预热（如果使用 Fast 模式）
    if args.fast:
        print("\n[3.5] JIT 预热中...")
        engine.warmup(n_iter=3)
        print("[3.5] 预热完成\n")

    # 4. 逐点 DTW 查询
    print(f"\n[4] 逐点 DTW 查询（sample_step={args.sample_step}）...")
    results, df_out = run_validation_dtw(engine, df_selected, time_col,
                                          sample_step=args.sample_step, use_fast=args.fast)

    if len(df_out) == 0:
        print("\n警告: 选取的数据为空，跳过可视化")
        return

    # 5. 可视化
    print("\n[5] 生成可视化...")
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / f"validate_dtw_{range_str}.html"

    plot_validation_dtw(df_out, time_col, str(output_path), title=title)

    print("\n=== DTW 验证完成 ===")
    print(f"时间范围: {df_selected[time_col].min()} ~ {df_selected[time_col].max()}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
