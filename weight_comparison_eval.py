# -*- coding: utf-8 -*-
"""
weight_comparison_eval.py — 三套权重参数对比评估脚本

评估三套权重在多个时间段上的表现：
1. 原始一套：[0, 0.4, 0.5, 0.4, 0.25, 0.30, 0.20, 0.20, 0.35]
2. 现在一套：defaults.yaml 当前配置
3. 新方案一套：吨煤产气量=0.1，其他与现在一套相同

评估指标：
  - 匹配度D：平均、最大、最小
  - 总加权loss
  - 每个输出维度的loss

用法：
    python plan_center/weight_comparison_eval.py
    python plan_center/weight_comparison_eval.py --config defaults.yaml
    python plan_center/weight_comparison_eval.py --months 2026-01 2026-03 2026-05
    python plan_center/weight_comparison_eval.py --output comparison_report.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')


# =========================
# 数据结构
# =========================

@dataclass
class EvalResult:
    """单套权重评估结果。"""
    name: str
    weights: dict[str, float]
    month_label: str = ""  # 时间段标签

    # 匹配度D统计
    d_mean: float = 0.0
    d_max: float = 0.0
    d_min: float = 0.0
    d_std: float = 0.0

    # 总loss
    total_loss: float = 0.0

    # 各维度loss
    per_dim_loss: dict[str, float] = field(default_factory=dict)  # {feature_name: loss}

    # 逐查询点的D和loss（用于后续分析）
    per_sample_d: np.ndarray = None
    per_sample_loss: np.ndarray = None
    per_dim_sample_loss: dict[str, np.ndarray] = field(default_factory=dict)


# =========================
# 权重配置定义
# =========================

# 9个特征顺序（对应原始一套的9个值）
WEIGHT_FEATURES_ORDER = [
    "主汽流量", "主汽压力", "炉膛差压", "一次风流量",
    "床温", "料层差压", "锅炉出口氧量", "二次风风量",
    "吨煤产气量"
]

# 现在一套（defaults.yaml当前配置）
CURRENT_WEIGHTS = {
    "主汽流量": 0.00,
    "吨煤产气量": 0.00,
    "主汽压力": 0.98,
    "炉膛差压": 0.50,
    "一次风流量": 0.49,
    "床温": 0.41,
    "料层差压": 0.57,
    "锅炉出口氧量": 0.42,
    "二次风风量": 0.30,
    "热值": 0.00,
}

# 原始一套：按特征顺序对应9个值
ORIGINAL_WEIGHTS_9 = [0, 0.4, 0.5, 0.4, 0.25, 0.30, 0.20, 0.20, 0.35]
ORIGINAL_WEIGHTS = {
    "主汽流量": 0.0,
    "主汽压力": 0.4,
    "炉膛差压": 0.5,
    "一次风流量": 0.4,
    "床温": 0.25,
    "料层差压": 0.30,
    "锅炉出口氧量": 0.20,
    "二次风风量": 0.20,
    "吨煤产气量": 0.35,
    "热值": 0.0,
}


def get_weight_configs():
    """返回三套权重配置。"""
    # 方案1：原始一套
    weights_v1 = dict(ORIGINAL_WEIGHTS)

    # 方案2：现在一套（defaults.yaml当前）
    weights_v2 = dict(CURRENT_WEIGHTS)

    # 方案3：吨煤产气量=0.1，其他与现在一套相同
    weights_v3 = dict(CURRENT_WEIGHTS)
    weights_v3["吨煤产气量"] = 0.1

    return [
        ("原始一套", weights_v1),
        ("现在一套", weights_v2),
        ("吨煤产气量0.1", weights_v3),
    ]


# =========================
# 评估器
# =========================

class StandardCache:
    """标准样本侧的预计算量（复用 optimize_weights.py 的实现）。"""

    def __init__(self, cfg, store):
        from plan_center.similarity import normalize_features

        feat = cfg.features
        residual_cols = [f"resid_{t}" for t in feat.residual_targets]
        self.sim_feature_cols = list(feat.raw_features) + residual_cols
        self.D = len(self.sim_feature_cols)

        V_norm_df = normalize_features(
            store.df_standard, self.sim_feature_cols, store.norm_stats, normalize_all=True
        )
        self.V_norm = V_norm_df[self.sim_feature_cols].values.astype(np.float64)
        self.eff_score = store.eff_score_all.astype(np.float64)
        self.loads_std = store.loads_standard.astype(np.float64)
        self.df_standard = store.df_standard
        self.plan_center_cols = list(feat.plan_center_cols)


class FastEvaluator:
    """
    向量化前向计算器（复用 optimize_weights.py 的实现）。
    支持额外返回 per-sample D 和 per-dimension loss。
    """

    def __init__(self, cfg, std_cache, store, query_batch_df, time_col):
        from plan_center.similarity import normalize_features

        feat = cfg.features
        match_cfg = cfg.matching
        gate_cfg = cfg.flow_gate

        # loss_feature_weights
        self.loss_feature_weights = {
            "主汽压力": 1.0, "炉膛差压": 1.0, "一次风流量": 1.0,
            "床温": 1.0, "料层差压": 1.0, "锅炉出口氧量": 0.6, "二次风风量": 0.6,
        }

        # 复用标准样本侧预计算
        self.sim_feature_cols = std_cache.sim_feature_cols
        self.D = std_cache.D
        self.V_norm = std_cache.V_norm
        self.eff_score = std_cache.eff_score
        self.loads_std = std_cache.loads_std
        self.df_standard = std_cache.df_standard
        self.plan_center_cols = std_cache.plan_center_cols

        # 8 个可调权重的特征名
        self.opt_feature_names = [
            c for c in feat.raw_features
            if c not in (feat.load_col, feat.heat_value_col)
        ]

        # 查询批次预计算
        residual_cols = [f"resid_{t}" for t in feat.residual_targets]
        sim_feature_cols = list(feat.raw_features) + residual_cols
        Q_norm_df = normalize_features(
            query_batch_df, sim_feature_cols, store.norm_stats, normalize_all=True
        )
        self.sim_feature_cols = sim_feature_cols
        self.Q_norm = Q_norm_df[sim_feature_cols].values.astype(np.float64)
        self.M = len(self.Q_norm)

        # loss 变量
        loss_cols = [c for c in self.loss_feature_weights if self.loss_feature_weights[c] > 0]
        self.loss_cols = loss_cols
        self.actual_vals = query_batch_df[loss_cols].values.astype(np.float64)
        self.loss_w = np.array(
            [self.loss_feature_weights[c] for c in loss_cols], dtype=np.float64
        )

        # IQR 尺度
        self.iqr_scales = np.array(
            [store.norm_stats[c]["iqr"] for c in loss_cols], dtype=np.float64
        )
        self.iqr_scales = np.maximum(self.iqr_scales, 1e-8)

        # 负荷
        self.Q_loads = query_batch_df[feat.load_col].values.astype(np.float64)

        # 硬门控掩码
        if gate_cfg.enable:
            diff = np.abs(self.Q_loads[:, None] - self.loads_std[None, :])
            if gate_cfg.mode == "absolute":
                self.gate_mask = diff <= gate_cfg.abs_threshold
            else:
                denom = np.maximum(np.abs(self.loads_std[None, :]), 1e-9)
                self.gate_mask = (diff / denom) <= gate_cfg.rel_threshold
        else:
            self.gate_mask = np.ones((self.M, len(self.loads_std)), dtype=bool)

        # 标准样本规划变量值矩阵
        self.V_plan = self.df_standard[loss_cols].values.astype(np.float64)

        self.top_k = match_cfg.top_k
        self.d_ws = float(match_cfg.d_weight_s)
        self.d_we = float(match_cfg.d_weight_e)

    def _build_weight_vector(self, opt_weights: np.ndarray) -> np.ndarray:
        """从8个可调权重重建15维权重向量。"""
        feat = self.df_standard  # dummy
        w_raw = {}
        for name, val in zip(self.opt_feature_names, opt_weights):
            w_raw[name] = max(float(val), 0.0)

        w = np.zeros(self.D, dtype=np.float64)
        for i, c in enumerate(self.sim_feature_cols):
            if c in w_raw:
                w[i] = w_raw[c]
            elif c.startswith("resid_"):
                target = c[len("resid_"):]
                raw_w = w_raw.get(target, 0.0)
                w[i] = raw_w * 0.5  # residual_weight_ratio = 0.5

        w_sum = w.sum()
        if w_sum < 1e-12:
            w[:] = 1.0 / self.D
        else:
            w /= w_sum
        return w

    def forward(self, opt_weights: np.ndarray) -> float:
        """
        前向计算，返回 total loss。

        同时保存 per-sample D 和 per-dimension loss 到 self 对象。
        """
        w = self._build_weight_vector(opt_weights)
        sqrt_w = np.sqrt(w)

        Q_xw = self.Q_norm * sqrt_w[None, :]
        V_xw = self.V_norm * sqrt_w[None, :]

        q_norm = np.linalg.norm(Q_xw, axis=1, keepdims=True)
        v_norm = np.linalg.norm(V_xw, axis=1, keepdims=True)
        q_norm = np.maximum(q_norm, 1e-12)
        v_norm = np.maximum(v_norm, 1e-12)

        Q_unit = Q_xw / q_norm
        V_unit = V_xw / v_norm

        cos_sim = Q_unit @ V_unit.T
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        S = (cos_sim + 1.0) / 2.0  # (M, N)

        # D = a*S + b*E
        D_score = self.d_ws * S + self.d_we * self.eff_score[None, :]

        # 硬门控
        D_gated = np.where(self.gate_mask, D_score, -np.inf)

        # Top-k by D
        top_k = min(self.top_k, D_gated.shape[1])
        top_k_idx = np.argpartition(D_gated, -top_k, axis=1)[:, -top_k:]  # (M, top_k)

        # 保存 per-sample best D（Top-1 的 D 值）
        D_top = np.take_along_axis(D_gated, top_k_idx, axis=1)
        self.per_sample_best_d = D_top[:, 0].copy()  # (M,) Top-1 D

        # 保存 Top-k mean D
        D_top_valid = np.where(D_top > -np.inf, D_top, np.nan)
        self.per_sample_topk_mean_d = np.nanmean(D_top_valid, axis=1)  # (M,)

        # D 加权均值规划中心
        D_top_clip = np.clip(D_top, 0.001, None)
        D_w = D_top_clip / D_top_clip.sum(axis=1, keepdims=True)

        V_plan_top = self.V_plan[top_k_idx]  # (M, top_k, K)
        plan_center = np.einsum("mt,mtk->mk", D_w, V_plan_top)  # (M, K)

        # 计算 per-dimension loss (IQR 归一化 MSE)
        err = (plan_center - self.actual_vals) / self.iqr_scales[None, :]  # (M, K)
        sq_err = err ** 2

        # 保存 per-sample per-dimension loss
        self.per_dim_sample_loss = {}
        for j, c in enumerate(self.loss_cols):
            self.per_dim_sample_loss[c] = sq_err[:, j].copy()  # (M,)

        # 保存 per-sample total loss (未乘以 loss_w 的均方误差)
        self.per_sample_loss = np.mean(sq_err, axis=1)  # (M,) 未加权 MSE

        # 总 loss（加权）
        loss = float(np.mean(sq_err * self.loss_w[None, :]))
        return loss


def make_opt_weights_vector(target_weights: dict, opt_feature_names: list) -> np.ndarray:
    """从权重字典构建优化器权重向量（8维）。"""
    return np.array(
        [target_weights.get(c, 0.0) for c in opt_feature_names],
        dtype=np.float64
    )


def evaluate_weights(
    name: str,
    target_weights: dict,
    std_cache: StandardCache,
    evaluator: FastEvaluator,
    query_df: pd.DataFrame,
    time_col: str,
    month_label: str = "",
) -> EvalResult:
    """
    评估单套权重配置。

    参数：
        name: 权重配置名称
        target_weights: {feature: weight} 权重字典
        std_cache: 标准样本缓存
        evaluator: FastEvaluator（固定 query batch）
        query_df: 查询数据 DataFrame
        time_col: 时间列名
        month_label: 时间段标签

    返回：
        EvalResult
    """
    opt_w = make_opt_weights_vector(target_weights, evaluator.opt_feature_names)

    # 前向计算
    total_loss = evaluator.forward(opt_w)

    # 提取统计量
    d_mean = float(np.nanmean(evaluator.per_sample_best_d))
    d_max = float(np.nanmax(evaluator.per_sample_best_d))
    d_min = float(np.nanmin(evaluator.per_sample_best_d))
    d_std = float(np.nanstd(evaluator.per_sample_best_d))

    # 各维度loss（加权MSE）
    per_dim_loss = {}
    for j, c in enumerate(evaluator.loss_cols):
        dim_sq_err = evaluator.per_dim_sample_loss[c]  # (M,)
        dim_loss = float(np.mean(dim_sq_err) * evaluator.loss_w[j])
        per_dim_loss[c] = dim_loss

    result = EvalResult(
        name=name,
        weights=dict(target_weights),
        month_label=month_label,
        d_mean=d_mean,
        d_max=d_max,
        d_min=d_min,
        d_std=d_std,
        total_loss=total_loss,
        per_dim_loss=per_dim_loss,
        per_sample_d=evaluator.per_sample_best_d.copy(),
        per_sample_loss=evaluator.per_sample_loss.copy(),
        per_dim_sample_loss={c: v.copy() for c, v in evaluator.per_dim_sample_loss.items()},
    )
    return result


def prepare_data(cfg, month_filter: str = None):
    """
    准备标准样本缓存和查询数据（复用 optimize_weights_v2 的逻辑）。

    参数：
        cfg: PlanningConfig
        month_filter: 月份筛选字符串，如 "2026-01"，None 表示最后一个月
    """
    from plan_center.standard_store import build_standard_store
    from plan_center.features import load_residual_models
    from plan_center.optimize_weights import add_residual_features_batch

    # 1. 加载标准样本和残差模型
    print("[1] 加载标准样本和残差模型...")
    store = build_standard_store(cfg)
    models = load_residual_models(cfg.paths.residual_model_dir, cfg.features.residual_targets)
    print(f"    标准样本数量: {len(store.df_standard)}")

    # 2. 读取查询数据
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

    # 3. 按月份筛选数据
    if month_filter:
        # 解析月份字符串 "2026-01" 或 "2026-1"
        parts = month_filter.split("-")
        year = int(parts[0])
        month = int(parts[1])
        # 筛选该月数据
        mask = (df_query[time_col].dt.year == year) & (df_query[time_col].dt.month == month)
        df_filtered = df_query[mask].reset_index(drop=True)
        print(f"\n[3] 筛选 {month_filter} 数据: {len(df_filtered)} 行 "
              f"({df_filtered[time_col].min().date()} ~ {df_filtered[time_col].max().date()})")
    else:
        # 默认筛选最后一个月
        max_time = df_query[time_col].max()
        start_time = max_time - pd.Timedelta(days=30)
        mask = df_query[time_col] >= start_time
        df_filtered = df_query[mask].reset_index(drop=True)
        print(f"\n[3] 筛选最后一个月: {len(df_filtered)} 行 "
              f"({df_filtered[time_col].min().date()} ~ {df_filtered[time_col].max().date()})")

    # 4. 计算残差特征
    print("\n[4] 计算残差特征...")
    t0 = time.perf_counter()
    df_filtered = add_residual_features_batch(df_filtered, models, cfg.features)
    print(f"    残差特征计算完成，耗时 {time.perf_counter() - t0:.1f}s")

    # 5. 丢弃 NaN 行
    residual_cols = [f"resid_{t}" for t in cfg.features.residual_targets]
    nan_check_cols = list(dict.fromkeys(
        list(cfg.features.raw_features) + residual_cols + [cfg.features.load_col]
    ))
    n_before = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=nan_check_cols).reset_index(drop=True)
    n_dropped = n_before - len(df_filtered)
    if n_dropped:
        print(f"    丢弃含 NaN 行: {n_dropped} 行（{n_dropped/n_before*100:.2f}%），剩余 {len(df_filtered)} 行")

    # 6. 构建 StandardCache
    print("\n[5] 构建 StandardCache...")
    std_cache = StandardCache(cfg, store)

    # 7. 构建 FastEvaluator
    print("\n[6] 构建 FastEvaluator...")
    evaluator = FastEvaluator(cfg, std_cache, store, df_filtered, time_col)
    print(f"    查询样本数: {evaluator.M}，损失变量数: {len(evaluator.loss_cols)}")

    return store, std_cache, evaluator, df_filtered, time_col


# =========================
# 报告生成
# =========================

def print_comparison_report(results: list[EvalResult], cfg):
    """打印对比报告（终端输出）。"""

    # 按时间段分组
    months = list(set(r.month_label for r in results))
    weight_names = list(set(r.name for r in results))

    for month in sorted(months):
        print(f"\n{'=' * 100}")
        print(f"时间段: {month}")
        print(f"{'=' * 100}")

        month_results = [r for r in results if r.month_label == month]

        for r in month_results:
            print(f"\n{'─' * 60}")
            print(f"【{r.name}】")
            print(f"  匹配度D: 均值={r.d_mean:.4f}  最大={r.d_max:.4f}  最小={r.d_min:.4f}  标准差={r.d_std:.4f}")
            print(f"  总加权Loss: {r.total_loss:.6f}")
            print(f"  各维度Loss:")
            for c, loss in r.per_dim_loss.items():
                print(f"    {c}: {loss:.6f}")

        # 对比表格
        print(f"\n{'─' * 100}")
        print(f"{'指标':<30} {'原始一套':>12} {'现在一套':>12} {'吨煤产气量0.1':>16}")
        print(f"{'─' * 100}")

        # 匹配度统计
        print(f"{'匹配度D_均值':<30} {month_results[0].d_mean:>12.4f} {month_results[1].d_mean:>12.4f} {month_results[2].d_mean:>16.4f}")
        print(f"{'匹配度D_最大':<30} {month_results[0].d_max:>12.4f} {month_results[1].d_max:>12.4f} {month_results[2].d_max:>16.4f}")
        print(f"{'匹配度D_最小':<30} {month_results[0].d_min:>12.4f} {month_results[1].d_min:>12.4f} {month_results[2].d_min:>16.4f}")
        print(f"{'匹配度D_标准差':<30} {month_results[0].d_std:>12.4f} {month_results[1].d_std:>12.4f} {month_results[2].d_std:>16.4f}")
        print(f"{'─' * 100}")

        # 总Loss
        print(f"{'总加权Loss':<30} {month_results[0].total_loss:>12.6f} {month_results[1].total_loss:>12.6f} {month_results[2].total_loss:>16.6f}")
        print(f"{'─' * 100}")

        # 各维度Loss
        loss_cols = month_results[0].per_dim_loss.keys()
        for c in loss_cols:
            v0 = month_results[0].per_dim_loss.get(c, float('nan'))
            v1 = month_results[1].per_dim_loss.get(c, float('nan'))
            v2 = month_results[2].per_dim_loss.get(c, float('nan'))
            print(f"  Loss_{c:<20} {v0:>12.6f} {v1:>12.6f} {v2:>16.6f}")

    # 汇总表（跨时间段）
    print(f"\n{'=' * 100}")
    print("跨时间段汇总")
    print(f"{'=' * 100}")
    print(f"{'时间段':<15} {'权重配置':<20} {'总Loss':>12} {'D均值':>10}")
    print(f"{'─' * 100}")
    for r in results:
        print(f"{r.month_label:<15} {r.name:<20} {r.total_loss:>12.6f} {r.d_mean:>10.4f}")


def save_report(results: list[EvalResult], output_path: str, cfg):
    """保存详细报告到 CSV。"""
    rows = []
    for r in results:
        row = {
            "时间段": r.month_label,
            "权重配置": r.name,
            "总加权Loss": r.total_loss,
            "匹配度D_均值": r.d_mean,
            "匹配度D_最大": r.d_max,
            "匹配度D_最小": r.d_min,
            "匹配度D_标准差": r.d_std,
        }
        for c, loss in r.per_dim_loss.items():
            row[f"Loss_{c}"] = loss
        for c, w in r.weights.items():
            row[f"权重_{c}"] = w
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n报告已保存: {output_path}")
    return df


# =========================
# 主流程
# =========================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三套权重配置对比评估")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--months", type=str, nargs="+", default=None,
                        help="要验证的月份列表，如 2026-01 2026-03 2026-05（默认最后一个月+随机两个月）")
    parser.add_argument("--output", type=str, default="weight_comparison_report.csv", help="输出报告路径")
    return parser.parse_args()


def run_evaluation(cfg, month_filter: str, weight_configs: list, std_cache, evaluator, df_query, time_col):
    """对指定月份运行评估。"""
    results = []
    for name, weights in weight_configs:
        t0 = time.perf_counter()
        result = evaluate_weights(name, weights, std_cache, evaluator, df_query, time_col, month_filter)
        elapsed = time.perf_counter() - t0
        print(f"    {month_filter} - {name} 完成，耗时 {elapsed:.2f}s，Loss={result.total_loss:.6f}")
        results.append(result)
    return results


def main():
    args = _parse_args()

    # 1. 加载配置
    print("=== 三套权重配置对比评估（多时间段）===\n")
    from plan_center.config import load_config
    cfg = load_config(args.config)
    print(f"配置文件: {args.config or 'defaults.yaml'}\n")

    # 2. 定义三套权重
    weight_configs = get_weight_configs()
    print("三套权重配置:")
    for name, weights in weight_configs:
        print(f"    {name}: {weights}\n")

    # 3. 确定要验证的月份
    if args.months:
        months_to_eval = args.months
    else:
        # 自动选择：最后一个月 + 随机两个月
        from plan_center.standard_store import build_standard_store
        store = build_standard_store(cfg)
        df_query = pd.read_parquet(cfg.paths.query_parquet)
        time_col = cfg.time_col or "时间"
        df_query[time_col] = pd.to_datetime(df_query[time_col], errors="coerce")

        # 获取所有月份
        all_months = sorted(df_query[time_col].dt.to_period("M").unique().astype(str).tolist())
        print(f"可用月份: {all_months}")

        if len(all_months) >= 3:
            # 选择最后一个月 + 中间两个月
            months_to_eval = [all_months[-1], all_months[len(all_months)//3], all_months[len(all_months)*2//3]]
        else:
            months_to_eval = all_months

    print(f"将验证以下月份: {months_to_eval}\n")

    # 4. 逐月评估
    all_results = []
    for month_filter in months_to_eval:
        print(f"\n{'=' * 60}")
        print(f"处理月份: {month_filter}")
        print(f"{'=' * 60}")

        # 准备该月数据
        store, std_cache, evaluator, df_filtered, time_col = prepare_data(cfg, month_filter)

        # 评估
        month_results = run_evaluation(cfg, month_filter, weight_configs, std_cache, evaluator, df_filtered, time_col)
        all_results.extend(month_results)

    # 5. 打印报告
    print_comparison_report(all_results, cfg)

    # 6. 保存报告
    output_path = args.output
    save_report(all_results, output_path, cfg)

    print("\n=== 评估完成 ===")
    return all_results


if __name__ == "__main__":
    main()
