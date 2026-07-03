# -*- coding: utf-8 -*-
"""
weight_comparison_eval.py — 三套权重参数对比评估脚本

评估三套权重在多个时间段上的表现：
1. 原始一套：[0, 0.4, 0.5, 0.4, 0.25, 0.30, 0.20, 0.20, 0.35]
2. 现在一套：defaults.yaml 当前配置
3. 新方案一套：吨煤产气量=0.1，其他与现在一套相同

评估两种归一化方式：
- 全局归一化：从整个标准样本库计算统计量
- 动态归一化：每个查询点从候选子集计算统计量

评估指标：
  - 匹配度D：平均、最大、最小
  - 总加权loss
  - 每个输出维度的loss

用法：
    python plan_center/weight_comparison_eval.py
    python plan_center/weight_comparison_eval.py --start 2026-05-11 --end 2026-05-21
    python plan_center/weight_comparison_eval.py --months 2026-01 2026-03 2026-05
    python plan_center/weight_comparison_eval.py --output comparison_report.csv
    python plan_comparison_eval.py --norm-compare  # 对比两种归一化方式
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')


# =========================
# 归一化方式枚举
# =========================

class NormMethod(Enum):
    """归一化方式"""
    GLOBAL = "全局归一化"      # 从整个标准样本库计算统计量
    DYNAMIC = "动态归一化"    # 每个查询点从候选子集计算统计量


# =========================
# 数据结构
# =========================

@dataclass
class EvalResult:
    """单套权重评估结果。"""
    name: str
    weights: dict[str, float]
    month_label: str = ""      # 时间段标签
    norm_method: str = ""      # 归一化方式

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
    weights_v1 = dict(ORIGINAL_WEIGHTS)
    weights_v2 = dict(CURRENT_WEIGHTS)
    weights_v3 = dict(CURRENT_WEIGHTS)
    weights_v3["吨煤产气量"] = 0.1

    return [
        ("原始一套", weights_v1),
        ("现在一套", weights_v2),
        ("吨煤产气量0.1", weights_v3),
    ]


# =========================
# 工具函数
# =========================

def compute_norm_stats_from_df(df: pd.DataFrame, feature_cols: list) -> dict:
    """从 DataFrame 计算 median 和 IQR 归一化统计量。"""
    from plan_center.similarity import compute_norm_stats_from_df as _compute
    return _compute(df, feature_cols)


def apply_global_normalization(df: pd.DataFrame, feature_cols: list, norm_stats: dict) -> pd.DataFrame:
    """使用全局统计量进行归一化。"""
    from plan_center.similarity import normalize_features
    return normalize_features(df, feature_cols, norm_stats, normalize_all=True)


# =========================
# 标准样本缓存
# =========================

class StandardCache:
    """标准样本侧的预计算量。计算全局归一化统计量用于评估。"""

    def __init__(self, cfg, store):
        feat = cfg.features
        residual_cols = [f"resid_{t}" for t in feat.residual_targets]
        self.sim_feature_cols = list(feat.raw_features) + residual_cols
        self.D = len(self.sim_feature_cols)

        self.eff_score = store.eff_score_all.astype(np.float64)
        self.loads_std = store.loads_standard.astype(np.float64)
        self.df_standard = store.df_standard
        self.plan_center_cols = list(feat.plan_center_cols)
        self.residual_targets = list(feat.residual_targets)

        # 计算全局归一化统计量
        self.norm_stats = compute_norm_stats_from_df(store.df_standard, self.sim_feature_cols)

        # 未归一化的原始特征矩阵（用于动态归一化）
        self.raw_feature_cols = list(feat.raw_features) + residual_cols
        self.V_raw = store.df_standard[self.raw_feature_cols].values.astype(np.float64)


# =========================
# 全局归一化评估器（向量化，Fast）
# =========================

class FastEvaluator:
    """
    向量化前向计算器（使用全局归一化）。
    """

    def __init__(self, cfg, std_cache, store, query_batch_df, time_col):
        feat = cfg.features
        match_cfg = cfg.matching
        gate_cfg = cfg.flow_gate

        self.loss_feature_weights = {
            "主汽压力": 1.0, "炉膛差压": 1.0, "一次风流量": 1.0,
            "床温": 1.0, "料层差压": 1.0, "锅炉出口氧量": 0.6, "二次风风量": 0.6,
        }

        self.sim_feature_cols = std_cache.sim_feature_cols
        self.D = std_cache.D
        self.eff_score = std_cache.eff_score
        self.loads_std = std_cache.loads_std
        self.df_standard = std_cache.df_standard
        self.plan_center_cols = std_cache.plan_center_cols
        self.norm_stats = std_cache.norm_stats

        self.opt_feature_names = [
            c for c in feat.raw_features
            if c not in (feat.load_col, feat.heat_value_col)
        ]

        # 使用原始特征列（包含 resid_*）
        residual_cols = [f"resid_{t}" for t in feat.residual_targets]
        sim_feature_cols = list(feat.raw_features) + residual_cols
        self.sim_feature_cols = sim_feature_cols

        # 查询批次归一化
        Q_norm_df = apply_global_normalization(query_batch_df, sim_feature_cols, self.norm_stats)
        self.Q_norm = Q_norm_df[sim_feature_cols].values.astype(np.float64)
        self.M = len(self.Q_norm)

        # 标准样本归一化
        V_norm_df = apply_global_normalization(store.df_standard, sim_feature_cols, self.norm_stats)
        self.V_norm = V_norm_df[sim_feature_cols].values.astype(np.float64)

        # loss 变量
        loss_cols = [c for c in self.loss_feature_weights if self.loss_feature_weights[c] > 0]
        self.loss_cols = loss_cols
        self.actual_vals = query_batch_df[loss_cols].values.astype(np.float64)
        self.loss_w = np.array([self.loss_feature_weights[c] for c in loss_cols], dtype=np.float64)
        self.iqr_scales = np.array([self.norm_stats[c]["iqr"] for c in loss_cols], dtype=np.float64)
        self.iqr_scales = np.maximum(self.iqr_scales, 1e-8)

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

        self.V_plan = self.df_standard[loss_cols].values.astype(np.float64)
        self.top_k = match_cfg.top_k
        self.d_ws = float(match_cfg.d_weight_s)
        self.d_we = float(match_cfg.d_weight_e)

    def _build_weight_vector(self, opt_weights: np.ndarray) -> np.ndarray:
        """从8个可调权重重建15维权重向量。"""
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
                w[i] = raw_w * 0.5

        w_sum = w.sum()
        if w_sum < 1e-12:
            w[:] = 1.0 / self.D
        else:
            w /= w_sum
        return w

    def forward(self, opt_weights: np.ndarray) -> float:
        """前向计算（全局归一化，向量化）。"""
        w = self._build_weight_vector(opt_weights)
        sqrt_w = np.sqrt(w)

        Q_xw = self.Q_norm * sqrt_w[None, :]
        V_xw = self.V_norm * sqrt_w[None, :]

        q_norm = np.maximum(np.linalg.norm(Q_xw, axis=1, keepdims=True), 1e-12)
        v_norm = np.maximum(np.linalg.norm(V_xw, axis=1, keepdims=True), 1e-12)

        Q_unit = Q_xw / q_norm
        V_unit = V_xw / v_norm

        cos_sim = np.clip(Q_unit @ V_unit.T, -1.0, 1.0)
        S = (cos_sim + 1.0) / 2.0

        D_score = self.d_ws * S + self.d_we * self.eff_score[None, :]
        D_gated = np.where(self.gate_mask, D_score, -np.inf)

        top_k = min(self.top_k, D_gated.shape[1])
        top_k_idx = np.argpartition(D_gated, -top_k, axis=1)[:, -top_k:]

        D_top = np.take_along_axis(D_gated, top_k_idx, axis=1)
        self.per_sample_best_d = D_top[:, 0].copy()
        D_top_valid = np.where(D_top > -np.inf, D_top, np.nan)
        self.per_sample_topk_mean_d = np.nanmean(D_top_valid, axis=1)

        D_top_clip = np.clip(D_top, 0.001, None)
        D_w = D_top_clip / D_top_clip.sum(axis=1, keepdims=True)
        V_plan_top = self.V_plan[top_k_idx]
        plan_center = np.einsum("mt,mtk->mk", D_w, V_plan_top)

        err = (plan_center - self.actual_vals) / self.iqr_scales[None, :]
        sq_err = err ** 2

        self.per_dim_sample_loss = {}
        for j, c in enumerate(self.loss_cols):
            self.per_dim_sample_loss[c] = sq_err[:, j].copy()

        self.per_sample_loss = np.mean(sq_err, axis=1)
        loss = float(np.mean(sq_err * self.loss_w[None, :]))
        return loss


# =========================
# 动态归一化评估器（逐点计算，精确）
# =========================

class DynamicNormEvaluator:
    """
    动态归一化前向计算器。
    每个查询点从候选子集计算统计量，然后归一化并计算相似度。
    """

    def __init__(self, cfg, std_cache, store, query_batch_df, time_col):
        feat = cfg.features
        match_cfg = cfg.matching
        gate_cfg = cfg.flow_gate

        self.loss_feature_weights = {
            "主汽压力": 1.0, "炉膛差压": 1.0, "一次风流量": 1.0,
            "床温": 1.0, "料层差压": 1.0, "锅炉出口氧量": 0.6, "二次风风量": 0.6,
        }

        self.sim_feature_cols = std_cache.sim_feature_cols
        self.D = std_cache.D
        self.eff_score = std_cache.eff_score
        self.loads_std = std_cache.loads_std
        self.df_standard = std_cache.df_standard
        self.plan_center_cols = std_cache.plan_center_cols
        self.norm_stats = std_cache.norm_stats
        self.V_raw = std_cache.V_raw

        self.opt_feature_names = [
            c for c in feat.raw_features
            if c not in (feat.load_col, feat.heat_value_col)
        ]

        residual_cols = [f"resid_{t}" for t in feat.residual_targets]
        sim_feature_cols = list(feat.raw_features) + residual_cols
        self.sim_feature_cols = sim_feature_cols

        # 查询批次原始特征（未归一化）
        self.Q_raw = query_batch_df[sim_feature_cols].values.astype(np.float64)
        self.M = len(self.Q_raw)

        loss_cols = [c for c in self.loss_feature_weights if self.loss_feature_weights[c] > 0]
        self.loss_cols = loss_cols
        self.actual_vals = query_batch_df[loss_cols].values.astype(np.float64)
        self.loss_w = np.array([self.loss_feature_weights[c] for c in loss_cols], dtype=np.float64)

        # 注意：iqr_scales 使用全局统计量（用于 loss 计算的归一化）
        self.iqr_scales = np.array([self.norm_stats[c]["iqr"] for c in loss_cols], dtype=np.float64)
        self.iqr_scales = np.maximum(self.iqr_scales, 1e-8)

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

        self.V_plan = self.df_standard[loss_cols].values.astype(np.float64)
        self.top_k = match_cfg.top_k
        self.d_ws = float(match_cfg.d_weight_s)
        self.d_we = float(match_cfg.d_weight_e)
        self.N = len(self.loads_std)

        # 预计算效率得分
        self.eff_score_expanded = self.eff_score[None, :]  # (1, N)

    def _build_weight_vector(self, opt_weights: np.ndarray) -> np.ndarray:
        """从8个可调权重重建15维权重向量。"""
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
                w[i] = raw_w * 0.5

        w_sum = w.sum()
        if w_sum < 1e-12:
            w[:] = 1.0 / self.D
        else:
            w /= w_sum
        return w

    def forward(self, opt_weights: np.ndarray) -> float:
        """
        前向计算（动态归一化，逐点计算）。
        """
        w = self._build_weight_vector(opt_weights)
        sqrt_w = np.sqrt(w)

        per_sample_d = np.zeros(self.M)
        per_sample_loss = np.zeros(self.M)
        per_dim_sample_loss = {c: np.zeros(self.M) for c in self.loss_cols}

        for i in range(self.M):
            q_raw = self.Q_raw[i]
            q_load = self.Q_loads[i]

            # 1. Flow gate 筛选候选子集
            mask = self.gate_mask[i]
            candidate_indices = np.where(mask)[0]

            if len(candidate_indices) < 5:
                # 候选太少，使用全局统计量
                norm_stats = self.norm_stats
            else:
                # 2. 从候选子集计算动态统计量
                candidate_raw = self.V_raw[candidate_indices]
                norm_stats = {}
                for j, c in enumerate(self.sim_feature_cols):
                    col_data = candidate_raw[:, j]
                    median_val = float(np.median(col_data))
                    q75, q25 = np.percentile(col_data, [75, 25])
                    iqr_val = float(q75 - q25)
                    if iqr_val < 1e-8:
                        iqr_val = 1e-8
                    norm_stats[c] = {"median": median_val, "iqr": iqr_val}

            # 3. 归一化查询点
            q_norm = np.zeros(self.D)
            for j, c in enumerate(self.sim_feature_cols):
                if c in norm_stats:
                    q_norm[j] = (q_raw[j] - norm_stats[c]["median"]) / norm_stats[c]["iqr"]

            # 归一化候选子集
            V_cand_raw = self.V_raw[candidate_indices]
            V_cand_norm = np.zeros((len(candidate_indices), self.D))
            for j, c in enumerate(self.sim_feature_cols):
                if c in norm_stats:
                    V_cand_norm[:, j] = (V_cand_raw[:, j] - norm_stats[c]["median"]) / norm_stats[c]["iqr"]

            # 4. 加权余弦相似度
            q_xw = q_norm * sqrt_w
            v_xw = V_cand_norm * sqrt_w

            q_norm_w = np.linalg.norm(q_xw)
            v_norm_w = np.linalg.norm(v_xw, axis=1)

            if q_norm_w < 1e-12 or np.any(v_norm_w < 1e-12):
                S_cand = np.zeros(len(candidate_indices))
            else:
                q_unit = q_xw / q_norm_w
                v_unit = v_xw / v_norm_w[:, None]
                cos_sim = np.clip(q_unit @ v_unit.T, -1.0, 1.0)
                S_cand = (cos_sim + 1.0) / 2.0

            # 5. 计算 D 分数
            D_cand = self.d_ws * S_cand + self.d_we * self.eff_score[candidate_indices]

            # 6. Top-k
            top_k = min(self.top_k, len(candidate_indices))
            if top_k > 0:
                top_k_idx = np.argpartition(D_cand, -top_k)[-top_k:]
                D_top = D_cand[top_k_idx]
                S_top = S_cand[top_k_idx]

                per_sample_d[i] = D_top[0]  # Top-1 D

                # 加权均值规划中心
                D_clip = np.clip(D_top, 0.001, None)
                D_w = D_clip / D_clip.sum()
                V_plan_top = self.V_plan[candidate_indices][top_k_idx]
                plan_center = np.dot(D_w, V_plan_top)

                # 计算 loss
                err = (plan_center - self.actual_vals[i]) / self.iqr_scales
                sq_err = err ** 2

                per_sample_loss[i] = float(np.mean(sq_err))
                for j, c in enumerate(self.loss_cols):
                    per_dim_sample_loss[c][i] = sq_err[j]

        # 保存结果
        self.per_sample_best_d = per_sample_d
        self.per_sample_loss = per_sample_loss
        self.per_dim_sample_loss = per_dim_sample_loss

        loss = float(np.mean(per_sample_loss * self.loss_w.mean()))
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
    evaluator,
    query_df: pd.DataFrame,
    time_col: str,
    month_label: str = "",
    norm_method: str = "",
) -> EvalResult:
    """评估单套权重配置。"""
    opt_w = make_opt_weights_vector(target_weights, evaluator.opt_feature_names)

    t0 = time.perf_counter()
    total_loss = evaluator.forward(opt_w)
    elapsed = time.perf_counter() - t0

    d_mean = float(np.nanmean(evaluator.per_sample_best_d))
    d_max = float(np.nanmax(evaluator.per_sample_best_d))
    d_min = float(np.nanmin(evaluator.per_sample_best_d))
    d_std = float(np.nanstd(evaluator.per_sample_best_d))

    per_dim_loss = {}
    for j, c in enumerate(evaluator.loss_cols):
        dim_sq_err = evaluator.per_dim_sample_loss[c]
        dim_loss = float(np.mean(dim_sq_err) * evaluator.loss_w[j])
        per_dim_loss[c] = dim_loss

    result = EvalResult(
        name=name,
        weights=dict(target_weights),
        month_label=month_label,
        norm_method=norm_method,
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

    print(f"      {name}({norm_method}) 完成，耗时 {elapsed:.1f}s，Loss={total_loss:.6f}")
    return result


def prepare_data(cfg, start_date: str = None, end_date: str = None, month_filter: str = None):
    """准备标准样本缓存和查询数据。"""
    from plan_center.standard_store import build_standard_store
    from plan_center.features import load_residual_models
    from plan_center.optimize_weights import add_residual_features_batch

    print("[1] 加载标准样本和残差模型...")
    store = build_standard_store(cfg)
    models = load_residual_models(cfg.paths.residual_model_dir, cfg.features.residual_targets)
    print(f"    标准样本数量: {len(store.df_standard)}")

    print("\n[2] 读取查询数据...")
    query_parquet = cfg.paths.query_parquet
    if not query_parquet or not Path(query_parquet).exists():
        raise FileNotFoundError(f"查询数据不存在: {query_parquet}")

    df_query = pd.read_parquet(query_parquet)

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
    if start_date and end_date:
        start_time = pd.Timestamp(start_date)
        end_time = pd.Timestamp(end_date)
        mask = (df_query[time_col] >= start_time) & (df_query[time_col] < end_time)
        df_filtered = df_query[mask].reset_index(drop=True)
        range_label = f"{start_date}_to_{end_date}"
        print(f"\n[3] 筛选 {start_date} ~ {end_date} 数据: {len(df_filtered)} 行")
    elif month_filter:
        parts = month_filter.split("-")
        year = int(parts[0])
        month = int(parts[1])
        mask = (df_query[time_col].dt.year == year) & (df_query[time_col].dt.month == month)
        df_filtered = df_query[mask].reset_index(drop=True)
        range_label = month_filter
        print(f"\n[3] 筛选 {month_filter} 数据: {len(df_filtered)} 行")
    else:
        max_time = df_query[time_col].max()
        start_time = max_time - pd.Timedelta(days=30)
        mask = df_query[time_col] >= start_time
        df_filtered = df_query[mask].reset_index(drop=True)
        range_label = "last_30days"
        print(f"\n[3] 筛选最后一个月: {len(df_filtered)} 行")

    # 计算残差特征
    print("\n[4] 计算残差特征...")
    t0 = time.perf_counter()
    df_filtered = add_residual_features_batch(df_filtered, models, cfg.features)
    print(f"    残差特征计算完成，耗时 {time.perf_counter() - t0:.1f}s")

    # 丢弃 NaN 行
    residual_cols = [f"resid_{t}" for t in cfg.features.residual_targets]
    nan_check_cols = list(dict.fromkeys(
        list(cfg.features.raw_features) + residual_cols + [cfg.features.load_col]
    ))
    n_before = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=nan_check_cols).reset_index(drop=True)
    n_dropped = n_before - len(df_filtered)
    if n_dropped:
        print(f"    丢弃含 NaN 行: {n_dropped} 行（{n_dropped/n_before*100:.2f}%），剩余 {len(df_filtered)} 行")

    # 构建缓存
    print("\n[5] 构建 StandardCache...")
    std_cache = StandardCache(cfg, store)

    print(f"    查询样本数: {len(df_filtered)}")

    return store, std_cache, df_filtered, time_col, range_label


# =========================
# 报告生成
# =========================

def print_comparison_report(results: list[EvalResult]):
    """打印对比报告。"""
    months = sorted(set(r.month_label for r in results))

    for month in months:
        month_results = [r for r in results if r.month_label == month]
        norm_methods = sorted(set(r.norm_method for r in month_results))
        weight_names = sorted(set(r.name for r in month_results))

        print(f"\n{'=' * 120}")
        print(f"时间段: {month}")
        print(f"{'=' * 120}")

        # 按权重方案分组显示
        for wname in weight_names:
            w_results = [r for r in month_results if r.name == wname]
            if not w_results:
                continue

            print(f"\n{'─' * 80}")
            print(f"【{wname}】")

            for r in w_results:
                print(f"\n  [{r.norm_method}]")
                print(f"    匹配度D: 均值={r.d_mean:.4f}  最大={r.d_max:.4f}  最小={r.d_min:.4f}  标准差={r.d_std:.4f}")
                print(f"    总加权Loss: {r.total_loss:.6f}")
                print(f"    各维度Loss:")
                for c, loss in r.per_dim_loss.items():
                    print(f"      {c}: {loss:.6f}")

        # 归一化方式对比表
        if len(norm_methods) == 2 and len(weight_names) >= 1:
            print(f"\n{'─' * 120}")
            print(f"{'指标':<35} " + "".join(f"{'[' + m + ']':>25}" for m in norm_methods))
            print(f"{'─' * 120}")

            # 按权重方案分别显示对比
            for wname in weight_names:
                w_results = {r.norm_method: r for r in month_results if r.name == wname}
                if len(w_results) != 2:
                    continue

                global_r = w_results.get(NormMethod.GLOBAL.value) or w_results.get("全局归一化")
                dynamic_r = w_results.get(NormMethod.DYNAMIC.value) or w_results.get("动态归一化")

                if global_r and dynamic_r:
                    print(f"\n{wname}:")
                    print(f"  {'匹配度D_均值':<35} {global_r.d_mean:>25.4f} {dynamic_r.d_mean:>25.4f}")
                    print(f"  {'匹配度D_最大':<35} {global_r.d_max:>25.4f} {dynamic_r.d_max:>25.4f}")
                    print(f"  {'匹配度D_最小':<35} {global_r.d_min:>25.4f} {dynamic_r.d_min:>25.4f}")
                    print(f"  {'总加权Loss':<35} {global_r.total_loss:>25.6f} {dynamic_r.total_loss:>25.6f}")
                    loss_diff = (dynamic_r.total_loss - global_r.total_loss) / global_r.total_loss * 100
                    print(f"  {'Loss差异(%)':<35} {'-':>25} {loss_diff:>+25.2f}%")

    # 汇总表
    print(f"\n{'=' * 120}")
    print("跨时间段汇总")
    print(f"{'=' * 120}")
    print(f"{'时间段':<25} {'权重配置':<18} {'归一化方式':<12} {'总Loss':>12} {'D均值':>10} {'vs全局差%':>10}")
    print(f"{'─' * 120}")

    # 计算全局基准
    global_results = {r.month_label: {r.name: r.total_loss for r in results if r.norm_method == "全局归一化"} for r in results}

    for r in sorted(results, key=lambda x: (x.month_label, x.name, x.norm_method)):
        base_loss = global_results.get(r.month_label, {}).get(r.name, r.total_loss)
        diff_pct = (r.total_loss - base_loss) / base_loss * 100 if base_loss > 0 else 0
        diff_str = f"{diff_pct:+.2f}%" if r.norm_method == "动态归一化" else "-"
        print(f"{r.month_label:<25} {r.name:<18} {r.norm_method:<12} {r.total_loss:>12.6f} {r.d_mean:>10.4f} {diff_str:>10}")


def save_report(results: list[EvalResult], output_path: str):
    """保存详细报告到 CSV。"""
    rows = []
    for r in results:
        row = {
            "时间段": r.month_label,
            "权重配置": r.name,
            "归一化方式": r.norm_method,
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
    parser = argparse.ArgumentParser(description="三套权重配置对比评估（支持归一化方式对比）")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--start", type=str, default=None, help="起始日期（如 2026-05-11）")
    parser.add_argument("--end", type=str, default=None, help="终止日期（如 2026-05-21）")
    parser.add_argument("--months", type=str, nargs="+", default=None,
                        help="要验证的月份列表，如 2026-01 2026-03 2026-05")
    parser.add_argument("--output", type=str, default="weight_comparison_report.csv", help="输出报告路径")
    parser.add_argument("--norm-compare", action="store_true", help="对比两种归一化方式")
    return parser.parse_args()


def main():
    args = _parse_args()

    print("=== 三套权重配置对比评估（多时间段 + 归一化方式对比）===\n")
    from plan_center.config import load_config
    cfg = load_config(args.config)
    print(f"配置文件: {args.config or 'defaults.yaml'}\n")

    weight_configs = get_weight_configs()
    print("三套权重配置:")
    for name, weights in weight_configs:
        print(f"    {name}: {weights}\n")

    # 准备数据
    if args.start and args.end:
        store, std_cache, df_filtered, time_col, range_label = prepare_data(
            cfg, start_date=args.start, end_date=args.end
        )
    elif args.months:
        # 只处理第一个月份
        store, std_cache, df_filtered, time_col, range_label = prepare_data(
            cfg, month_filter=args.months[0]
        )
    else:
        store, std_cache, df_filtered, time_col, range_label = prepare_data(cfg)

    print(f"\n{'=' * 60}")
    print(f"评估范围: {range_label}，共 {len(df_filtered)} 条数据")
    print(f"归一化方式对比: {'开启' if args.norm_compare else '关闭'}")
    print(f"{'=' * 60}")

    all_results = []

    for name, weights in weight_configs:
        print(f"\n{'─' * 60}")
        print(f"评估权重方案: {name}")
        print(f"{'─' * 60}")

        # 全局归一化
        print(f"\n  [全局归一化] 准备评估器...")
        fast_eval = FastEvaluator(cfg, std_cache, store, df_filtered, time_col)
        r_global = evaluate_weights(
            name, weights, std_cache, fast_eval, df_filtered, time_col,
            month_label=range_label, norm_method=NormMethod.GLOBAL.value
        )
        all_results.append(r_global)

        # 动态归一化
        if args.norm_compare:
            print(f"\n  [动态归一化] 准备评估器（逐点计算，请耐心等待）...")
            dynamic_eval = DynamicNormEvaluator(cfg, std_cache, store, df_filtered, time_col)
            r_dynamic = evaluate_weights(
                name, weights, std_cache, dynamic_eval, df_filtered, time_col,
                month_label=range_label, norm_method=NormMethod.DYNAMIC.value
            )
            all_results.append(r_dynamic)

    # 打印报告
    print_comparison_report(all_results)

    # 保存报告
    save_report(all_results, args.output)

    print("\n=== 评估完成 ===")
    return all_results


if __name__ == "__main__":
    main()