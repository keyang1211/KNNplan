# -*- coding: utf-8 -*-
"""
config.py — 配置 dataclass + load_config(yaml) + build_feature_weights
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# =========================
# 配置 dataclass
# =========================

@dataclass(frozen=True)
class FeatureConfig:
    raw_features: list[str]
    residual_targets: list[str]
    residual_inputs: list[str]
    weights: dict[str, float]
    residual_weight_ratio: float
    plan_center_cols: list[str]
    load_col: str
    eff_col: str
    heat_value_col: str
    column_aliases: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchingConfig:
    d_weight_s: float
    d_weight_e: float
    top_k: int
    plan_center_mode: int
    topk_avg_method: str
    low_sim_fallback_threshold: float
    enable_low_sim_fallback: bool
    allow_fallback_nearest_load: bool


@dataclass(frozen=True)
class FlowGateConfig:
    enable: bool
    mode: str
    abs_threshold: float
    rel_threshold: float


@dataclass(frozen=True)
class ContinuityConfig:
    enable_rate_limit: bool
    rate_limit_features: list[str]
    rate_limit_abs: dict[str, float]
    reset_on_time_gap: bool
    max_gap_minutes: float


@dataclass(frozen=True)
class FilterConfig:
    enable: bool
    filter_cols: list[str]
    n_load_bins: int
    q_low: float
    q_high: float
    max_bad_features: int
    high_eff_top_ratio: float | None


@dataclass(frozen=True)
class PathsConfig:
    stable_parquet: str
    residual_model_dir: str
    norm_stats_path: str | None = None
    query_parquet: str | None = None
    cache_path: str | None = None


@dataclass(frozen=True)
class TrainConfig:
    input_parquet: str
    output_dir: str
    enable_filter: bool
    filter_cols: list[str]
    filter_q_low: float
    filter_q_high: float
    filter_n_bins: int
    filter_max_bad_features: int
    enable_stratified_split: bool
    train_ratio: float
    valid_ratio: float
    test_ratio: float
    oof_n_splits: int
    residual_model_params: dict


@dataclass(frozen=True)
class PlanningConfig:
    features: FeatureConfig
    matching: MatchingConfig
    flow_gate: FlowGateConfig
    continuity: ContinuityConfig
    paths: PathsConfig
    filter: FilterConfig | None = None
    train: TrainConfig | None = None
    time_col: str | None = None


# =========================
# 工具函数
# =========================

def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base（override 覆盖 base）。"""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(yaml_path: str | Path = None, override: dict[str, Any] = None) -> PlanningConfig:
    """
    从 YAML 加载配置，合并 override（如有），返回 PlanningConfig。

    参数：
        yaml_path: YAML 文件路径（默认读 defaults.yaml）
        override: 需要覆盖的参数字典（可选）

    返回：
        PlanningConfig 实例
    """
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "defaults.yaml"
    else:
        yaml_path = Path(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if override:
        cfg = _deep_merge(cfg, override)

    # 解析各个子配置
    feat_raw = cfg.get("features", {})
    match_raw = cfg.get("matching", {})
    gate_raw = cfg.get("flow_gate", {})
    cont_raw = cfg.get("continuity", {})
    filt_raw = cfg.get("filter", {})
    paths_raw = cfg.get("paths", {})

    features = FeatureConfig(
        raw_features=feat_raw.get("raw_features", []),
        residual_targets=feat_raw.get("residual_targets", []),
        residual_inputs=feat_raw.get("residual_inputs", []),
        weights=feat_raw.get("weights", {}),
        residual_weight_ratio=float(feat_raw.get("residual_weight_ratio", 0.5)),
        plan_center_cols=feat_raw.get("plan_center_cols", []),
        load_col=feat_raw.get("load_col", "主汽流量"),
        eff_col=feat_raw.get("eff_col", "锅炉效率"),
        heat_value_col=feat_raw.get("heat_value_col", "热值"),
        column_aliases=feat_raw.get("column_aliases", {}),
    )

    matching = MatchingConfig(
        d_weight_s=float(match_raw.get("d_weight_s", 1.0)),
        d_weight_e=float(match_raw.get("d_weight_e", 0.0)),
        top_k=int(match_raw.get("top_k", 5)),
        plan_center_mode=int(match_raw.get("plan_center_mode", 2)),
        topk_avg_method=str(match_raw.get("topk_avg_method", "weighted")),
        low_sim_fallback_threshold=float(match_raw.get("low_sim_fallback_threshold", 0.97)),
        enable_low_sim_fallback=bool(match_raw.get("enable_low_sim_fallback", True)),
        allow_fallback_nearest_load=bool(match_raw.get("allow_fallback_nearest_load", False)),
    )

    flow_gate = FlowGateConfig(
        enable=bool(gate_raw.get("enable", True)),
        mode=str(gate_raw.get("mode", "absolute")),
        abs_threshold=float(gate_raw.get("abs_threshold", 15.0)),
        rel_threshold=float(gate_raw.get("rel_threshold", 0.05)),
    )

    continuity = ContinuityConfig(
        enable_rate_limit=bool(cont_raw.get("enable_rate_limit", False)),
        rate_limit_features=cont_raw.get("rate_limit_features", []),
        rate_limit_abs={k: float(v) for k, v in cont_raw.get("rate_limit_abs", {}).items()},
        reset_on_time_gap=bool(cont_raw.get("reset_on_time_gap", True)),
        max_gap_minutes=float(cont_raw.get("max_gap_minutes", 5.0)),
    )

    # filter 配置已移除，保留 None
    filter_cfg = None

    paths = PathsConfig(
        stable_parquet=str(paths_raw.get("stable_parquet", "")),
        residual_model_dir=str(paths_raw.get("residual_model_dir", "")),
        norm_stats_path=paths_raw.get("norm_stats_path", None),
        query_parquet=paths_raw.get("query_parquet", None),
        cache_path=paths_raw.get("cache_path", None),
    )

    # 解析 train 配置（可选）
    train_raw = cfg.get("train", {})
    train_cfg = None
    if train_raw:
        train_cfg = TrainConfig(
            input_parquet=str(train_raw.get("input_parquet", "")),
            output_dir=str(train_raw.get("output_dir", "")),
            enable_filter=bool(train_raw.get("enable_filter", True)),
            filter_cols=train_raw.get("filter_cols", ["炉膛差压", "一次风流量", "床温", "料层差压", "锅炉出口氧量"]),
            filter_q_low=float(train_raw.get("filter_q_low", 0.02)),
            filter_q_high=float(train_raw.get("filter_q_high", 0.98)),
            filter_n_bins=int(train_raw.get("filter_n_bins", 15)),
            filter_max_bad_features=int(train_raw.get("filter_max_bad_features", 2)),
            enable_stratified_split=bool(train_raw.get("enable_stratified_split", True)),
            train_ratio=float(train_raw.get("train_ratio", 0.70)),
            valid_ratio=float(train_raw.get("valid_ratio", 0.15)),
            test_ratio=float(train_raw.get("test_ratio", 0.15)),
            oof_n_splits=int(train_raw.get("oof_n_splits", 5)),
            residual_model_params=train_raw.get("residual_model_params", {}),
        )

    time_col = cfg.get("time_col", None)

    return PlanningConfig(
        features=features,
        matching=matching,
        flow_gate=flow_gate,
        continuity=continuity,
        filter=filter_cfg,
        paths=paths,
        train=train_cfg,
        time_col=time_col,
    )


def build_feature_weights(feat: FeatureConfig) -> dict[str, float]:
    """
    根据 FeatureConfig 构建完整的特征权重字典（含残差特征）。

    返回：
        {特征名: 权重}，残差特征权重 = 原始特征权重 × residual_weight_ratio
    """
    weights = dict(feat.weights)

    for target in feat.residual_targets:
        resid_col = f"resid_{target}"
        raw_weight = weights.get(target, 0.0)
        weights[resid_col] = raw_weight * feat.residual_weight_ratio

    return weights


def validate_config(cfg: PlanningConfig) -> None:
    """
    校验配置合法性。抛出 ValueError 如果校验失败。

    校验规则：
    - d_weight_s + d_weight_e 不能全为0
    - plan_center_cols 必须是 raw_features 的子集
    - 残差目标必须存在于 raw_features（否则无法计算残差）
    """
    if abs(cfg.matching.d_weight_s) < 1e-12 and abs(cfg.matching.d_weight_e) < 1e-12:
        raise ValueError("matching.d_weight_s 和 matching.d_weight_e 不能同时为0")

    raw_set = set(cfg.features.raw_features)
    # plan_center_cols 可以包含 load_col（负荷作为输出控制变量，但不参与相似度）
    center_set = set(cfg.features.plan_center_cols)
    allowed_center = raw_set | {cfg.features.load_col}
    extra = center_set - allowed_center
    if extra:
        raise ValueError(
            f"plan_center_cols 包含 raw_features 和 load_col 之外的列: {extra}"
        )

    for target in cfg.features.residual_targets:
        if target not in raw_set:
            raise ValueError(f"残差目标 {target} 不在 raw_features 中")
