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
    date_col: str = "日期"      # 稳定工况数据中的日期列名
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
    covariance_path: str | None = None
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
class OptimizeConfig:
    learning_rate: float
    num_epochs: int
    batch_days: int
    fd_step: float
    random_seed: int
    max_weight: float
    loss_feature_weights: dict[str, float]
    disable_fallback_during_opt: bool
    report_csv: str
    report_json: str
    grad_mode: str = "forward"      # "forward"=前向差分(快), "central"=中心差分(准)
    query_stride: int = 1           # 查询数据降采样步长（每 N 行取 1 行，1=不降采样）


@dataclass(frozen=True)
class OptimizeGeneticConfig:
    """遗传进化算法权重寻优配置。"""
    population_size: int
    n_generations: int
    elite_size: int
    tournament_size: int
    crossover_rate: float
    crossover_method: str       # "blend" | "uniform"
    mutation_rate: float
    mutation_scale: float
    min_weight: float
    max_weight: float
    query_stride: int
    batch_days: int
    sbx_eta: float
    disable_fallback_during_opt: bool
    report_csv: str
    report_json: str


@dataclass(frozen=True)
class OptimizeV2Config:
    """遗传进化算法 V2 配置（单 evaluator 版）。"""
    population_size: int
    n_generations: int
    elite_size: int
    tournament_size: int
    crossover_rate: float
    crossover_method: str
    mutation_rate: float
    mutation_scale: float
    min_weight: float
    max_weight: float
    query_stride: int
    sbx_eta: float
    disable_fallback_during_opt: bool
    loss_feature_weights: dict[str, float]
    report_csv: str
    report_json: str


@dataclass(frozen=True)
class DTWPrefilterConfig:
    """DTW 预筛配置（负荷初筛 + 马氏距离相似度筛选）。"""
    enable: bool = True              # 是否启用预筛
    load_threshold: float = 15.0     # 负荷初筛阈值（t/h）
    sim_threshold: float = 0.5       # 柯西核相似度阈值 S=1/(1+d²)
    slide_step: int = 1              # 预筛滑窗步长（分钟）


@dataclass(frozen=True)
class DTWQueryConfig:
    """DTW 时序查询配置。"""
    ref_days: int              # 参考窗口天数
    query_seq_len: int         # 查询序列长度（分钟）
    dtw_min_len: int           # DTW 候选最短长度（分钟）
    dtw_max_len: int           # DTW 候选最长长度（分钟）
    slide_step: int            # 滑动步长（分钟）
    top_k: int                 # Top-k
    resid_cache_parquet: str   # 残差缓存 parquet 路径
    dtw_feature_weights: dict[str, float]  # DTW 距离权重（欧氏距离中各特征权重）
    prefilter: DTWPrefilterConfig = field(default_factory=DTWPrefilterConfig)
    n_workers: int = 4             # 单次查询内部 DTW 并行线程数（numpy 释放 GIL）
    sakoe_chiba_w: int = 1         # Sakoe-Chiba 带宽（0=无约束，>=1 限制 |i-j|<=w）
    min_coverage: int = 4          # DTW 路径最少不重复候选帧数（独立于 dtw_min_len）


@dataclass(frozen=True)
class PlanningConfig:
    features: FeatureConfig
    matching: MatchingConfig
    flow_gate: FlowGateConfig
    continuity: ContinuityConfig
    paths: PathsConfig
    dtw_query: DTWQueryConfig | None = None
    filter: FilterConfig | None = None
    train: TrainConfig | None = None
    optimize: OptimizeConfig | None = None
    optimize_genetic: OptimizeGeneticConfig | None = None
    optimize_v2: OptimizeV2Config | None = None
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
        date_col=str(feat_raw.get("date_col", "日期")),
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
        covariance_path=paths_raw.get("covariance_path", None),
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

    # 解析 optimize 配置（可选）
    opt_raw = cfg.get("optimize", {})
    opt_cfg = None
    if opt_raw:
        opt_cfg = OptimizeConfig(
            learning_rate=float(opt_raw.get("learning_rate", 0.05)),
            num_epochs=int(opt_raw.get("num_epochs", 20)),
            batch_days=int(opt_raw.get("batch_days", 5)),
            fd_step=float(opt_raw.get("fd_step", 1e-3)),
            random_seed=int(opt_raw.get("random_seed", 42)),
            max_weight=float(opt_raw.get("max_weight", 5.0)),
            loss_feature_weights={k: float(v) for k, v in opt_raw.get("loss_feature_weights", {}).items()},
            disable_fallback_during_opt=bool(opt_raw.get("disable_fallback_during_opt", True)),
            report_csv=str(opt_raw.get("report_csv", "optimize_report.csv")),
            report_json=str(opt_raw.get("report_json", "optimize_report.json")),
            grad_mode=str(opt_raw.get("grad_mode", "forward")),
            query_stride=int(opt_raw.get("query_stride", 1)),
        )

    # 解析 optimize_genetic 配置（可选）
    opt_gen_raw = cfg.get("optimize_genetic", {})
    opt_gen_cfg = None
    if opt_gen_raw:
        opt_gen_cfg = OptimizeGeneticConfig(
            population_size=int(opt_gen_raw.get("population_size", 50)),
            n_generations=int(opt_gen_raw.get("n_generations", 40)),
            elite_size=int(opt_gen_raw.get("elite_size", 5)),
            tournament_size=int(opt_gen_raw.get("tournament_size", 3)),
            crossover_rate=float(opt_gen_raw.get("crossover_rate", 0.8)),
            crossover_method=str(opt_gen_raw.get("crossover_method", "blend")),
            mutation_rate=float(opt_gen_raw.get("mutation_rate", 0.15)),
            mutation_scale=float(opt_gen_raw.get("mutation_scale", 0.1)),
            min_weight=float(opt_gen_raw.get("min_weight", 0.0)),
            max_weight=float(opt_gen_raw.get("max_weight", 5.0)),
            query_stride=int(opt_gen_raw.get("query_stride", 5)),
            batch_days=int(opt_gen_raw.get("batch_days", 5)),
            sbx_eta=float(opt_gen_raw.get("sbx_eta", 15.0)),
            disable_fallback_during_opt=bool(opt_gen_raw.get("disable_fallback_during_opt", True)),
            report_csv=str(opt_gen_raw.get("report_csv", "optimize_genetic_report.csv")),
            report_json=str(opt_gen_raw.get("report_json", "optimize_genetic_report.json")),
        )

    # 解析 optimize_v2 配置（可选）
    opt_v2_raw = cfg.get("optimize_v2", {})
    opt_v2_cfg = None
    if opt_v2_raw:
        opt_v2_cfg = OptimizeV2Config(
            population_size=int(opt_v2_raw.get("population_size", 50)),
            n_generations=int(opt_v2_raw.get("n_generations", 40)),
            elite_size=int(opt_v2_raw.get("elite_size", 5)),
            tournament_size=int(opt_v2_raw.get("tournament_size", 3)),
            crossover_rate=float(opt_v2_raw.get("crossover_rate", 0.8)),
            crossover_method=str(opt_v2_raw.get("crossover_method", "blend")),
            mutation_rate=float(opt_v2_raw.get("mutation_rate", 0.15)),
            mutation_scale=float(opt_v2_raw.get("mutation_scale", 0.1)),
            min_weight=float(opt_v2_raw.get("min_weight", 0.0)),
            max_weight=float(opt_v2_raw.get("max_weight", 1.0)),
            query_stride=int(opt_v2_raw.get("query_stride", 5)),
            sbx_eta=float(opt_v2_raw.get("sbx_eta", 15.0)),
            disable_fallback_during_opt=bool(opt_v2_raw.get("disable_fallback_during_opt", True)),
            loss_feature_weights={k: float(v) for k, v in opt_v2_raw.get("loss_feature_weights", {}).items()},
            report_csv=str(opt_v2_raw.get("report_csv", "optimize_v2_report.csv")),
            report_json=str(opt_v2_raw.get("report_json", "optimize_v2_report.json")),
        )

    # 解析 dtw_query 配置（可选）
    dtw_raw = cfg.get("dtw_query", {})
    dtw_cfg = None
    if dtw_raw:
        dtw_cfg = DTWQueryConfig(
            ref_days=int(dtw_raw.get("ref_days", 3)),
            query_seq_len=int(dtw_raw.get("query_seq_len", 5)),
            dtw_min_len=int(dtw_raw.get("dtw_min_len", 4)),
            dtw_max_len=int(dtw_raw.get("dtw_max_len", 6)),
            slide_step=int(dtw_raw.get("slide_step", 1)),
            top_k=int(dtw_raw.get("top_k", 5)),
            resid_cache_parquet=str(dtw_raw.get("resid_cache_parquet", "")),
            dtw_feature_weights={k: float(v) for k, v in dtw_raw.get("dtw_feature_weights", {}).items()},
            prefilter=DTWPrefilterConfig(
                enable=bool(dtw_raw.get("prefilter", {}).get("enable", True)),
                load_threshold=float(dtw_raw.get("prefilter", {}).get("load_threshold", 15.0)),
                sim_threshold=float(dtw_raw.get("prefilter", {}).get("sim_threshold", 0.5)),
                slide_step=int(dtw_raw.get("prefilter", {}).get("slide_step", 1)),
            ),
            n_workers=int(dtw_raw.get("n_workers", 4)),
            sakoe_chiba_w=int(dtw_raw.get("sakoe_chiba_w", 1)),
            min_coverage=int(dtw_raw.get("min_coverage", 4)),
        )

    return PlanningConfig(
        features=features,
        matching=matching,
        flow_gate=flow_gate,
        continuity=continuity,
        filter=filter_cfg,
        paths=paths,
        dtw_query=dtw_cfg,
        train=train_cfg,
        optimize=opt_cfg,
        optimize_genetic=opt_gen_cfg,
        optimize_v2=opt_v2_cfg,
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
