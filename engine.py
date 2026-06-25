# -*- coding: utf-8 -*-
"""
engine.py — PlanningEngine 类（组装所有模块，持有 V + 模型）
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PlanningConfig, load_config, validate_config
from .features import load_residual_models
from .query import query_one, query_one_full
from .schemas import PlanResult
from .standard_store import StandardStore, build_standard_store


class PlanningEngine:
    """
    规划中心引擎：持有配置、标准样本 V、残差模型，提供单次和批量入口。

    用法：
        engine = PlanningEngine("config.yaml")
        result = engine.plan_one(raw_features)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: PlanningConfig | None = None,
        skip_validation: bool = False,
    ):
        """
        参数：
            config_path: YAML 配置文件路径（默认读 defaults.yaml）
            config: 直接传入 PlanningConfig（优先级高于 config_path）
            skip_validation: 跳过配置校验（调试用）
        """
        if config is not None:
            self.cfg = config
        else:
            self.cfg = load_config(config_path)

        if not skip_validation:
            validate_config(self.cfg)

        # 加载标准样本 V 和残差模型
        self.store: StandardStore = build_standard_store(self.cfg)
        self.models: dict[str, object] = load_residual_models(
            self.cfg.paths.residual_model_dir,
            self.cfg.features.residual_targets,
        )

    # =========================
    # 单次调用接口
    # =========================

    def plan_one(
        self,
        raw_features: dict[str, float] | pd.Series,
        prev_center: dict[str, float] | None = None,
        prev_time: object = None,
        current_time: object = None,
    ) -> PlanResult:
        """
        单次调用主接口（含连续性处理）。

        参数：
            raw_features: 原始特征字典/Series（含 load_col + raw_features 的所有列）
            prev_center: 上一分钟最终中心（None=首点）
            prev_time: 上一分钟时间戳
            current_time: 当前时间戳

        返回：
            PlanResult
        """
        return query_one_full(
            raw_features=raw_features,
            store=self.store,
            cfg=self.cfg,
            models=self.models,
            prev_center=prev_center,
            prev_time=prev_time,
            current_time=current_time,
        )

    def plan_one_no_continuity(
        self,
        raw_features: dict[str, float] | pd.Series,
    ) -> PlanResult:
        """
        单次调用（不含连续性处理）。

        适用于：调试、不需要连续性约束的场景。
        """
        return query_one(
            raw_features=raw_features,
            store=self.store,
            cfg=self.cfg,
            models=self.models,
        )

    # =========================
    # 工具方法
    # =========================

    def reload_standard_store(self) -> None:
        """重新加载标准样本 V（数据更新后调用）。"""
        self.store = build_standard_store(self.cfg)

    def __repr__(self) -> str:
        return (
            f"PlanningEngine("
            f"标准样本数={len(self.store.df_standard)}, "
            f"残差模型数={len(self.models)}, "
            f"相似度特征维度={len(self.store.sim_feature_cols)})"
        )
