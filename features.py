# -*- coding: utf-8 -*-
"""
features.py — 残差模型加载 + 残差特征计算（9原→15维）
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import FeatureConfig


def load_residual_models(model_dir: str, targets: list[str]) -> dict[str, object]:
    """
    加载已训练好的残差模型 (.joblib)。

    参数：
        model_dir: 模型文件目录
        targets: 残差目标特征列表（如 ["炉膛差压", "床温", ...]）

    返回：
        {target: model} 字典，模型需有 .predict() 和可选 .feature_names_in_ 属性

    异常：
        FileNotFoundError: 模型文件不存在时
    """
    import joblib

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"残差模型目录不存在: {model_dir}")

    models = {}
    missing = []

    for target in targets:
        model_path = os.path.join(model_dir, f"residual_model_{target}.joblib")
        if not os.path.exists(model_path):
            missing.append(model_path)
            continue
        models[target] = joblib.load(model_path)

    if missing:
        msg = "\n".join(missing)
        raise FileNotFoundError(f"以下残差模型文件不存在:\n{msg}")

    return models


def add_residual_features(
    df: pd.DataFrame,
    models: dict[str, object],
    feat: FeatureConfig,
    name: str = "数据",
) -> pd.DataFrame:
    """
    使用已训练好的残差模型生成 resid_ 特征。

    对每个目标特征:
        pred_target = model(residual_inputs)
        resid_target = target - pred_target

    参数：
        df: 含原始特征列的 DataFrame
        models: {target: model} 字典
        feat: FeatureConfig
        name: 日志标识

    返回：
        DataFrame，新增 pred_* 和 resid_* 列（仅 resid_* 参与相似度，pred_* 仅诊断用）
    """
    out = df.copy()

    # 确保残差模型输入列存在
    for c in feat.residual_inputs:
        if c not in out.columns:
            raise ValueError(f"{name}: 缺少残差模型输入列 '{c}'")

    # 确保残差目标真实值列存在
    for target in models:
        if target not in out.columns:
            raise ValueError(f"{name}: 缺少残差目标真实值列 '{target}'")

    for target, model in models.items():
        pred_col = f"pred_{target}"
        resid_col = f"resid_{target}"

        # 模型输入列：优先用 model.feature_names_in_，否则用 feat.residual_inputs
        input_cols = list(getattr(model, "feature_names_in_", feat.residual_inputs))
        for c in input_cols:
            if c not in out.columns:
                raise ValueError(f"{name}: 模型 '{target}' 需要输入列 '{c}'，但数据中不存在")

        pred = model.predict(out[input_cols].astype(float))
        out[pred_col] = pred.astype(float)
        out[resid_col] = pd.to_numeric(out[target], errors="coerce") - out[pred_col]

    return out


def make_query_vector_15d(
    raw_row: pd.Series | dict,
    models: dict[str, object],
    feat: FeatureConfig,
) -> np.ndarray:
    """
    单次调用专用：将9个原始特征扩展为15维相似度向量。

    步骤：
    1. 用残差模型对 residual_inputs 预测 pred_target
    2. resid_target = raw[target] - pred_target
    3. 拼接 raw_features + residual_features = 15 维

    参数：
        raw_row: 原始特征值（Series 或 dict，需含 load_col 和 raw_features 中的所有列）
        models: {target: model} 字典
        feat: FeatureConfig

    返回：
        (15,) 特征向量，顺序为 [raw_features..., resid_*...]
    """
    if isinstance(raw_row, dict):
        raw_row = pd.Series(raw_row)

    # 检查必要列
    needed = set(feat.raw_features + feat.residual_inputs)
    missing = needed - set(raw_row.index)
    if missing:
        raise ValueError(f"输入特征缺少列: {missing}")

    # 构建 raw 部分（9维）
    result = {}
    for c in feat.raw_features:
        result[c] = float(raw_row[c])

    # 计算残差（6维）
    for target, model in models.items():
        input_cols = list(getattr(model, "feature_names_in_", feat.residual_inputs))
        x = pd.DataFrame([[float(raw_row[c]) for c in input_cols]], columns=input_cols)
        pred = model.predict(x)[0]
        resid_col = f"resid_{target}"
        result[resid_col] = float(raw_row[target]) - float(pred)

    # 拼成有序数组：先 raw_features，再 residual_features
    sim_feature_cols = feat.raw_features + [f"resid_{t}" for t in feat.residual_targets]
    return np.array([result[c] for c in sim_feature_cols], dtype=float)
