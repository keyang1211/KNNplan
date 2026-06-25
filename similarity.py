# -*- coding: utf-8 -*-
"""
similarity.py — 相似度计算的纯函数集合
从 notebook 代码块2抽取，参数化，去除全局常量依赖
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import FlowGateConfig


# =========================
# 归一化
# =========================

def robust_norm_stats(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    计算 robust 归一化参数：median 和 IQR。

    返回：
        {col: {"median": float, "iqr": float, "mean": float, "std": float}}
        其中 iqr 接近0时退化为1.0，避免除0
    """
    stats = {}
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        iqr = q75 - q25
        std = s.std(ddof=0)
        stats[c] = {
            "median": float(s.median()),
            "iqr": float(iqr if abs(iqr) > 1e-12 else 1.0),
            "mean": float(s.mean()),
            "std": float(std if abs(std) > 1e-12 else 1.0),
        }
    return stats


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    norm_stats: dict,
    normalize_all: bool = False,
) -> pd.DataFrame:
    """
    Robust 归一化：z = (x - median) / IQR

    参数：
        normalize_all: True=所有特征都归一化；False=只归一化 norm_stats 中存在的特征

    返回：
        归一化后的 DataFrame（只含 feature_cols 列）
    """
    x = df[feature_cols].astype(float).copy()
    for c in feature_cols:
        if normalize_all or c in norm_stats:
            x[c] = (x[c] - norm_stats[c]["median"]) / norm_stats[c]["iqr"]
        # else: 保持原值（不做归一化）
    return x


# =========================
# 加权
# =========================

def weight_array(feature_cols: list[str], weights_dict: dict[str, float]) -> np.ndarray:
    """
    根据权重字典生成归一化权重向量（求和为1）。

    权重为负数时裁剪为0。所有权重之和若≤0，抛出 ValueError。
    """
    w = np.array([weights_dict.get(c, 0.0) for c in feature_cols], dtype=float)
    w = np.maximum(w, 0.0)

    if w.sum() <= 1e-12:
        raise ValueError("所有特征权重均为0，无法构建加权向量")

    w = w / w.sum()
    return w


def weighted_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    norm_stats: dict,
    weights_dict: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建加权特征矩阵。

    对所有特征做 robust 归一化，然后乘以 sqrt(归一化权重)，
    使得后续标准余弦相似度等价于加权余弦相似度。

    返回：
        (xw, normalized_weights)
        xw: (N, D) 加权特征矩阵，float32
        normalized_weights: (D,) 归一化后的权重向量
    """
    x_norm = normalize_features(df, feature_cols, norm_stats, normalize_all=True)
    w = weight_array(feature_cols, weights_dict=weights_dict)
    xw = (x_norm[feature_cols].values * np.sqrt(w)).astype(np.float32)
    return xw, w


def weighted_vector_1d(
    values: np.ndarray | pd.Series,
    feature_cols: list[str],
    norm_stats: dict,
    weights_dict: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    对单条向量做归一化+加权（query_one 专用，避免构建单行 DataFrame）。

    对所有特征做 robust 归一化。

    参数：
        values: 与 feature_cols 顺序对应的特征值数组/Series

    返回：
        (q_xw, normalized_weights)
        q_xw: (D,) 加权特征向量，float32
    """
    x_arr = np.asarray(values, dtype=float)
    w = weight_array(feature_cols, weights_dict=weights_dict)

    # 归一化所有特征
    x_norm = np.empty_like(x_arr)
    for i, c in enumerate(feature_cols):
        if c in norm_stats:
            x_norm[i] = (x_arr[i] - norm_stats[c]["median"]) / norm_stats[c]["iqr"]
        else:
            # 无归一化参数时，使用 min-max 归一化到 [0,1]（临时方案）
            # 实际应确保所有特征都在 norm_stats 中
            x_norm[i] = x_arr[i]

    q_xw = (x_norm * np.sqrt(w)).astype(np.float32)
    return q_xw, w


# =========================
# 相似度
# =========================

def cosine01(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    加权余弦相似度，映射到 [0, 1]。

    参数：
        a: (M, D) 加权向量矩阵
        b: (N, D) 加权向量矩阵

    返回：
        (M, N) 相似度矩阵，值域 [0, 1]
    """
    sim = cosine_similarity(a, b)
    sim = np.clip(sim, -1.0, 1.0)
    return ((sim + 1.0) / 2.0).astype(np.float32)


# =========================
# 硬门控
# =========================

def flow_gate_keep_mask(
    load_query: float,
    load_candidates: np.ndarray,
    gate: FlowGateConfig,
) -> np.ndarray:
    """
    主汽流量硬门控：返回候选样本是否允许参与匹配的布尔掩码。

    参数：
        load_query: 查询样本的主汽流量
        load_candidates: (N,) 候选样本的主汽流量
        gate: FlowGateConfig 配置

    返回：
        (N,) 布尔数组
    """
    if not gate.enable:
        return np.ones(len(load_candidates), dtype=bool)

    diff = np.abs(float(load_query) - np.asarray(load_candidates, dtype=float))

    if gate.mode == "absolute":
        return diff <= gate.abs_threshold
    elif gate.mode == "relative":
        denom = np.maximum(np.abs(load_candidates), 1e-9)
        return (diff / denom) <= gate.rel_threshold
    else:
        raise ValueError(f"flow_gate.mode 只能是 'absolute' 或 'relative'，当前为 '{gate.mode}'")


# =========================
# 分位数归一化（用于效率得分 E）
# =========================

def pct_rank(values: np.ndarray) -> np.ndarray:
    """
    分位数归一化（百分位排名），映射到 [0, 1]。

    长度为1时返回1。空数组返回空。
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    if len(values) == 1:
        return np.ones(1, dtype=float)

    s = pd.Series(values)
    return s.rank(pct=True, method="average").values.astype(float)
