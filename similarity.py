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
            x_norm[i] = x_arr[i]

    # NaN 保护：将 NaN/Inf 替换为 0（中位数归一化后的 0 代表中位数水平）
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

    q_xw = (x_norm * np.sqrt(w)).astype(np.float32)
    return q_xw, w


# =========================
# 相似度
# =========================

def cosine01(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    加权余弦相似度，映射到 [0, 1]。
    输入含 NaN 时，自动替换为 0（中位数归一化后的 0 代表中位数水平，避免中断计算）。

    参数：
        a: (M, D) 加权向量矩阵
        b: (N, D) 加权向量矩阵

    返回：
        (M, N) 相似度矩阵，值域 [0, 1]
    """
    # NaN 保护：将 NaN 替换为 0.0（中位数归一化后的 0 = 中位数水平）
    if np.isscalar(a) or np.isscalar(b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
    else:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

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


# =========================
# 马氏距离（稳定工况查询用）
# =========================

def compute_covariance_matrix(
    X: np.ndarray,
    reg_lambda: float = 1e-6,
) -> np.ndarray:
    """
    计算正则化协方差矩阵。

    参数：
        X: (N, D) 特征矩阵
        reg_lambda: 正则化系数，防奇异

    返回：
        (D, D) 正则化协方差矩阵 Σ_reg = Σ + λ·mean(diag(Σ))·I
    """
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:  # 单特征退化
        cov = cov.reshape(1, 1)
    reg = reg_lambda * np.mean(np.diag(cov)) * np.eye(cov.shape[0])
    return cov + reg


def weighted_cov_inv_matrix(
    cov: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    构造加权协方差逆矩阵 M = W^{1/2} · Σ⁻¹ · W^{1/2}。

    数学等价于加权协方差矩阵 Σ_W = W^{-1/2}·Σ·W^{-1/2} 的逆矩阵。
    权重大的特征在 M 中对应位置被放大，使该特征差异对距离贡献增大。
    权重=0 的特征对应行列为0，不参与距离。

    参数：
        cov: (D, D) 正则化协方差矩阵
        weights: (D,) 归一化权重向量（和为1）

    返回：
        (D, D) 加权协方差逆矩阵 M
    """
    inv_cov = np.linalg.pinv(cov)   # 伪逆，数值稳定
    w_sqrt = np.sqrt(np.maximum(weights, 0.0))
    W_sqrt = np.diag(w_sqrt)
    M = W_sqrt @ inv_cov @ W_sqrt
    return M


def mahalanobis_similarity(
    q: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """
    批量计算马氏距离相似度 S = 1/(1+d²)（柯西核）。

    参数：
        q: (D,) 查询向量（原始特征，未归一化）
        X: (N, D) 标准样本矩阵（原始特征）
        M: (D, D) 加权协方差逆矩阵

    返回：
        (N,) 相似度数组 ∈ (0, 1]，1表示完全相同
    """
    q = np.asarray(q, dtype=float)
    X = np.asarray(X, dtype=float)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    diff = X - q  # (N, D)
    d_sq = np.einsum('ij,jk,ik->i', diff, M, diff)  # (N,) 二次型
    d_sq = np.maximum(d_sq, 0.0)  # 数值保护，避免负数
    return (1.0 / (1.0 + d_sq)).astype(np.float32)
