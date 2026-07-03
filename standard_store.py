# -*- coding: utf-8 -*-
"""
standard_store.py — 标准样本 V（向量数据库）的加载与管理

简化版：向量数据库 parquet 已经是完全处理好的，包含原始特征 + resid_* 列 + 锅炉效率列。
本模块只负责：读 parquet → 加载归一化参数 → 构建加权矩阵 → 计算效率分位数 E → 缓存
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import PlanningConfig, build_feature_weights
from .similarity import robust_norm_stats, pct_rank


# =========================
# 标准样本数据结构
# =========================

@dataclass
class StandardStore:
    """加载并预处理后的标准样本 V。"""

    df_standard: pd.DataFrame          # N 行标准样本（已含 resid_* 和锅炉效率列）
    loads_standard: np.ndarray         # (N,) 主汽流量
    sim_feature_cols: list[str]        # D 个相似度特征列名（raw + residual）
    eff_score_all: np.ndarray          # (N,) 效率分位数 E


# =========================
# 辅助函数
# =========================

def deduplicate_columns_keep_first(df: pd.DataFrame) -> pd.DataFrame:
    """处理重复列名，保留第一次出现的列。"""
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        print(f"发现重复列名，已保留第一次出现并删除后续重复列: {sorted(set(duplicated))}")
    return df.loc[:, ~df.columns.duplicated()].copy()


def require_cols(df: pd.DataFrame, cols: list[str], name: str = "数据") -> None:
    """检查 df 是否包含所需列，缺失则抛出 ValueError。"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要列: {missing}")


def to_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """将指定列转为数值（错误转 NaN）。"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# 缓存签名
# =========================

def _stable_file_signature(path: str) -> dict:
    """用于判断标准样本缓存是否仍然有效。"""
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "size": int(p.stat().st_size) if p.exists() else -1,
        "mtime_ns": int(p.stat().st_mtime_ns) if p.exists() else -1,
    }


def build_cache_signature(cfg: PlanningConfig) -> dict:
    """缓存签名：影响标准样本的参数都放进来。"""
    sig = {
        "stable_file": _stable_file_signature(cfg.paths.stable_parquet),
        "raw_features": list(cfg.features.raw_features),
        "residual_targets": list(cfg.features.residual_targets),
        "weights": {k: float(v) for k, v in sorted(cfg.features.weights.items())},
        "residual_weight_ratio": float(cfg.features.residual_weight_ratio),
        "eff_col": cfg.features.eff_col,
        "load_col": cfg.features.load_col,
    }
    return sig


def try_load_cache(cfg: PlanningConfig) -> StandardStore | None:
    """尝试加载缓存。签名不匹配返回 None。"""
    if not cfg.paths.cache_path or not os.path.exists(cfg.paths.cache_path):
        return None

    try:
        payload = joblib.load(cfg.paths.cache_path)
        if payload.get("signature") == build_cache_signature(cfg):
            print(f"已命中标准样本缓存: {cfg.paths.cache_path}")
            return payload.get("store")
        print("发现标准样本缓存，但配置或输入文件已变化，重新构建。")
        return None
    except Exception as e:
        print(f"读取标准样本缓存失败，将重新构建: {e}")
        return None


def save_cache(store: StandardStore, cfg: PlanningConfig) -> None:
    """保存缓存。"""
    if not cfg.paths.cache_path:
        return
    try:
        payload = {
            "signature": build_cache_signature(cfg),
            "store": store,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        joblib.dump(payload, cfg.paths.cache_path, compress=3)
        print(f"已保存标准样本缓存: {cfg.paths.cache_path}")
    except Exception as e:
        print(f"保存标准样本缓存失败，不影响后续计算: {e}")


# =========================
# 主构建函数（简化版）
# =========================

def build_standard_store(cfg: PlanningConfig) -> StandardStore:
    """
    读取向量数据库 parquet，加载归一化参数，构建标准样本 V。

    步骤:
    1. 尝试加载缓存
    2. 读 parquet
    3. 加载归一化参数（从 norm_stats.json）
    4. 构建加权特征矩阵
    5. 计算效率分位数 E
    6. 保存缓存
    """
    # 1. 缓存
    cached = try_load_cache(cfg)
    if cached is not None:
        return cached

    t0 = time.perf_counter()
    feat = cfg.features
    residual_feat_cols = [f"resid_{t}" for t in feat.residual_targets]
    sim_feature_cols = feat.raw_features + residual_feat_cols

    # 2. 读取 parquet
    df = pd.read_parquet(cfg.paths.stable_parquet)
    df = deduplicate_columns_keep_first(df)

    # 应用列别名映射（将数据源的列名对齐到配置中的特征名）
    aliases = cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in df.columns:
                if new in df.columns:
                    # 目标列已存在，直接删除源列（避免重复列）
                    df = df.drop(columns=[old])
                    print(f"目标列 '{new}' 已存在，已删除源列 '{old}'")
                else:
                    df = df.rename(columns={old: new})
                    print(f"已应用列别名映射: {old} → {new}")

    print(f"向量数据库读取形状: {df.shape}，耗时 {time.perf_counter() - t0:.2f}s")

    # 3. 检查必要列
    required_cols = [feat.load_col] + sim_feature_cols + [feat.eff_col]
    require_cols(df, required_cols, "向量数据库")

    # 数值转换
    df = to_numeric_inplace(df, required_cols)

    # 删除核心特征缺失的样本
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"标准样本数量: {len(df)}")

    # 4. 效率分位数 E
    eff_score_all = pct_rank(df[feat.eff_col].values.astype(float))

    loads_standard = df[feat.load_col].values.astype(float)

    store = StandardStore(
        df_standard=df,
        loads_standard=loads_standard.astype(np.float32),
        sim_feature_cols=sim_feature_cols,
        eff_score_all=eff_score_all.astype(np.float32),
    )

    save_cache(store, cfg)
    print(f"标准样本构建完成，样本数: {len(df)}，总耗时 {time.perf_counter() - t0:.2f}s")
    return store
