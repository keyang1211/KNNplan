# -*- coding: utf-8 -*-
"""
dtw_fast.py — DTW Fast 模式（Numba JIT 加速）

仅替换核心计算函数，不改变候选池和匹配逻辑。
与 dtw_query.py 中的原始 Python 实现完全兼容。

用法：
    from plan_center.dtw_fast import dtw_align_jit, dtw_weighted_cosine_mean_jit
    aligned_pairs, path_cost = dtw_align_jit(query_seq, cand_seq, feature_weights)
    sim = dtw_weighted_cosine_mean_jit(query_seq, cand_seq, aligned_pairs)
"""

from __future__ import annotations

import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, cache=True)
def dtw_align_jit(
    query_seq: np.ndarray,
    cand_seq: np.ndarray,
    feature_weights: np.ndarray,
) -> tuple[list, float]:
    """
    Numba JIT 加速的 DTW 对齐。

    参数：
        query_seq: (T_q, D) 查询序列
        cand_seq: (T_c, D) 候选序列
        feature_weights: (D,) 各特征的平方根权重

    返回：
        (aligned_pairs, path_cost)
        aligned_pairs: [(i_q, i_c), ...]，正序
        path_cost: 最小累积代价（DTW 距离）
    """
    n, m, d = query_seq.shape[0], cand_seq.shape[0], query_seq.shape[1]

    # DP 成本矩阵（使用大值代替 np.inf，兼容 numba）
    D = np.full((n, m), 1e20, dtype=np.float64)
    D[0, 0] = 0.0

    # 内点 DP 计算（合并首行、首列和内点逻辑）
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue

            # 计算欧氏距离
            diff = query_seq[i] - cand_seq[j]
            dist_sq = 0.0
            for k in range(d):
                dist_sq += diff[k] * diff[k] * feature_weights[k]
            dist = np.sqrt(dist_sq)

            # DP 递推
            if i == 0:
                D[i, j] = D[i, j - 1] + dist
            elif j == 0:
                D[i, j] = D[i - 1, j] + dist
            else:
                d1 = D[i - 1, j - 1]
                d2 = D[i - 1, j]
                d3 = D[i, j - 1]
                D[i, j] = dist + min(d1, d2, d3)

    # 路径回溯
    aligned_pairs = []
    i, j = n - 1, m - 1
    aligned_pairs.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            d1 = D[i - 1, j - 1]
            d2 = D[i - 1, j]
            d3 = D[i, j - 1]
            if d1 <= d2 and d1 <= d3:
                i -= 1
                j -= 1
            elif d2 <= d3:
                i -= 1
            else:
                j -= 1
        aligned_pairs.append((i, j))

    aligned_pairs.reverse()
    return aligned_pairs, float(D[-1, -1])


@jit(nopython=True, cache=True)
def dtw_align_with_coverage_jit(
    query_seq: np.ndarray,
    cand_seq: np.ndarray,
    feature_weights: np.ndarray,
    min_coverage: int,
) -> tuple[list, float, bool]:
    """
    DTW 对齐 + 覆盖帧检查（JIT 版本）。

    参数：
        query_seq: (T_q, D) 查询序列
        cand_seq: (T_c, D) 候选序列
        feature_weights: (D,) 各特征的平方根权重
        min_coverage: 最少不重复候选帧数

    返回：
        (aligned_pairs, path_cost, coverage_ok)
    """
    aligned_pairs, path_cost = dtw_align_jit(query_seq, cand_seq, feature_weights)

    # 早期检查覆盖帧数
    seen_count = 0
    seen_last = -1
    for _, j in aligned_pairs:
        if j != seen_last:
            seen_last = j
            seen_count += 1
            if seen_count >= min_coverage:
                return aligned_pairs, path_cost, True

    return aligned_pairs, path_cost, False


@jit(nopython=True, cache=True)
def dtw_weighted_cosine_mean_jit(
    query_seq: np.ndarray,
    cand_seq: np.ndarray,
    aligned_pairs: list,
) -> float:
    """
    DTW 对齐后的加权余弦相似度均值（JIT 版本）。

    参数：
        query_seq: (T_q, D) 查询序列
        cand_seq: (T_c, D) 候选序列
        aligned_pairs: DTW 对齐索引对 [(i_q, i_c), ...]

    返回：
        float，均值相似度 [0, 1]
    """
    if len(aligned_pairs) == 0:
        return 0.0

    d = query_seq.shape[1]
    total = 0.0

    for k in range(len(aligned_pairs)):
        i, j = aligned_pairs[k]
        q = query_seq[i]
        c = cand_seq[j]

        # 内联余弦相似度计算
        dot = 0.0
        q_norm_sq = 0.0
        c_norm_sq = 0.0
        for dim in range(d):
            dot += q[dim] * c[dim]
            q_norm_sq += q[dim] * q[dim]
            c_norm_sq += c[dim] * c[dim]

        q_norm = np.sqrt(q_norm_sq)
        c_norm = np.sqrt(c_norm_sq)

        if q_norm > 1e-10 and c_norm > 1e-10:
            cos = dot / (q_norm * c_norm)
            total += (cos + 1.0) * 0.5  # 映射到 [0, 1]

    return total / len(aligned_pairs)


# ============================================================
# 预热函数
# ============================================================

def warmup_jit(sample_query: np.ndarray, sample_cand: np.ndarray, weights: np.ndarray, n_iter: int = 3):
    """
    预热 JIT 编译，避免首次调用开销。

    参数：
        sample_query: (T_q, D) 样例查询序列
        sample_cand: (T_c, D) 样例候选序列
        weights: (D,) 特征权重
        n_iter: 预热迭代次数
    """
    for _ in range(n_iter):
        dtw_align_jit(sample_query, sample_cand, weights)
        dtw_align_with_coverage_jit(sample_query, sample_cand, weights, min_coverage=4)
        dtw_weighted_cosine_mean_jit(sample_query, sample_cand,
                                      [(i, i) for i in range(min(len(sample_query), len(sample_cand)))])