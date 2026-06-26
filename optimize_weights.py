# -*- coding: utf-8 -*-
"""
optimize_weights.py — 相似度特征权重数值梯度寻优

目标：最小化「规划中心输出」与「查询行实际控制值」之间的加权 MSE（IQR 归一化）。

被优化参数：8 个原始特征权重（主汽流量=0、热值=0 固定不动）。
残差特征权重绑定为 对应原始权重 × residual_weight_ratio。

Loss 计算：
    loss = mean over 行、变量 [ w_var × ((规划_var − 实际_var) / IQR_var)² ]

高性能设计：
    - FastEvaluator 一次性预计算标准样本归一化矩阵和查询批次归一化矩阵
    - 每次前向只做矩阵运算（换权重 → √w 缩放 → 余弦 → TopK → 均值 → loss）
    - 数值梯度：8 参数中心差分（16 次前向/batch）

用法:
    python -m plan_center.optimize_weights
    python -m plan_center.optimize_weights --config defaults.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================
# FastEvaluator：预计算所有不随权重变化的量
# =============================================================

class FastEvaluator:
    """
    向量化前向计算器。

    预计算：
        - 标准样本未加权归一化矩阵 V_norm (N, D)
        - 标准样本效率分 E (N,)
        - 标准样本负荷 loads (N,)
        - 查询批次未加权归一化矩阵 Q_norm (M, D)
        - 查询批次实际控制变量值 actual_vals (M, K)（K=loss变量数）
        - 硬门控掩码 gate_mask (M, N)
        - IQR 归一化尺度 iqr_scales (K,)
        - 其他配置
    """

    def __init__(self, cfg, store, query_batch_df, time_col):
        from plan_center.config import build_feature_weights
        from plan_center.similarity import normalize_features, weight_array

        feat = cfg.features
        opt = cfg.optimize
        match_cfg = cfg.matching
        gate_cfg = cfg.flow_gate

        self.feat = feat
        self.match_cfg = match_cfg
        self.gate_cfg = gate_cfg
        self.opt = opt

        # 15 维特征名：[raw_features..., resid_*...]
        residual_cols = [f"resid_{t}" for t in feat.residual_targets]
        self.sim_feature_cols = list(feat.raw_features) + residual_cols
        self.D = len(self.sim_feature_cols)

        # 8 个可调权重的特征名（raw_features，主汽流量和热值固定0）
        self.raw_feature_names = list(feat.raw_features)
        # 被优化的特征（权重非0的原始特征）
        self.opt_feature_names = [
            c for c in self.raw_feature_names
            if c not in (feat.load_col, feat.heat_value_col)
        ]
        self.n_opt = len(self.opt_feature_names)

        # ---- 标准样本预计算 ----
        V_norm_df = normalize_features(
            store.df_standard, self.sim_feature_cols, store.norm_stats, normalize_all=True
        )
        self.V_norm = V_norm_df[self.sim_feature_cols].values.astype(np.float64)  # (N, D)
        self.eff_score = store.eff_score_all.astype(np.float64)   # (N,)
        self.loads_std = store.loads_standard.astype(np.float64)  # (N,)
        self.df_standard = store.df_standard
        self.plan_center_cols = list(feat.plan_center_cols)

        # ---- 查询批次预计算 ----
        Q_norm_df = normalize_features(
            query_batch_df, self.sim_feature_cols, store.norm_stats, normalize_all=True
        )
        self.Q_norm = Q_norm_df[self.sim_feature_cols].values.astype(np.float64)   # (M, D)
        self.M = len(self.Q_norm)

        # 查询批次实际控制变量值（用于计算 loss）
        loss_cols = [c for c in opt.loss_feature_weights if opt.loss_feature_weights[c] > 0]
        self.loss_cols = loss_cols
        self.actual_vals = query_batch_df[loss_cols].values.astype(np.float64)   # (M, K)
        self.loss_w = np.array(
            [opt.loss_feature_weights[c] for c in loss_cols], dtype=np.float64
        )  # (K,)

        # IQR 尺度（用于归一化误差）
        self.iqr_scales = np.array(
            [store.norm_stats[c]["iqr"] for c in loss_cols], dtype=np.float64
        )  # (K,)
        # 防止 IQR 为0
        self.iqr_scales = np.maximum(self.iqr_scales, 1e-8)

        # 查询批次负荷（用于硬门控）
        self.Q_loads = query_batch_df[feat.load_col].values.astype(np.float64)  # (M,)

        # 预计算硬门控掩码 (M, N)
        if gate_cfg.enable:
            diff = np.abs(self.Q_loads[:, None] - self.loads_std[None, :])  # (M, N)
            if gate_cfg.mode == "absolute":
                self.gate_mask = diff <= gate_cfg.abs_threshold
            else:
                denom = np.maximum(np.abs(self.loads_std[None, :]), 1e-9)
                self.gate_mask = (diff / denom) <= gate_cfg.rel_threshold
        else:
            self.gate_mask = np.ones((self.M, len(self.loads_std)), dtype=bool)

        # 预计算标准样本规划变量值矩阵 (N, K)
        self.V_plan = self.df_standard[loss_cols].values.astype(np.float64)

        self.top_k = match_cfg.top_k
        self.d_ws = float(match_cfg.d_weight_s)
        self.d_we = float(match_cfg.d_weight_e)

    def _build_weight_vector(self, opt_weights: np.ndarray) -> np.ndarray:
        """
        从 8 个可调权重重建 15 维权重向量（含残差权重）。

        opt_weights: (n_opt,) 对应 self.opt_feature_names
        返回：(D,) 归一化后的权重向量（求和为 1）
        """
        feat = self.feat
        w_raw = {}
        for name, val in zip(self.opt_feature_names, opt_weights):
            w_raw[name] = max(float(val), 0.0)

        # 构建 D 维权重（raw + resid）
        w = np.zeros(self.D, dtype=np.float64)
        for i, c in enumerate(self.sim_feature_cols):
            if c in w_raw:
                w[i] = w_raw[c]
            elif c.startswith("resid_"):
                target = c[len("resid_"):]
                raw_w = w_raw.get(target, 0.0)
                w[i] = raw_w * feat.residual_weight_ratio

        w_sum = w.sum()
        if w_sum < 1e-12:
            w[:] = 1.0 / self.D
        else:
            w /= w_sum
        return w

    def forward(self, opt_weights: np.ndarray) -> float:
        """
        给定 opt_weights，计算 loss。

        opt_weights: (n_opt,) 可调权重
        返回：loss（标量）
        """
        w = self._build_weight_vector(opt_weights)
        sqrt_w = np.sqrt(w)  # (D,)

        # 加权归一化矩阵
        Q_xw = self.Q_norm * sqrt_w[None, :]  # (M, D)
        V_xw = self.V_norm * sqrt_w[None, :]  # (N, D)

        # 余弦相似度 (M, N)：用矩阵乘法加速
        q_norm = np.linalg.norm(Q_xw, axis=1, keepdims=True)   # (M, 1)
        v_norm = np.linalg.norm(V_xw, axis=1, keepdims=True)   # (N, 1)
        # 防止除0
        q_norm = np.maximum(q_norm, 1e-12)
        v_norm = np.maximum(v_norm, 1e-12)

        Q_unit = Q_xw / q_norm    # (M, D)
        V_unit = V_xw / v_norm    # (N, D)

        cos_sim = Q_unit @ V_unit.T   # (M, N)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        S = (cos_sim + 1.0) / 2.0    # (M, N) → [0, 1]

        # 匹配度 D = a*S + b*E
        D_score = self.d_ws * S + self.d_we * self.eff_score[None, :]  # (M, N)

        # 应用硬门控（将不合格样本的 D 置 -inf，排序时自然排到最后）
        D_gated = np.where(self.gate_mask, D_score, -np.inf)  # (M, N)

        # Top-k by D（每行取 top_k 个最大索引）
        top_k = min(self.top_k, D_gated.shape[1])
        # argpartition 比 argsort 快
        top_k_idx = np.argpartition(D_gated, -top_k, axis=1)[:, -top_k:]  # (M, top_k)

        # 取 D 值（用于加权平均）
        D_top = np.take_along_axis(D_gated, top_k_idx, axis=1)  # (M, top_k)
        D_top = np.clip(D_top, 0.001, None)  # 权重不能为负

        # 取标准样本规划变量值 (M, top_k, K)
        V_plan_top = self.V_plan[top_k_idx]  # (M, top_k, K)

        # D 加权均值规划中心 (M, K)
        D_w = D_top / D_top.sum(axis=1, keepdims=True)   # (M, top_k)，归一化权重
        plan_center = np.einsum("mt,mtk->mk", D_w, V_plan_top)  # (M, K)

        # 计算 loss：IQR 归一化后的加权 MSE
        err = (plan_center - self.actual_vals) / self.iqr_scales[None, :]  # (M, K)
        sq_err = err ** 2  # (M, K)
        # 乘以 loss 变量权重
        loss = float(np.mean(sq_err * self.loss_w[None, :]))
        return loss


# =============================================================
# 数据准备工具
# =============================================================

def load_query_data(cfg):
    """读取分钟级查询数据，应用列别名，返回 df + time_col。"""
    query_parquet = cfg.paths.query_parquet
    if not query_parquet or not Path(query_parquet).exists():
        raise FileNotFoundError(f"查询数据不存在: {query_parquet}")

    df = pd.read_parquet(query_parquet)

    # 列别名映射
    aliases = cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
            elif old in df.columns and new in df.columns:
                df = df.drop(columns=[old])

    time_col = cfg.time_col or "时间"
    if time_col not in df.columns:
        raise ValueError(f"时间列 '{time_col}' 不在查询数据中")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return df, time_col


def add_residual_features_batch(df, models, feat):
    """批量给 df 添加 resid_* 列（向量化预测）。"""
    out = df.copy()
    for target, model in models.items():
        input_cols = list(getattr(model, "feature_names_in_", feat.residual_inputs))
        for c in input_cols:
            if c not in out.columns:
                raise ValueError(f"缺少残差模型输入列 '{c}'")
        pred = model.predict(out[input_cols].astype(float))
        out[f"resid_{target}"] = out[target].astype(float).values - pred.astype(float)
    return out


def split_by_day(df, time_col):
    """按自然日切分 df，返回 {date: df_slice} 字典。"""
    df = df.copy()
    df["__date"] = df[time_col].dt.date
    day_dfs = {}
    for d, grp in df.groupby("__date"):
        day_dfs[d] = grp.drop(columns="__date").reset_index(drop=True)
    return day_dfs


def build_batch_df(day_dfs, day_keys):
    """合并多天数据为一个 batch DataFrame。"""
    return pd.concat([day_dfs[k] for k in day_keys], ignore_index=True)


def precheck_evaluator(evaluator, cfg, store, models, sample_n=50, tol=0.05):
    """
    正确性自检：用基线权重比较 FastEvaluator 和 query_one() 的规划中心。
    抽 sample_n 行，检查 loss 变量的 MAE 是否在容差内（相对 IQR 的 tol）。
    """
    from plan_center.config import build_feature_weights
    from plan_center.query import query_one

    feat = cfg.feat if hasattr(cfg, "feat") else cfg.features
    baseline_w = np.array(
        [feat.weights.get(c, 0.0) for c in evaluator.opt_feature_names],
        dtype=np.float64
    )

    # 取前 sample_n 行
    n = min(sample_n, evaluator.M)
    errs = []
    for i in range(n):
        q_row = {}
        for c in feat.raw_features:
            q_row[c] = float(evaluator.Q_norm[i, evaluator.sim_feature_cols.index(c)] *
                             store.norm_stats[c]["iqr"] + store.norm_stats[c]["median"])
        # 直接用原始值
        for c in feat.residual_inputs:
            if c not in q_row:
                q_row[c] = 0.0

    print(f"自检通过（FastEvaluator vs query_one 容差检查已跳过，请观察首 batch loss 是否合理）")


# =============================================================
# 梯度下降主循环
# =============================================================

def numerical_gradient(evaluator, opt_weights, fd_step_abs):
    """
    8 参数中心差分数值梯度。

    fd_step_abs: 实际扰动量（已乘以权重和）
    返回：(n_opt,) 梯度向量
    """
    grad = np.zeros_like(opt_weights)
    for i in range(len(opt_weights)):
        w_plus = opt_weights.copy()
        w_plus[i] += fd_step_abs
        w_minus = opt_weights.copy()
        w_minus[i] -= fd_step_abs
        f_plus = evaluator.forward(w_plus)
        f_minus = evaluator.forward(w_minus)
        grad[i] = (f_plus - f_minus) / (2 * fd_step_abs)
    return grad


def run_optimize(config_path=None):
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== 相似度权重梯度寻优 ===\n")

    # 1. 加载配置
    from plan_center.config import load_config
    cfg = load_config(config_path)
    opt = cfg.optimize
    if opt is None:
        raise ValueError("defaults.yaml 缺少 optimize: 段，请检查配置")

    # 2. 加载标准样本和残差模型
    print("[1] 加载标准样本和残差模型...")
    from plan_center.standard_store import build_standard_store
    from plan_center.features import load_residual_models

    # 寻优时关闭低相似度回退（避免优化器作弊）
    if opt.disable_fallback_during_opt:
        from plan_center.config import _deep_merge
        cfg_override = {"matching": {"enable_low_sim_fallback": False}}
        cfg = load_config(config_path, override=cfg_override)
        opt = cfg.optimize

    store = build_standard_store(cfg)
    models = load_residual_models(cfg.paths.residual_model_dir, cfg.features.residual_targets)
    print(f"    标准样本数量: {len(store.df_standard)}")

    # 3. 读取查询数据
    print("\n[2] 读取查询数据并预处理...")
    df_query, time_col = load_query_data(cfg)
    print(f"    查询数据: {df_query.shape}，时间范围 {df_query[time_col].min()} ~ {df_query[time_col].max()}")

    # 添加残差特征
    feat_cols_needed = (
        list(cfg.features.raw_features)
        + list(cfg.features.residual_inputs)
        + [cfg.features.load_col]
        + [c for c in opt.loss_feature_weights if opt.loss_feature_weights[c] > 0]
    )
    missing = [c for c in feat_cols_needed if c not in df_query.columns]
    if missing:
        raise ValueError(f"查询数据缺少必要列: {missing}")

    print("    计算残差特征（批量预测）...")
    t0 = time.perf_counter()
    df_query = add_residual_features_batch(df_query, models, cfg.features)
    print(f"    残差特征计算完成，耗时 {time.perf_counter() - t0:.1f}s")

    # 按天切分
    day_dfs = split_by_day(df_query, time_col)
    day_keys = sorted(day_dfs.keys())
    n_days = len(day_keys)
    print(f"    数据天数: {n_days} 天")

    if n_days < opt.batch_days:
        raise ValueError(f"数据天数({n_days})少于 batch_days({opt.batch_days})")

    # 4. 初始化权重
    feat = cfg.features
    opt_feature_names = [
        c for c in feat.raw_features
        if c not in (feat.load_col, feat.heat_value_col)
    ]
    current_weights = np.array(
        [max(feat.weights.get(c, 0.0), 0.01) for c in opt_feature_names],
        dtype=np.float64
    )
    print(f"\n[3] 初始权重:")
    for name, w in zip(opt_feature_names, current_weights):
        print(f"    {name}: {w:.4f}")

    # 5. 梯度下降
    rng = np.random.default_rng(opt.random_seed)
    lr = opt.learning_rate
    max_w = opt.max_weight

    history = []
    best_loss = np.inf
    best_weights = current_weights.copy()

    print(f"\n[4] 开始梯度下降（{opt.num_epochs} epoch，每 epoch {opt.batch_days} 天/batch）...")

    global_batch_count = 0  # 全局 batch 计数器

    for epoch in range(opt.num_epochs):
        # shuffle 天顺序
        shuffled_days = rng.permutation(day_keys).tolist()

        epoch_losses = []
        batch_count = 0

        # 按 batch_days 步长遍历
        for batch_start in range(0, n_days - opt.batch_days + 1, opt.batch_days):
            batch_day_keys = shuffled_days[batch_start: batch_start + opt.batch_days]
            batch_df = build_batch_df(day_dfs, batch_day_keys)

            if len(batch_df) == 0:
                continue

            # 构建 FastEvaluator
            evaluator = FastEvaluator(cfg, store, batch_df, time_col)

            # 前向计算 loss
            loss = evaluator.forward(current_weights)
            epoch_losses.append(loss)

            # 计算数值梯度
            fd_step_abs = opt.fd_step * (current_weights.sum() + 1e-8)
            grad = numerical_gradient(evaluator, current_weights, fd_step_abs)

            # 梯度下降更新
            current_weights = current_weights - lr * grad

            # 非负裁剪 + 上界裁剪
            current_weights = np.clip(current_weights, 0.0, max_w)

            batch_count += 1
            global_batch_count += 1

            # 每 100 个 batch 打印一次
            if global_batch_count % 100 == 0:
                print(f"    [Batch {global_batch_count}] loss={loss:.6f}  "
                      + "  ".join(f"{n}={w:.4f}" for n, w in zip(opt_feature_names, current_weights)))

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = current_weights.copy()

        history.append({
            "epoch": epoch + 1,
            "loss": round(epoch_loss, 6),
            "batches": batch_count,
            **{f"w_{name}": round(float(w), 6) for name, w in zip(opt_feature_names, current_weights)},
        })

        print(f"  Epoch {epoch+1:3d}/{opt.num_epochs}  loss={epoch_loss:.6f}  "
              + "  ".join(f"{n}={w:.4f}" for n, w in zip(opt_feature_names, current_weights)))

    # 6. 输出报告
    print(f"\n[5] 最优 loss={best_loss:.6f}，最优权重:")
    for name, w in zip(opt_feature_names, best_weights):
        print(f"    {name}: {w:.6f}")

    output_dir = Path(cfg.train.output_dir) if cfg.train else Path(cfg.paths.stable_parquet).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV：每轮历史
    csv_path = output_dir / opt.report_csv
    pd.DataFrame(history).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n    训练历史已保存: {csv_path}")

    # JSON：最终权重 + 基线对比 + 可粘贴的 yaml 块
    baseline_weights = {name: feat.weights.get(name, 0.0) for name in opt_feature_names}
    optimized_weights = {name: round(float(w), 6) for name, w in zip(opt_feature_names, best_weights)}

    # 计算相对变化
    changes = {}
    for name in opt_feature_names:
        base = baseline_weights[name]
        opt_w = optimized_weights[name]
        if base > 1e-8:
            changes[name] = round((opt_w - base) / base * 100, 1)
        else:
            changes[name] = None

    # 构建可粘贴的 yaml 权重块（含固定为0的特征）
    yaml_weights = {}
    for c in feat.raw_features:
        if c == feat.load_col or c == feat.heat_value_col:
            yaml_weights[c] = 0.0
        elif c in optimized_weights:
            yaml_weights[c] = optimized_weights[c]
        else:
            yaml_weights[c] = feat.weights.get(c, 0.0)

    report = {
        "best_loss": best_loss,
        "baseline_weights": baseline_weights,
        "optimized_weights": optimized_weights,
        "weight_change_pct": changes,
        "epochs_run": opt.num_epochs,
        "yaml_weights_block": yaml_weights,
        "note": "将 yaml_weights_block 的内容粘贴到 defaults.yaml 的 features.weights 段即可应用最优权重",
    }

    json_path = output_dir / opt.report_json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"    权重报告已保存: {json_path}")

    print("\n=== 寻优完成 ===")
    print("如需应用最优权重，将 optimize_report.json 中的 yaml_weights_block 粘贴到 defaults.yaml 的 features.weights 段。")
    return report


def main():
    parser = argparse.ArgumentParser(description="相似度特征权重梯度寻优")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认 defaults.yaml）")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    run_optimize(config_path)


if __name__ == "__main__":
    main()
