# -*- coding: utf-8 -*-
"""
train_residual.py — 残差特征训练模块

从稳定工况原始数据训练残差模型，生成向量数据库。

流程：
1. 读取稳定工况数据
2. 合理工况筛选（2~98% 分位数，可开关）
3. 分层抽样（可开关）
4. 训练残差模型（HistGradientBoostingRegressor）
5. 5-fold OOF 生成残差
6. 计算加权协方差逆矩阵（马氏距离用）
7. 保存输出（向量数据库、模型、协方差矩阵、报告）
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import PlanningConfig, load_config, build_feature_weights
from .similarity import compute_covariance_matrix, weighted_cov_inv_matrix, weight_array


# =========================
# 数据加载
# =========================

def load_stable_data(
    parquet_path: str,
    column_aliases: dict[str, str] | None = None,
    last_month_exclusion_parquet: str | None = None,
    date_col: str = "日期",
) -> pd.DataFrame:
    """读取稳定工况数据，可选排除与另一数据集最后一个月重叠的样本。"""
    df = pd.read_parquet(parquet_path)

    # 应用列别名映射
    if column_aliases:
        for old, new in column_aliases.items():
            if old in df.columns:
                if new in df.columns:
                    df = df.drop(columns=[old])
                    print(f"目标列 '{new}' 已存在，已删除源列 '{old}'")
                else:
                    df = df.rename(columns={old: new})
                    print(f"已应用列别名映射: {old} → {new}")

    # 排除最后一个月（防止数据泄露）
    if last_month_exclusion_parquet:
        _exclude_last_month(df, last_month_exclusion_parquet, date_col)

    print(f"读取稳定工况数据: {df.shape}")
    return df


def _exclude_last_month(
    df: pd.DataFrame,
    reference_parquet: str,
    date_col: str = "日期",
) -> None:
    """
    根据参考数据集（如 #4_df_all_1min.parquet）的时间范围，
    剔除稳定工况数据中处于参考数据最后一个月内的样本。

    通过确定参考数据的最大时间并向前推一个月作为截止时间，
    然后筛选掉稳定工况数据中日期在此截止时间之后的样本。

    参数：
        df: 稳定工况 DataFrame（原地修改）
        reference_parquet: 参考数据 parquet 路径
        date_col: 稳定工况数据中的日期列名
    """
    if date_col not in df.columns:
        print(f"警告: 稳定工况数据中未找到日期列 '{date_col}'，跳过最后一个月排除")
        return

    # 读取参考数据的时间范围
    ref_df = pd.read_parquet(reference_parquet)
    time_col = "时间" if "时间" in ref_df.columns else None
    if time_col is None:
        print(f"警告: 参考数据中未找到 '时间' 列，跳过最后一个月排除")
        return

    ref_df[time_col] = pd.to_datetime(ref_df[time_col])
    max_time = ref_df[time_col].max()
    cutoff = max_time - pd.DateOffset(months=1)

    # 确保日期列为 datetime 类型
    df[date_col] = pd.to_datetime(df[date_col])

    before = len(df)
    mask = df[date_col] < cutoff
    df.drop(index=df.index[~mask], inplace=True)
    df.reset_index(drop=True, inplace=True)
    after = len(df)

    print(f"最后一个月排除: 参考数据截止时间 {max_time}，剔除 >= {cutoff} 的样本")
    print(f"  {before} → {after}，排除 {before - after} 行（{ (before - after) / max(before, 1) * 100:.2f}%）")


def filter_by_cutoff_date(
    df: pd.DataFrame,
    cutoff_date: str,
    date_col: str = "日期",
) -> pd.DataFrame:
    """
    按截止日期过滤稳定工况数据，只保留 < cutoff_date 的样本。

    参数：
        df: 稳定工况 DataFrame
        cutoff_date: 截止日期字符串（如 '2025-05-11'），保留此日期之前的样本
        date_col: 日期列名

    返回：
        过滤后的 DataFrame
    """
    if date_col not in df.columns:
        print(f"警告: 稳定工况数据中未找到日期列 '{date_col}'，跳过截止日期过滤")
        return df

    cutoff = pd.Timestamp(cutoff_date)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    before = len(df)
    df = df[df[date_col] < cutoff].reset_index(drop=True)
    after = len(df)

    print(f"截止日期过滤: 只保留 < {cutoff_date} 的样本")
    print(f"  {before} → {after}，排除 {before - after} 行（{ (before - after) / max(before, 1) * 100:.2f}%）")
    return df


# =========================
# 合理工况筛选
# =========================

def filter_reasonable_conditions(
    df: pd.DataFrame,
    load_col: str = "主汽流量",
    filter_cols: list[str] = None,
    n_bins: int = 15,
    q_low: float = 0.02,
    q_high: float = 0.98,
    max_bad_features: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    合理工况筛选：按负荷分箱，2~98% 分位数。
    每个样本允许最多 max_bad_features 个特征超出范围。

    返回:
        (筛选后数据, 分位数表)
    """
    if filter_cols is None:
        filter_cols = ["炉膛差压", "一次风流量", "床温", "料层差压", "锅炉出口氧量"]

    df = df.copy()
    df["_load_bin"] = pd.qcut(df[load_col], q=n_bins, duplicates="drop")

    keep_parts = []
    records = []

    for bin_name, g in df.groupby("_load_bin", observed=True):
        low = g[filter_cols].quantile(q_low)
        high = g[filter_cols].quantile(q_high)
        bad_count = ((g[filter_cols] < low) | (g[filter_cols] > high)).sum(axis=1)
        keep_parts.append(bad_count <= max_bad_features)

        for c in filter_cols:
            records.append({
                "负荷箱体": str(bin_name),
                "特征": c,
                f"{q_low*100:.0f}%分位数": float(low[c]),
                f"{q_high*100:.0f}%分位数": float(high[c]),
                "箱体样本数": int(len(g)),
            })

    keep_flag = pd.concat(keep_parts).sort_index()
    filtered = df.loc[keep_flag].drop(columns=["_load_bin"]).reset_index(drop=True)
    q_table = pd.DataFrame(records)

    print(f"合理工况筛选: {len(df)} → {len(filtered)}，保留比例 {len(filtered) / max(len(df), 1):.3f}")
    return filtered, q_table


# =========================
# 分层抽样
# =========================

def add_quantile_strata(
    data: pd.DataFrame,
    bin_config: dict,
    strata_col: str = "__strata",
) -> tuple[pd.DataFrame, list[str]]:
    """
    按多个连续变量做分位数分箱，并把箱号组合成分层标签。
    例如：负荷10箱 × 压力5箱 × 吨煤产汽平均值5箱。
    """
    data = data.copy()
    bin_cols = []

    for col, q in bin_config.items():
        if col not in data.columns:
            raise ValueError(f"分箱列不存在：{col}")

        bin_col = f"__bin_{col}"
        data[bin_col] = pd.qcut(
            data[col],
            q=q,
            labels=False,
            duplicates="drop"
        ).astype("Int64")

        bin_cols.append(bin_col)

    data[strata_col] = data[bin_cols].astype(str).agg("|".join, axis=1)
    return data, bin_cols


def proportional_split_by_strata(
    data: pd.DataFrame,
    strata_col: str = "__strata",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    在每个分位数组合箱内部按比例抽样。
    约束：每个非空箱至少保留1个样本在训练集，保证模型见过所有工况箱。
    """
    rng = np.random.default_rng(random_state)
    split = pd.Series(index=data.index, dtype=object)

    records = []

    for strata, g in data.groupby(strata_col, observed=True):
        idx = np.array(g.index)
        rng.shuffle(idx)
        n = len(idx)

        if n == 1:
            train_idx = idx
            valid_idx = np.array([], dtype=int)
            test_idx = np.array([], dtype=int)
        elif n == 2:
            train_idx = idx[:1]
            valid_idx = np.array([], dtype=int)
            test_idx = idx[1:]
        elif n == 3:
            train_idx = idx[:1]
            valid_idx = idx[1:2]
            test_idx = idx[2:]
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_valid = max(1, int(round(n * valid_ratio)))
            n_test = n - n_train - n_valid

            if n_test <= 0:
                n_test = 1
                n_train = max(1, n - n_valid - n_test)

            if n_train + n_valid + n_test > n:
                n_train = n - n_valid - n_test

            train_idx = idx[:n_train]
            valid_idx = idx[n_train:n_train + n_valid]
            test_idx = idx[n_train + n_valid:]

        split.loc[train_idx] = "train"
        split.loc[valid_idx] = "valid"
        split.loc[test_idx] = "test"

        records.append({
            "分层箱": strata,
            "样本数": n,
            "train": len(train_idx),
            "valid": len(valid_idx),
            "test": len(test_idx),
        })

    return split, pd.DataFrame(records)


def make_stratified_oof_folds(
    data: pd.DataFrame,
    strata_col: str = "__strata",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """
    构造 OOF 折号。
    对于样本数>=2的分层箱，尽量把样本分到不同折；
    对于单样本箱，无法无泄漏 OOF，标记为 -1，后面用全量最终模型预测。
    """
    rng = np.random.default_rng(random_state)
    fold_id = pd.Series(-1, index=data.index, dtype=int)

    for strata, g in data.groupby(strata_col, observed=True):
        idx = np.array(g.index)
        rng.shuffle(idx)
        n = len(idx)

        if n <= 1:
            continue

        for k, row_idx in enumerate(idx):
            fold_id.loc[row_idx] = k % n_splits

    return fold_id


# =========================
# 残差模型训练
# =========================

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def eval_regression(y_true, y_pred, prefix):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return {
            f"{prefix}_样本数": 0,
            f"{prefix}_R2": np.nan,
            f"{prefix}_MAE": np.nan,
            f"{prefix}_RMSE": np.nan,
            f"{prefix}_MAPE": np.nan,
        }

    return {
        f"{prefix}_样本数": int(len(y_true)),
        f"{prefix}_R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_MAPE": safe_mape(y_true, y_pred),
    }


def train_residual_models(
    df: pd.DataFrame,
    input_cols: list[str],
    target_cols: list[str],
    split_col: str,
    model_params: dict,
    oof_n_splits: int = 5,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    训练残差模型（每个目标一个 HistGradientBoostingRegressor）。

    返回:
        models: {target: model}
        model_report: 训练报告（R², MAE, RMSE, MAPE）
        residual_report: 残差统计报告
    """
    # 构建基础模型
    base_model = HistGradientBoostingRegressor(**model_params)

    oof_fold_id = make_stratified_oof_folds(
        data=df,
        strata_col="__strata",
        n_splits=oof_n_splits,
        random_state=42,
    )

    df["__oof_fold"] = oof_fold_id

    train_idx = df.index[df[split_col] == "train"]
    valid_idx = df.index[df[split_col] == "valid"]
    test_idx = df.index[df[split_col] == "test"]

    model_report_rows = []
    residual_report_rows = []
    models = {}

    for target in target_cols:
        pred_col = f"pred_{target}"
        resid_col = f"resid_{target}"

        print(f"\n训练残差模型: {target}")

        # 1) 固定 train/valid/test 评估
        eval_model = clone(base_model)
        eval_model.fit(
            df.loc[train_idx, input_cols],
            df.loc[train_idx, target],
        )

        report = {"target": target}
        for split_name, idx in [
            ("train", train_idx),
            ("valid", valid_idx),
            ("test", test_idx),
        ]:
            if len(idx) > 0:
                y_true = df.loc[idx, target]
                y_pred = eval_model.predict(df.loc[idx, input_cols])
                report.update(eval_regression(y_true, y_pred, split_name))
            else:
                report.update(eval_regression([], [], split_name))

        # 2) OOF 预测，用于生成全样本残差特征
        oof_pred = pd.Series(np.nan, index=df.index, dtype=float)

        for fold in range(oof_n_splits):
            hold_idx = df.index[df["__oof_fold"] == fold]

            if len(hold_idx) == 0:
                continue

            fit_idx = df.index[
                (df["__oof_fold"] != fold)
                | (df["__oof_fold"] == -1)
            ]

            fold_model = clone(base_model)
            fold_model.fit(
                df.loc[fit_idx, input_cols],
                df.loc[fit_idx, target],
            )

            oof_pred.loc[hold_idx] = fold_model.predict(
                df.loc[hold_idx, input_cols]
            )

        # 对单样本分层箱，无法生成严格 OOF，使用全量模型预测并标记
        full_model = clone(base_model)
        full_model.fit(df[input_cols], df[target])

        need_full_pred_idx = oof_pred.index[oof_pred.isna()]
        if len(need_full_pred_idx) > 0:
            oof_pred.loc[need_full_pred_idx] = full_model.predict(
                df.loc[need_full_pred_idx, input_cols]
            )

        df[pred_col] = oof_pred
        df[resid_col] = df[target] - df[pred_col]

        report["oof_样本数"] = int(df[pred_col].notna().sum())
        report["oof_R2"] = float(r2_score(df[target], df[pred_col]))
        report["oof_MAE"] = float(mean_absolute_error(df[target], df[pred_col]))
        report["oof_RMSE"] = float(np.sqrt(mean_squared_error(df[target], df[pred_col])))
        report["oof_MAPE"] = safe_mape(df[target], df[pred_col])
        report["full_model补预测样本数"] = int(len(need_full_pred_idx))

        model_report_rows.append(report)

        r = df[resid_col].astype(float)
        q25 = r.quantile(0.25)
        q75 = r.quantile(0.75)

        residual_report_rows.append({
            "target": target,
            "residual_col": resid_col,
            "残差均值": float(r.mean()),
            "残差中位数": float(r.median()),
            "残差标准差": float(r.std(ddof=0)),
            "残差IQR": float(q75 - q25),
            "残差绝对值均值": float(r.abs().mean()),
            "残差绝对值90%分位": float(r.abs().quantile(0.90)),
        })

        models[target] = full_model

        print(f"  OOF R²: {report['oof_R2']:.4f}, OOF MAE: {report['oof_MAE']:.4f}")

    model_report = pd.DataFrame(model_report_rows)
    residual_report = pd.DataFrame(residual_report_rows)

    return models, model_report, residual_report


# =========================
# 加权协方差逆矩阵计算
# =========================

def compute_and_save_covariance(
    df: pd.DataFrame,
    sim_feature_cols: list[str],
    weights_dict: dict[str, float],
    output_path: Path,
    reg_lambda: float = 1e-6,
) -> np.ndarray:
    """
    计算加权协方差逆矩阵 M = W^{1/2}·Σ⁻¹·W^{1/2} 并保存到 JSON。

    参数：
        df: 含 sim_feature_cols 列的 DataFrame
        sim_feature_cols: 相似度特征列名（raw + residual）
        weights_dict: 特征权重字典
        output_path: 输出 JSON 路径
        reg_lambda: 协方差矩阵正则化系数

    返回：
        (D, D) 加权协方差逆矩阵 M
    """
    X = df[sim_feature_cols].values.astype(np.float64)
    cov = compute_covariance_matrix(X, reg_lambda=reg_lambda)
    w = weight_array(sim_feature_cols, weights_dict)
    M = weighted_cov_inv_matrix(cov, w)

    cov_data = {
        "cov_inv_matrix": M.tolist(),
        "feature_cols": sim_feature_cols,
        "reg_lambda": reg_lambda,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cov_data, f, ensure_ascii=False, indent=2)

    print(f"已保存协方差逆矩阵: {output_path}，形状 {M.shape}")
    return M


# =========================
# 保存输出
# =========================

def save_outputs(
    df_vector_db: pd.DataFrame,
    models: dict,
    model_report: pd.DataFrame,
    residual_report: pd.DataFrame,
    output_dir: str,
) -> dict:
    """
    保存输出：
    - vector_db.parquet
    - residual_model_*.joblib（6个）
    - residual_report.csv
    - model_report.csv

    注意：协方差逆矩阵由 compute_and_save_covariance 独立保存。

    返回输出路径字典。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. 向量数据库
    vector_db_path = output_dir / "vector_db.parquet"
    df_vector_db.to_parquet(vector_db_path, index=False)
    paths["vector_db"] = str(vector_db_path)
    print(f"已保存向量数据库: {vector_db_path}")

    # 2. 残差模型
    model_dir = output_dir / "residual_models"
    model_dir.mkdir(exist_ok=True)
    for target, model in models.items():
        model_path = model_dir / f"residual_model_{target}.joblib"
        joblib.dump(model, model_path)
    paths["model_dir"] = str(model_dir)
    print(f"已保存残差模型: {model_dir}")

    # 3. 训练报告
    model_report_path = output_dir / "model_report.csv"
    model_report.to_csv(model_report_path, index=False, encoding="utf-8-sig")
    paths["model_report"] = str(model_report_path)

    residual_report_path = output_dir / "residual_report.csv"
    residual_report.to_csv(residual_report_path, index=False, encoding="utf-8-sig")
    paths["residual_report"] = str(residual_report_path)
    print(f"已保存训练报告: {output_dir}")

    return paths


# =========================
# 主流程
# =========================

def main():
    """主流程：读取 → 筛选 → 训练 → 计算残差 → 归一化 → 保存。"""
    parser = argparse.ArgumentParser(description="残差特征训练：生成向量数据库 + 协方差矩阵")
    parser.add_argument(
        "--cutoff-date", type=str, default=None,
        help="截止日期（如 '2025-05-11'），只保留此日期之前的稳定工况样本。不传则不过滤",
    )
    args = parser.parse_args()

    print("=== 残差特征训练 ===\n")

    # 加载配置
    cfg = load_config()

    if cfg.train is None:
        print("错误: 未配置 train 参数")
        return

    train_cfg = cfg.train

    # 1. 读取稳定工况数据
    print("[1] 读取稳定工况数据...")
    df = load_stable_data(
        train_cfg.input_parquet,
        cfg.features.column_aliases,
        last_month_exclusion_parquet=cfg.paths.query_parquet,
        date_col=cfg.features.date_col if hasattr(cfg.features, "date_col") else "日期",
    )

    # 1.5 截止日期过滤（可选）
    if args.cutoff_date:
        print("\n[1.5] 截止日期过滤...")
        df = filter_by_cutoff_date(
            df,
            cutoff_date=args.cutoff_date,
            date_col=cfg.features.date_col if hasattr(cfg.features, "date_col") else "日期",
        )

    # 2. 合理工况筛选
    if train_cfg.enable_filter:
        print("\n[2] 合理工况筛选...")
        df, q_table = filter_reasonable_conditions(
            df,
            load_col=cfg.features.load_col,
            filter_cols=train_cfg.filter_cols,
            n_bins=train_cfg.filter_n_bins,
            q_low=train_cfg.filter_q_low,
            q_high=train_cfg.filter_q_high,
            max_bad_features=train_cfg.filter_max_bad_features,
        )
    else:
        print("\n[2] 跳过合理工况筛选（已禁用）")

    # 3. 分层抽样
    if train_cfg.enable_stratified_split:
        print("\n[3] 分层抽样...")
        bin_config = {
            cfg.features.load_col: 10,
            cfg.features.residual_inputs[1]: 5,  # 主汽压力
            cfg.features.residual_inputs[2]: 5,  # 吨煤产气量
        }
        df, bin_cols = add_quantile_strata(df, bin_config)
        split_series, split_table = proportional_split_by_strata(
            df,
            train_ratio=train_cfg.train_ratio,
            valid_ratio=train_cfg.valid_ratio,
            test_ratio=train_cfg.test_ratio,
        )
        df["__split"] = split_series
        print(f"分层箱数量: {df['__strata'].nunique()}")
        print(f"train/valid/test: {df['__split'].value_counts().to_dict()}")
    else:
        print("\n[3] 跳过分层抽样（已禁用）")
        df["__split"] = "train"

    # 4. 训练残差模型
    print("\n[4] 训练残差模型...")
    models, model_report, residual_report = train_residual_models(
        df,
        input_cols=cfg.features.residual_inputs,
        target_cols=cfg.features.residual_targets,
        split_col="__split",
        model_params=train_cfg.residual_model_params,
        oof_n_splits=train_cfg.oof_n_splits,
    )

    print("\n模型训练报告:")
    print(model_report[["target", "train_R2", "valid_R2", "test_R2", "oof_R2"]])

    # 5. 计算加权协方差逆矩阵（对所有相似度特征：原始特征 + 残差特征）
    print("\n[5] 计算加权协方差逆矩阵...")
    residual_feat_cols = [f"resid_{t}" for t in cfg.features.residual_targets]
    all_sim_feature_cols = cfg.features.raw_features + residual_feat_cols
    # 只计算数据中实际存在的特征
    all_sim_feature_cols = [c for c in all_sim_feature_cols if c in df.columns]
    weights = build_feature_weights(cfg.features)
    covariance_path = Path(train_cfg.output_dir) / "covariance.json"
    compute_and_save_covariance(
        df, all_sim_feature_cols, weights, covariance_path
    )
    print(f"相似度特征数: {len(all_sim_feature_cols)}（原始特征 + 残差特征）")

    # 6. 构建向量数据库
    print("\n[6] 构建向量数据库...")
    # 保留原始特征 + 残差特征 + 效率 + 身份列 + 负荷列（硬门控）
    keep_cols = list(dict.fromkeys(
        ["稳定工况ID", "稳定窗口时间范围", "来源片段ID"]
        + [cfg.features.load_col]    # 主汽流量（硬门控）
        + cfg.features.raw_features
        + [f"resid_{t}" for t in cfg.features.residual_targets]
        + [cfg.features.eff_col]
    ))
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_vector_db = df[keep_cols].copy()
    print(f"向量数据库形状: {df_vector_db.shape}")

    # 7. 保存输出
    print("\n[7] 保存输出...")
    paths = save_outputs(
        df_vector_db,
        models,
        model_report,
        residual_report,
        train_cfg.output_dir,
    )

    print("\n=== 训练完成 ===")
    print(f"向量数据库: {paths['vector_db']}")
    print(f"残差模型: {paths['model_dir']}")
    print(f"协方差逆矩阵: {covariance_path}")


if __name__ == "__main__":
    main()
