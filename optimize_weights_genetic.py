# -*- coding: utf-8 -*-
"""
optimize_weights_genetic.py — 遗传进化算法相似度特征权重寻优

目标：最小化「规划中心输出」与「查询行实际控制值」之间的加权 MSE（IQR 归一化）。

被优化参数：8 个原始特征权重（主汽流量=0、热值=0 固定不动）。
残差特征权重绑定为 对应原始权重 × residual_weight_ratio。

遗传算子：
    - 选择：锦标赛选择
    - 交叉：模拟二进制交叉（SBX）+ 均匀交叉
    - 变异：高斯变异
    - 精英保留：每代最优 N 个个体直接进入下一代

Loss 计算：复用 FastEvaluator（同 optimize_weights.py）

用法:
    python -m plan_center.optimize_weights_genetic
    python -m plan_center.optimize_weights_genetic --config defaults.yaml
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
# 遗传算法算子
# =============================================================

class GeneticOperators:
    """遗传算法算子集合（纯函数，无状态）。"""

    @staticmethod
    def initialize_population(pop_size: int, n_opt: int, min_w: float, max_w: float, rng: np.random.Generator) -> np.ndarray:
        """
        均匀随机初始化种群。

        返回：(pop_size, n_opt) 数组
        """
        return rng.uniform(min_w, max_w, size=(pop_size, n_opt)).astype(np.float64)

    @staticmethod
    def tournament_select(
        population: np.ndarray,
        fitnesses: np.ndarray,
        tournament_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        锦标赛选择：随机选 k 个个体，取 fitness 最优（最小 loss）的个体。

        参数：
            population: (pop_size, n_opt)
            fitnesses: (pop_size,) 越小越好
            tournament_size: k
            rng: 随机数生成器

        返回：(n_opt,) 选中的个体
        """
        pop_size = len(population)
        k = min(tournament_size, pop_size)
        indices = rng.choice(pop_size, size=k, replace=False)
        best_idx = indices[np.argmin(fitnesses[indices])]
        return population[best_idx].copy()

    @staticmethod
    def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """
        模拟二进制交叉（Simulated Binary Crossover, SBX）。

        参数：
            p1, p2: (n_opt,) 父代
            eta: 分布指数（越大子代越接近父代，默认 15）
            rng: 随机数生成器

        返回：(c1, c2) 两个子代
        """
        n = len(p1)
        c1 = p1.copy()
        c2 = p2.copy()

        for i in range(n):
            if rng.random() < 0.5:
                u = rng.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

                c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])

        return c1, c2

    @staticmethod
    def uniform_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """
        均匀交叉：每个基因独立以 50% 概率从 p1 或 p2 继承。

        返回：(c1, c2) 两个子代
        """
        mask = rng.random(len(p1)) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1.copy(), c2.copy()

    @staticmethod
    def gaussian_mutation(
        individual: np.ndarray,
        mutation_rate: float,
        mutation_scale: float,
        weight_range: float,
        min_w: float,
        max_w: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        高斯变异：每个基因以 mutation_rate 概率添加高斯噪声。

        参数：
            individual: (n_opt,) 个体
            mutation_rate: 每个基因的变异概率
            mutation_scale: 变异幅度（weight_range 的比例）
            weight_range: 权重范围（max_w - min_w）
            min_w, max_w: 裁剪边界
            rng: 随机数生成器

        返回：变异后的个体
        """
        result = individual.copy()
        std = weight_range * mutation_scale
        mask = rng.random(len(individual)) < mutation_rate
        result[mask] += rng.normal(0.0, std, size=mask.sum())
        return np.clip(result, min_w, max_w)


# =============================================================
# 评估器（复用 optimize_weights.py 的 StandardCache + FastEvaluator）
# =============================================================

def _build_evaluators(cfg, store, df_query, time_col, std_cache):
    """
    根据 batch_days 构建 FastEvaluator 列表（每个 batch 一个 evaluator）。

    参数：
        cfg: PlanningConfig
        store: StandardStore
        df_query: 已含 resid_* 列的查询数据
        time_col: 时间列名
        std_cache: StandardCache

    返回：list[FastEvaluator]
    """
    # 丢弃 NaN 行
    from plan_center.optimize_weights import split_by_day
    residual_cols = [f"resid_{t}" for t in cfg.features.residual_targets]
    nan_check_cols = list(dict.fromkeys(
        list(cfg.features.raw_features) + residual_cols + [cfg.features.load_col]
    ))
    df_query = df_query.dropna(subset=nan_check_cols).reset_index(drop=True)

    # 降采样
    if cfg.optimize_genetic.query_stride > 1:
        df_query = df_query.iloc[::cfg.optimize_genetic.query_stride].reset_index(drop=True)

    # 按天切分
    day_dfs = split_by_day(df_query, time_col)
    day_keys = sorted(day_dfs.keys())
    n_days = len(day_keys)

    if n_days < cfg.optimize_genetic.batch_days:
        raise ValueError(f"数据天数({n_days})少于 batch_days({cfg.optimize_genetic.batch_days})")

    from plan_center.optimize_weights import FastEvaluator

    evaluators = []
    for batch_start in range(0, n_days - cfg.optimize_genetic.batch_days + 1, cfg.optimize_genetic.batch_days):
        batch_day_keys = day_keys[batch_start: batch_start + cfg.optimize_genetic.batch_days]
        batch_df = pd.concat([day_dfs[k] for k in batch_day_keys], ignore_index=True)
        evaluator = FastEvaluator(cfg, std_cache, store, batch_df, time_col)
        evaluators.append(evaluator)

    return evaluators


# =============================================================
# 遗传算法主类
# =============================================================

class GeneticOptimizer:
    """遗传进化算法权重寻优器。"""

    def __init__(self, cfg, std_cache, store, evaluators: list):
        """
        参数：
            cfg: PlanningConfig
            std_cache: StandardCache
            store: StandardStore
            evaluators: list[FastEvaluator] 评估器列表
        """
        self.cfg = cfg
        self.std_cache = std_cache
        self.store = store
        self.evaluators = evaluators

        opt_gen = cfg.optimize_genetic
        feat = cfg.features

        self.pop_size = opt_gen.population_size
        self.n_generations = opt_gen.n_generations
        self.elite_size = opt_gen.elite_size
        self.tournament_size = opt_gen.tournament_size
        self.crossover_rate = opt_gen.crossover_rate
        self.crossover_method = opt_gen.crossover_method
        self.mutation_rate = opt_gen.mutation_rate
        self.mutation_scale = opt_gen.mutation_scale
        self.min_w = opt_gen.min_weight
        self.max_w = opt_gen.max_weight
        self.sbx_eta = opt_gen.sbx_eta

        # 可优化特征名（raw_features 中排除 load_col 和 heat_value_col）
        self.opt_feature_names = [
            c for c in feat.raw_features
            if c not in (feat.load_col, feat.heat_value_col)
        ]
        self.n_opt = len(self.opt_feature_names)

        self.rng = np.random.default_rng(42)

        # 权重范围
        self.weight_range = self.max_w - self.min_w

    def _evaluate_individual(self, weights: np.ndarray) -> float:
        """
        评估单个个体的 fitness（batch 平均 loss）。

        参数：
            weights: (n_opt,) 权重向量

        返回：loss 标量（越小越好）
        """
        losses = []
        for ev in self.evaluators:
            loss = ev.forward(weights)
            if np.isfinite(loss):
                losses.append(loss)
        if not losses:
            return np.inf
        return float(np.mean(losses))

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        批量评估种群（向量化）。

        参数：
            population: (pop_size, n_opt)

        返回：(pop_size,) fitness 数组
        """
        pop_size = len(population)
        fitnesses = np.full(pop_size, np.inf, dtype=np.float64)

        for i in range(pop_size):
            fitnesses[i] = self._evaluate_individual(population[i])

        return fitnesses

    def _select_parents(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """
        锦标赛选择，返回一个父代。

        返回：(n_opt,) 父代权重
        """
        return GeneticOperators.tournament_select(
            population, fitnesses, self.tournament_size, self.rng
        )

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """交叉两个父代，返回两个子代。"""
        if self.rng.random() < self.crossover_rate:
            if self.crossover_method == "blend":
                return GeneticOperators.sbx_crossover(p1, p2, self.sbx_eta, self.rng)
            elif self.crossover_method == "uniform":
                return GeneticOperators.uniform_crossover(p1, p2, self.rng)
            else:
                raise ValueError(f"未知 crossover_method: {self.crossover_method}")
        return p1.copy(), p2.copy()

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """变异单个个体。"""
        return GeneticOperators.gaussian_mutation(
            individual, self.mutation_rate, self.mutation_scale,
            self.weight_range, self.min_w, self.max_w, self.rng
        )

    def _evolve_one_generation(self, population: np.ndarray, fitnesses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        进化一代。

        流程：
            1. 精英保留
            2. 选择 + 交叉 + 变异 生成子代
            3. 评估子代 fitness

        返回：(new_population, new_fitnesses)
        """
        pop_size = len(population)
        n_offspring = pop_size - self.elite_size

        # 1. 精英保留（按 fitness 排序，取前 elite_size）
        elite_order = np.argsort(fitnesses)[:self.elite_size]
        new_pop = population[elite_order].copy()
        new_fitnesses = fitnesses[elite_order].copy()

        # 2. 生成子代
        offspring = []
        while len(offspring) < n_offspring:
            # 选择
            p1 = self._select_parents(population, fitnesses)
            p2 = self._select_parents(population, fitnesses)

            # 交叉
            c1, c2 = self._crossover(p1, p2)

            # 变异
            c1 = self._mutate(c1)
            c2 = self._mutate(c2)

            offspring.append(c1)
            if len(offspring) < n_offspring:
                offspring.append(c2)

        offspring = np.array(offspring[:n_offspring])

        # 3. 评估子代
        offspring_fitnesses = self._evaluate_population(offspring)

        # 合并
        new_pop = np.vstack([new_pop, offspring])
        new_fitnesses = np.concatenate([new_fitnesses, offspring_fitnesses])

        return new_pop, new_fitnesses

    def run(self) -> dict:
        """
        运行遗传算法寻优。

        返回：报告字典
        """
        print(f"=== 遗传进化算法权重寻优 ===\n")
        print(f"种群大小: {self.pop_size}")
        print(f"迭代代数: {self.n_generations}")
        print(f"精英保留: {self.elite_size}")
        print(f"可优化特征: {self.opt_feature_names} (共 {self.n_opt} 个)\n")

        # 初始化种群
        print("[1] 初始化种群...")
        population = GeneticOperators.initialize_population(
            self.pop_size, self.n_opt, self.min_w, self.max_w, self.rng
        )

        # 评估初始种群
        print("[2] 评估初始种群...")
        fitnesses = self._evaluate_population(population)

        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_individual = population[best_idx].copy()

        # 打印初始最佳
        print(f"    初始最佳 loss: {best_fitness:.6f}")
        print(f"    初始最佳权重: " + "  ".join(
            f"{n}={w:.4f}" for n, w in zip(self.opt_feature_names, best_individual)
        ) + "\n")

        # 进化主循环
        print(f"[3] 开始进化（{self.n_generations} 代）...\n")
        history = []
        start_time = time.perf_counter()

        for gen in range(self.n_generations):
            gen_start = time.perf_counter()

            # 进化一代
            population, fitnesses = self._evolve_one_generation(population, fitnesses)

            # 记录统计
            gen_best_idx = np.argmin(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_mean_fitness = float(np.mean(fitnesses[np.isfinite(fitnesses)]))

            # 更新全局最优
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()

            gen_time = time.perf_counter() - gen_start

            history.append({
                "generation": gen + 1,
                "best_loss": round(float(best_fitness), 4),
                "gen_best_loss": round(float(gen_best_fitness), 4),
                "mean_loss": round(gen_mean_fitness, 4),
                "time_sec": round(gen_time, 2),
                **{f"w_{name}": round(float(w), 2)
                   for name, w in zip(self.opt_feature_names, population[gen_best_idx])},
            })

            print(f"  Gen {gen+1:3d}/{self.n_generations}  "
                  f"best={best_fitness:.4f}  "
                  f"gen_best={gen_best_fitness:.4f}  "
                  f"mean={gen_mean_fitness:.4f}  "
                  f"time={gen_time:.1f}s  "
                  + "  ".join(f"{n}={w:.2f}"
                              for n, w in zip(self.opt_feature_names, population[gen_best_idx])))

        total_time = time.perf_counter() - start_time
        print(f"\n进化完成，总耗时: {total_time:.1f}s")
        print(f"最优 loss: {best_fitness:.4f}")
        print(f"最优权重: " + "  ".join(
            f"{n}={w:.2f}" for n, w in zip(self.opt_feature_names, best_individual)
        ))

        # 输出报告
        return self._save_report(best_individual, best_fitness, history, total_time)

    def _save_report(self, best_weights: np.ndarray, best_loss: float, history: list, total_time: float) -> dict:
        """保存报告文件。"""
        feat = self.cfg.features

        # 构建最终权重（含残差权重绑定）
        raw_weights = {}
        for name, val in zip(self.opt_feature_names, best_weights):
            raw_weights[name] = max(float(val), 0.0)

        optimized_weights = {}
        for c in feat.raw_features:
            if c == feat.load_col or c == feat.heat_value_col:
                optimized_weights[c] = 0.0
            else:
                optimized_weights[c] = raw_weights.get(c, feat.weights.get(c, 0.0))

        # 基线对比
        baseline_weights = {name: feat.weights.get(name, 0.0) for name in self.opt_feature_names}
        optimized_raw = {name: round(float(w), 2) for name, w in zip(self.opt_feature_names, best_weights)}

        changes = {}
        for name in self.opt_feature_names:
            base = baseline_weights[name]
            opt_w = optimized_raw[name]
            if base > 1e-8:
                changes[name] = round((opt_w - base) / base * 100, 1)
            else:
                changes[name] = None

        # 构建 yaml 权重块
        yaml_weights = {}
        for c in feat.raw_features:
            if c == feat.load_col or c == feat.heat_value_col:
                yaml_weights[c] = 0.0
            elif c in optimized_raw:
                yaml_weights[c] = optimized_raw[c]
            else:
                yaml_weights[c] = feat.weights.get(c, 0.0)

        report = {
            "best_loss": best_loss,
            "total_time_sec": round(total_time, 1),
            "n_generations": self.n_generations,
            "population_size": self.pop_size,
            "baseline_weights": baseline_weights,
            "optimized_raw_weights": optimized_raw,
            "weight_change_pct": changes,
            "yaml_weights_block": yaml_weights,
            "note": "将 yaml_weights_block 的内容粘贴到 defaults.yaml 的 features.weights 段即可应用最优权重",
            "history": history,
        }

        output_dir = Path(self.cfg.paths.stable_parquet).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / self.cfg.optimize_genetic.report_csv
        pd.DataFrame(history).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n训练历史已保存: {csv_path}")

        json_path = output_dir / self.cfg.optimize_genetic.report_json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"权重报告已保存: {json_path}")

        return report


# =============================================================
# 数据准备
# =============================================================

def _prepare(cfg):
    """加载标准样本、残差模型、查询数据，返回 (store, evaluators, time_col)。"""
    from plan_center.standard_store import build_standard_store
    from plan_center.features import load_residual_models

    print("[1] 加载标准样本和残差模型...")
    store = build_standard_store(cfg)
    models = load_residual_models(cfg.paths.residual_model_dir, cfg.features.residual_targets)
    print(f"    标准样本数量: {len(store.df_standard)}")

    print("\n[2] 读取查询数据...")
    df_query, time_col = _load_query_data(cfg)
    print(f"    查询数据: {df_query.shape}，时间范围 {df_query[time_col].min()} ~ {df_query[time_col].max()}")

    # 计算残差特征
    from plan_center.optimize_weights import add_residual_features_batch
    print("    计算残差特征（批量预测）...")
    t0 = time.perf_counter()
    df_query = add_residual_features_batch(df_query, models, cfg.features)
    print(f"    残差特征计算完成，耗时 {time.perf_counter() - t0:.1f}s")

    # 预计算标准样本侧（StandardCache）
    from plan_center.optimize_weights import StandardCache
    std_cache = StandardCache(cfg, store)

    # 构建评估器
    print("\n[3] 构建评估器...")
    evaluators = _build_evaluators(cfg, store, df_query, time_col, std_cache)
    print(f"    评估器数量: {len(evaluators)}（每 batch {cfg.optimize_genetic.batch_days} 天）")

    return store, evaluators, time_col


def _load_query_data(cfg):
    """读取分钟级查询数据，应用列别名，返回 df + time_col。"""
    query_parquet = cfg.paths.query_parquet
    if not query_parquet or not Path(query_parquet).exists():
        raise FileNotFoundError(f"查询数据不存在: {query_parquet}")

    df = pd.read_parquet(query_parquet)

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


# =============================================================
# 主入口
# =============================================================

def run_genetic_optimize(config_path=None):
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== 遗传进化算法权重寻优 ===\n")

    # 1. 加载配置
    from plan_center.config import load_config
    cfg = load_config(config_path)
    opt_gen = cfg.optimize_genetic
    if opt_gen is None:
        raise ValueError("defaults.yaml 缺少 optimize_genetic: 段，请检查配置")

    # 2. 准备
    store, evaluators, time_col = _prepare(cfg)

    # 3. 初始化优化器并运行
    optimizer = GeneticOptimizer(cfg, None, store, evaluators)
    report = optimizer.run()

    print("\n=== 寻优完成 ===")
    print("如需应用最优权重，将 optimize_genetic_report.json 中的 yaml_weights_block 粘贴到 defaults.yaml 的 features.weights 段。")
    return report


def main():
    parser = argparse.ArgumentParser(description="遗传进化算法权重寻优")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认 defaults.yaml）")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    run_genetic_optimize(config_path)


if __name__ == "__main__":
    main()
