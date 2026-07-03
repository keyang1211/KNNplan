# -*- coding: utf-8 -*-
"""
optimize_weights_v2.py — 遗传进化算法权重寻优 V2（单 evaluator 版）

目标：最小化「规划中心输出」与「查询行实际控制值」之间的加权 MSE（IQR 归一化）。

V2 改进（相比 V1）：
    - 放弃 batch 分天评估，直接使用最后一个月数据
    - 单 FastEvaluator，简化评估流程
    - 保留 StandardCache + FastEvaluator 的向量化计算优化
    - 速度提升：约 35 小时 → 55 分钟

被优化参数：8 个原始特征权重（主汽流量=0、热值=0 固定不动）。
残差特征权重绑定为 对应原始权重 × residual_weight_ratio。

遗传算子：
    - 选择：锦标赛选择
    - 交叉：模拟二进制交叉（SBX）+ 均匀交叉
    - 变异：高斯变异
    - 精英保留：每代最优 N 个个体直接进入下一代

用法:
    python -m plan_center.optimize_weights_v2
    python -m plan_center.optimize_weights_v2 --config defaults.yaml
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
# 遗传算法算子（复用自 optimize_weights_genetic.py）
# =============================================================

class GeneticOperators:
    """遗传算法算子集合（纯函数，无状态）。"""

    @staticmethod
    def initialize_population(pop_size, n_opt, min_w, max_w, rng):
        return rng.uniform(min_w, max_w, size=(pop_size, n_opt)).astype(np.float64)

    @staticmethod
    def tournament_select(population, fitnesses, tournament_size, rng):
        pop_size = len(population)
        k = min(tournament_size, pop_size)
        indices = rng.choice(pop_size, size=k, replace=False)
        best_idx = indices[np.argmin(fitnesses[indices])]
        return population[best_idx].copy()

    @staticmethod
    def sbx_crossover(p1, p2, eta, rng):
        n = len(p1)
        c1, c2 = p1.copy(), p2.copy()
        for i in range(n):
            if rng.random() < 0.5:
                u = rng.random()
                beta = (2.0 * u) ** (1.0 / (eta + 1.0)) if u <= 0.5 else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
        return c1, c2

    @staticmethod
    def uniform_crossover(p1, p2, rng):
        mask = rng.random(len(p1)) < 0.5
        return np.where(mask, p1, p2), np.where(mask, p2, p1)

    @staticmethod
    def gaussian_mutation(individual, mutation_rate, mutation_scale, weight_range, min_w, max_w, rng):
        result = individual.copy()
        std = weight_range * mutation_scale
        mask = rng.random(len(individual)) < mutation_rate
        result[mask] += rng.normal(0.0, std, size=mask.sum())
        return np.clip(result, min_w, max_w)


# =============================================================
# 数据准备
# =============================================================

def _filter_last_month(df, time_col, days=30):
    """筛选最后 N 天数据。"""
    cutoff = df[time_col].max() - pd.Timedelta(days=days)
    filtered = df[df[time_col] >= cutoff].reset_index(drop=True)
    print(f"    最后 {days} 天数据: {len(filtered)} 行 "
          f"({filtered[time_col].min().date()} ~ {filtered[time_col].max().date()})")
    return filtered


def _prepare(cfg):
    """
    加载数据，构建单个 FastEvaluator（使用最后一个月数据）。

    返回：
        store: StandardStore
        evaluator: FastEvaluator（单 evaluator）
        time_col: 时间列名
    """
    from plan_center.standard_store import build_standard_store
    from plan_center.features import load_residual_models
    from plan_center.optimize_weights import add_residual_features_batch, StandardCache, FastEvaluator, split_by_day

    # 1. 加载标准样本和残差模型
    print("[1] 加载标准样本和残差模型...")
    store = build_standard_store(cfg)
    models = load_residual_models(cfg.paths.residual_model_dir, cfg.features.residual_targets)
    print(f"    标准样本数量: {len(store.df_standard)}")

    # 2. 读取查询数据
    print("\n[2] 读取查询数据...")
    query_parquet = cfg.paths.query_parquet
    if not query_parquet or not Path(query_parquet).exists():
        raise FileNotFoundError(f"查询数据不存在: {query_parquet}")

    df_query = pd.read_parquet(query_parquet)

    # 列别名映射
    aliases = cfg.features.column_aliases
    if aliases:
        for old, new in aliases.items():
            if old in df_query.columns and new not in df_query.columns:
                df_query[new] = df_query[old]
            elif old in df_query.columns and new in df_query.columns:
                df_query = df_query.drop(columns=[old])

    time_col = cfg.time_col or "时间"
    if time_col not in df_query.columns:
        raise ValueError(f"时间列 '{time_col}' 不在查询数据中")

    df_query[time_col] = pd.to_datetime(df_query[time_col], errors="coerce")
    df_query = df_query.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    print(f"    查询数据: {df_query.shape}，时间范围 {df_query[time_col].min()} ~ {df_query[time_col].max()}")

    # 3. 筛选最后一个月
    print("\n[3] 筛选最后一个月数据...")
    df_last_month = _filter_last_month(df_query, time_col, days=30)

    # 4. 计算残差特征
    print("\n[4] 计算残差特征...")
    t0 = time.perf_counter()
    df_last_month = add_residual_features_batch(df_last_month, models, cfg.features)
    print(f"    残差特征计算完成，耗时 {time.perf_counter() - t0:.1f}s")

    # 5. 丢弃 NaN 行
    residual_cols = [f"resid_{t}" for t in cfg.features.residual_targets]
    nan_check_cols = list(dict.fromkeys(
        list(cfg.features.raw_features) + residual_cols + [cfg.features.load_col]
    ))
    n_before = len(df_last_month)
    df_last_month = df_last_month.dropna(subset=nan_check_cols).reset_index(drop=True)
    n_dropped = n_before - len(df_last_month)
    if n_dropped:
        print(f"    丢弃含 NaN 行: {n_dropped} 行")

    # 6. 降采样（如果需要）
    stride = getattr(cfg, 'optimize_v2', None)
    stride = stride.query_stride if stride and hasattr(stride, 'query_stride') else 60
    if stride > 1:
        df_last_month = df_last_month.iloc[::stride].reset_index(drop=True)
        print(f"    降采样 stride={stride}，剩余 {len(df_last_month)} 行")

    # 7. 构建 StandardCache
    print("\n[5] 构建 StandardCache...")
    std_cache = StandardCache(cfg, store)

    # 8. 构建单个 FastEvaluator
    print("\n[6] 构建 FastEvaluator（最后一个月）...")
    evaluator = FastEvaluator(cfg, std_cache, store, df_last_month, time_col)
    print(f"    查询样本数: {evaluator.M}，损失变量数: {len(evaluator.loss_cols)}")

    return store, evaluator, time_col


# =============================================================
# 遗传算法主类（V2 单 evaluator 版）
# =============================================================

class V2Optimizer:
    """遗传进化算法权重寻优器 V2（单 evaluator 版）。"""

    def __init__(self, cfg, std_cache, store, evaluator):
        self.cfg = cfg
        self.std_cache = std_cache
        self.store = store
        self.evaluator = evaluator

        # 获取遗传参数（优先 optimize_v2，fallback 到 optimize_genetic）
        opt = getattr(cfg, 'optimize_v2', None) or cfg.optimize_genetic
        feat = cfg.features

        self.pop_size = opt.population_size
        self.n_generations = opt.n_generations
        self.elite_size = opt.elite_size
        self.tournament_size = opt.tournament_size
        self.crossover_rate = opt.crossover_rate
        self.crossover_method = opt.crossover_method
        self.mutation_rate = opt.mutation_rate
        self.mutation_scale = opt.mutation_scale
        self.min_w = opt.min_weight
        self.max_w = opt.max_weight
        self.sbx_eta = opt.sbx_eta

        # 可优化特征名（raw_features 中排除 load_col 和 heat_value_col）
        self.opt_feature_names = [
            c for c in feat.raw_features
            if c not in (feat.load_col, feat.heat_value_col)
        ]
        self.n_opt = len(self.opt_feature_names)

        self.rng = np.random.default_rng(42)
        self.weight_range = self.max_w - self.min_w

        # 报告输出路径
        self.report_csv = getattr(opt, 'report_csv', 'optimize_v2_report.csv')
        self.report_json = getattr(opt, 'report_json', 'optimize_v2_report.json')

    def _evaluate_individual(self, weights):
        """评估单个个体（单次 forward）。"""
        loss = self.evaluator.forward(weights)
        return loss if np.isfinite(loss) else np.inf

    def _evaluate_population(self, population):
        """批量评估种群。"""
        pop_size = len(population)
        fitnesses = np.full(pop_size, np.inf, dtype=np.float64)
        for i in range(pop_size):
            fitnesses[i] = self._evaluate_individual(population[i])
        return fitnesses

    def _select_parents(self, population, fitnesses):
        return GeneticOperators.tournament_select(
            population, fitnesses, self.tournament_size, self.rng
        )

    def _crossover(self, p1, p2):
        if self.rng.random() < self.crossover_rate:
            if self.crossover_method == "blend":
                return GeneticOperators.sbx_crossover(p1, p2, self.sbx_eta, self.rng)
            elif self.crossover_method == "uniform":
                return GeneticOperators.uniform_crossover(p1, p2, self.rng)
        return p1.copy(), p2.copy()

    def _mutate(self, individual):
        return GeneticOperators.gaussian_mutation(
            individual, self.mutation_rate, self.mutation_scale,
            self.weight_range, self.min_w, self.max_w, self.rng
        )

    def _evolve_one_generation(self, population, fitnesses):
        """进化一代：精英保留 + 选择交叉变异 + 评估子代。"""
        pop_size = len(population)
        n_offspring = pop_size - self.elite_size

        # 1. 精英保留
        elite_order = np.argsort(fitnesses)[:self.elite_size]
        new_pop = population[elite_order].copy()
        new_fitnesses = fitnesses[elite_order].copy()

        # 2. 生成子代
        offspring = []
        while len(offspring) < n_offspring:
            p1 = self._select_parents(population, fitnesses)
            p2 = self._select_parents(population, fitnesses)
            c1, c2 = self._crossover(p1, p2)
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

    def run(self):
        """运行遗传算法寻优。"""
        print(f"=== 遗传进化算法权重寻优 V2（单 evaluator）===\n")
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

        print(f"    初始最佳 loss: {best_fitness:.4f}")
        print(f"    初始最佳权重: " + "  ".join(
            f"{n}={w:.2f}" for n, w in zip(self.opt_feature_names, best_individual)
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

    def _save_report(self, best_weights, best_loss, history, total_time):
        """保存报告文件。"""
        feat = self.cfg.features

        # 构建最终权重
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

        csv_path = output_dir / self.report_csv
        pd.DataFrame(history).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n训练历史已保存: {csv_path}")

        json_path = output_dir / self.report_json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"权重报告已保存: {json_path}")

        return report


# =============================================================
# 主入口
# =============================================================

def run_v2(config_path=None):
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== 遗传进化算法权重寻优 V2（单 evaluator）===\n")

    # 1. 加载配置
    from plan_center.config import load_config
    cfg = load_config(config_path)

    # 优先使用 optimize_v2，fallback 到 optimize_genetic
    opt_v2 = getattr(cfg, 'optimize_v2', None)
    opt_gen = cfg.optimize_genetic

    if opt_v2 is None and opt_gen is None:
        raise ValueError("defaults.yaml 缺少 optimize_v2 或 optimize_genetic 段，请检查配置")

    print(f"使用配置: {'optimize_v2' if opt_v2 else 'optimize_genetic'}")

    # 2. 准备（单个 evaluator）
    store, evaluator, time_col = _prepare(cfg)

    # 3. 初始化优化器并运行
    optimizer = V2Optimizer(cfg, None, store, evaluator)
    report = optimizer.run()

    print("\n=== 寻优完成 ===")
    print("如需应用最优权重，将 optimize_v2_report.json 中的 yaml_weights_block 粘贴到 defaults.yaml 的 features.weights 段。")
    return report


def main():
    parser = argparse.ArgumentParser(description="遗传进化算法权重寻优 V2")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认 defaults.yaml）")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    run_v2(config_path)


if __name__ == "__main__":
    main()
