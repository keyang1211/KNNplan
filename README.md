# 规划中心模块化框架 — 使用说明

## 文件结构

```
plan_center/
├── __init__.py              # 包入口，导出主要类和函数
├── config.py                # 配置 dataclass + load_config(yaml)
├── defaults.yaml            # 默认配置文件（所有参数集中在此）
├── similarity.py            # 相似度计算纯函数（归一化、加权余弦、硬门控、动态候选归一化）
├── features.py              # 残差模型加载 + 残差特征计算（9原→15维）
├── standard_store.py        # 标准样本V加载（读parquet，不预归一化）
├── query.py                 # 单次查询核心 query_one() → PlanResult
├── continuity.py            # 输出端连续性（时间间隔重置 + 变化率限幅）
├── engine.py                # PlanningEngine 类（组装模块，持有V+模型）
├── batch.py                 # 批量驱动 run_batch()：逐行查询 + parquet输出
├── schemas.py               # 列名前缀常量 + PlanResult dataclass
├── train_residual.py        # 残差特征训练模块（训练 + 向量数据库，不计算归一化参数）
├── run_once.py              # 单次查询示例：输出 Top-K 结果到 CSV（含输入信息）
├── validate_visual.py       # 连续数据可视化（随机或指定时间范围）
├── run_10day_sample.py      # 批量随机采样查询：最后30天随机采样，输出 Top-5 详情
├── optimize_weights.py      # 权重梯度寻优（数值梯度下降）
├── optimize_weights_genetic.py  # 遗传进化算法权重寻优
├── optimize_weights_v2.py   # 遗传进化算法 V2（单 evaluator）
├── visual_compare.py        # 可视化对比工具
└── weight_comparison_eval.py # 权重对比评估工具
```

## 核心设计决策

1. **归一化策略**：查询时动态计算候选子集的 median/IQR，不使用全局归一化参数
2. **主汽流量**：权重为 0，作为硬门控（偏差超阈值直接筛掉）
3. **残差特征**：6个 HistGradientBoostingRegressor，每个目标一个模型
4. **相似度**：加权余弦 `(cos+1)/2 ∈ [0,1]`，归一化采用 robust 方法 `(x - median) / IQR`
5. **聚类**：complete-link agglomerative，`distance = 1 - similarity`
6. **规划中心**：模式1=最佳单样本，模式2=Top-k加权平均

## 快速开始

### 1. 环境要求

```bash
conda activate AI_env
pip install pandas numpy scikit-learn joblib pyyaml plotly
```

### 2. 数据准备

需要两类数据：
- **稳定工况数据**：`#4_final_stable_df_mean_3_4_02.parquet`（用于训练残差模型，构建向量数据库）
- **查询数据**：`#4_df_all_1min.parquet`（分钟级实时数据，用于查询和可视化）

### 3. 第一步：训练残差模型 + 构建向量数据库

```bash
python -m plan_center.train_residual
```

**输入**：稳定工况 parquet（只有原始特征 + 效率，无 resid_* 列）  
**输出**（保存在 `defaults.yaml → train.output_dir`）：
- `vector_db.parquet` — 向量数据库（原始特征 + resid_* + 效率 + 身份列）
- `residual_models/residual_model_*.joblib` — 6个残差模型
- `model_report.csv` / `residual_report.csv` — 训练报告

**注意**：训练阶段不再计算归一化参数，归一化改为查询时动态计算候选子集的统计量。

**配置**（`defaults.yaml → train` 段）：
```yaml
train:
  input_parquet: "D:/redian/vectorsearch/#4_final_stable_df_mean_3_4_02.parquet"
  output_dir: "D:/redian/vectorsearch/plan_center/output"
  enable_filter: true        # 合理工况筛选开关
  filter_q_low: 0.02         # 下分位数
  filter_q_high: 0.98        # 上分位数
  enable_stratified_split: true  # 分层抽样开关
  oof_n_splits: 5            # OOF 折数
```

### 4. 第二步：查询/可视化/寻优

训练完成后，使用 `PlanningEngine` 或各脚本进行查询、可视化、寻优。

---

## 组件详解

### A. config.py + defaults.yaml — 配置中心

**职责**：集中管理所有参数，加载 YAML 配置为 dataclass。

**关键配置结构**：

```yaml
features:      # 特征配置
  raw_features: [吨煤产气量, 主汽压力, ...]    # 9个原始特征
  residual_targets: [炉膛差压, 床温, ...]     # 6个残差目标
  weights: {主汽压力: 0.98, 炉膛差压: 0.50, ...}  # 原始特征权重
  residual_weight_ratio: 0.5                  # 残差权重 = 原始权重 × 0.5
  load_col: 主汽流量                          # 负荷列（硬门控）
  eff_col: 锅炉效率                           # 效率列
  plan_center_cols: [主汽流量, 主汽压力, ...] # 输出控制变量

matching:      # 匹配配置
  d_weight_s: 1.0    # D = a*S + b*E 中的 a
  d_weight_e: 0.0    # D = a*S + b*E 中的 b
  top_k: 5           # Top-k 数量
  plan_center_mode: 1  # 1=最佳单样本, 2=Top-k加权均值

flow_gate:     # 硬门控配置
  enable: true
  mode: absolute     # absolute | relative
  abs_threshold: 15.0  # t/h

paths:         # 路径配置
  stable_parquet: "向量数据库路径"
  residual_model_dir: "残差模型目录"
  query_parquet: "查询数据路径"
  cache_path: null   # 缓存路径（null=不缓存）
```

**引用关系**：所有组件都通过 `PlanningConfig` 读取配置。

---

### B. train_residual.py — 残差特征训练

**职责**：从原始稳定工况数据训练残差模型，生成向量数据库。

**数据流**：

```
原始稳定工况数据 (#4_final_stable_df_mean_3_4_02.parquet)
    ↓ load_stable_data()
DataFrame（含原始特征 + 效率）
    ↓ 合理工况筛选（可选）
筛选后 DataFrame
    ↓ 分层抽样（可选）
带 __split 列的 DataFrame
    ↓ train_residual_models()
6个残差模型（HistGradientBoostingRegressor）
    ↓ 5-fold OOF 预测
DataFrame（增加 resid_* 列）
    ↓ 保留必要列
vector_db.parquet（原始特征 + resid_* + 效率 + 身份列）
    ↓ save_outputs()
输出：vector_db.parquet + residual_models/ + model_report.csv
```

**关键函数**：
- `load_stable_data()` — 读取数据，应用列别名，可选排除最后一个月
- `train_residual_models()` — 训练 6 个 HistGradientBoostingRegressor，5-fold OOF
- `save_outputs()` — 保存向量数据库和模型

**注意**：训练阶段**不再计算**归一化参数，归一化改为查询时动态计算。

**使用方式**：
```bash
python -m plan_center.train_residual
```

---

### C. standard_store.py — 标准样本 V 加载

**职责**：读取向量数据库 parquet，加载到内存，供查询使用。

**数据流**：

```
vector_db.parquet
    ↓ pd.read_parquet()
DataFrame（9109行 × 20列）
    ↓ 列别名映射、数值转换、缺失值删除
df_standard（DataFrame）
    ↓ 提取列
loads_standard（主汽流量数组）
sim_feature_cols（特征列名列表）
eff_score_all（效率分位数数组）
    ↓ StandardStore(...)
StandardStore 实例
```

**StandardStore 字段**：
- `df_standard` — 完整 DataFrame（含 resid_* 和效率）
- `loads_standard` — 主汽流量数组，用于硬门控
- `sim_feature_cols` — 15 个相似度特征列名（9 raw + 6 resid）
- `eff_score_all` — 效率分位数 E（0~1）

**注意**：`StandardStore` 不再存储 `norm_stats` 或 `xw_standard`，归一化在查询时动态计算。

**缓存**：支持 joblib 缓存，签名不匹配时自动重建。

**引用关系**：
- 被 `engine.py` 调用（`build_standard_store()`）
- 被 `query.py` 调用（访问 `df_standard`、`loads_standard`、`eff_score_all`）

---

### D. similarity.py — 相似度计算

**职责**：提供相似度相关的纯函数，包括归一化、加权、余弦相似度、硬门控、动态候选归一化。

**核心函数**：

| 函数 | 用途 |
|------|------|
| `robust_norm_stats(df, cols)` | 计算 median/IQR（训练时用） |
| `normalize_features(df, cols, stats)` | robust 归一化 `z = (x - median) / IQR` |
| `weight_array(cols, weights)` | 生成归一化权重向量 |
| `weighted_matrix(df, cols, stats, weights)` | 归一化 + 加权矩阵 |
| `weighted_vector_1d(values, cols, stats, weights)` | 单条向量归一化 + 加权 |
| `candidate_similarity(df, values, cols, stats, weights, override)` | 候选子集相似度（支持动态统计量） |
| `compute_norm_stats_from_df(df, cols)` | **新增**：从 DataFrame 动态计算 median/IQR |
| `compute_and_normalize_candidates(df, values, cols, weights, global_stats, min)` | **新增**：动态归一化 + 相似度 + 回退 |
| `cosine01(a, b)` | 加权余弦相似度，映射到 [0,1] |
| `flow_gate_keep_mask(load_q, loads, gate)` | 主汽流量硬门控，返回布尔掩码 |
| `pct_rank(values)` | 分位数归一化，映射到 [0,1]（用于效率得分 E） |

**动态归一化流程**：

```
候选子集 DataFrame（M 行 × 15 列）
    ↓ compute_norm_stats_from_df()
候选集统计量 {col: {median, iqr}}
    ↓ weighted_matrix() + weighted_vector_1d()
归一化 + 加权后的候选矩阵 + 查询向量
    ↓ cosine01()
余弦相似度数组 [0,1]^M
```

**回退逻辑**：
- 候选集 >= 5 条：使用候选集动态统计量
- 候选集 < 5 条 + 全局统计存在：回退全局统计
- 候选集 < 5 条 + 全局统计不存在：仍使用候选集统计量

**引用关系**：
- 被 `query.py` 调用（`compute_and_normalize_candidates`）
- 被 `run_10day_sample.py` 调用
- 被 `run_once.py` 调用
- 被 `standard_store.py` 调用（`robust_norm_stats`、`pct_rank`）

---

### E. features.py — 残差特征计算

**职责**：加载残差模型，将 9 维原始特征扩展为 15 维（增加 6 个 resid_*）。

**数据流**：

```
原始特征 dict（9维：吨煤产气量、主汽压力、...）
    ↓ make_query_vector_15d()
15维查询向量（9 raw + 6 resid）
    ↓
用于后续相似度计算
```

**关键函数**：
- `load_residual_models(model_dir, targets)` — 加载 6 个 .joblib 模型
- `make_query_vector_15d(raw_features, models, feat)` — 构建 15 维查询向量

**引用关系**：
- 被 `query.py` 调用
- 被 `run_once.py` 调用
- 被 `run_10day_sample.py` 调用

---

### F. query.py — 单次查询核心

**职责**：给定一条工况向量，在向量数据库中做 Top-k 相似度搜索，输出规划中心。

**数据流**：

```
raw_features（dict，9维原始特征）
    ↓ make_query_vector_15d()
q_15d（15维查询向量）
    ↓ flow_gate_keep_mask()
valid_pos（候选样本索引，M 条）
    ↓ df_candidates = store.df_standard.iloc[valid_pos]
候选子集 DataFrame（M × 15）
    ↓ compute_and_normalize_candidates()
s_candidates（M 维相似度数组）
    ↓ D = a*S + b*E
d_candidates（M 维匹配度数组）
    ↓ argsort + top-k
top_pos（K 维原始索引）
    ↓ 填充 PlanResult
PlanResult（规划中心 + 诊断信息）
```

**关键函数**：
- `query_one()` — 不含连续性的单次查询
- `query_one_full()` — 含连续性处理的完整查询

**执行顺序**：
1. 构建 15 维查询向量
2. 硬门控筛选候选（负荷 ±15 t/h）
3. 动态计算候选集归一化参数
4. 归一化候选集 + 查询向量
5. 计算余弦相似度
6. D = a*S + b*E
7. Top-k 排序，映射回原始索引
8. 填充 PlanResult

**引用关系**：
- 调用 `features.make_query_vector_15d()`
- 调用 `similarity.compute_and_normalize_candidates()`、`flow_gate_keep_mask()`
- 调用 `continuity.apply_output_continuity()`（在 `query_one_full` 中）
- 被 `engine.py` 调用

---

### G. continuity.py — 输出端连续性

**职责**：对规划中心输出做连续性处理，防止规划值跳变。

**处理逻辑**：
1. 时间间隔重置：如果间隔 > max_gap_minutes，重置为当前真实工况
2. 变化率限幅：各特征的变化率超过阈值时，限制变化幅度

**关键函数**：
- `apply_output_continuity()` — 应用连续性处理
- `should_reset_continuity()` — 判断是否需要重置
- `has_valid_center()` — 判断中心是否有效

**引用关系**：
- 被 `query.py` 调用（`query_one_full`）

---

### H. engine.py — PlanningEngine 类

**职责**：组装所有模块，对外提供统一接口。

**初始化流程**：

```
PlanningEngine.__init__()
    ↓ load_config()
PlanningConfig（配置对象）
    ↓ build_standard_store()
StandardStore（向量数据库）
    ↓ load_residual_models()
models dict（6个残差模型）
    ↓
PlanningEngine 实例（持有 store + models + cfg）
```

**公开接口**：

| 方法 | 用途 |
|------|------|
| `plan_one(raw_features, prev_center, prev_time, current_time)` | 单次查询（含连续性） |
| `plan_one_no_continuity(raw_features)` | 单次查询（不含连续性） |
| `reload_standard_store()` | 重新加载向量数据库 |

**引用关系**：
- 调用 `config.load_config()`
- 调用 `standard_store.build_standard_store()`
- 调用 `features.load_residual_models()`
- 调用 `query.query_one()` / `query_one_full()`

---

### I. batch.py — 批量驱动

**职责**：读取查询 parquet，逐行调用 `plan_one`，输出 parquet。

**数据流**：

```
query_parquet（分钟级查询数据）
    ↓ pd.read_parquet()
DataFrame（67万行 × 65列）
    ↓ 时间范围筛选
raw_calc（筛选后 DataFrame）
    ↓ 逐行 iterrows()
    ↓ engine.plan_one()
List[PlanResult]
    ↓ build_output_dataframe()
df_out（原始数据 + 规划中心 + 诊断列）
    ↓ to_parquet()
output_parquet
```

**关键函数**：
- `run_batch()` — 批量处理主函数
- `BatchState` — 批量处理状态（prev_center、prev_time）

**引用关系**：
- 调用 `engine.plan_one()`
- 调用 `schemas.build_output_dataframe()`

---

### J. schemas.py — 数据结构

**职责**：定义 PlanResult dataclass 和结果装配工具。

**核心结构**：

```python
@dataclass
class PlanResult:
    raw_plan_center: dict[str, float]      # 原始规划中心
    final_plan_center: dict[str, float]    # 连续性处理后
    match_status: str                       # 匹配状态
    topk_indices: list[int]                 # Top-K 索引
    similarity_best: float                  # 最佳相似度 S
    score_d_best: float                     # 最佳匹配度 D
    # ... 更多诊断字段
```

**关键函数**：
- `plan_result_to_row()` — 将 PlanResult 展平为一行 dict
- `build_output_dataframe()` — 将原始 DataFrame 与 PlanResult 列表拼接

**引用关系**：
- 被 `query.py` 使用（返回 PlanResult）
- 被 `batch.py` 使用（拼接输出）
- 被 `validate_visual.py` 使用

---

### K. validate_visual.py — 可视化验证

**职责**：从分钟级查询数据中选取连续数据，逐分钟调用规划中心，生成实际值 vs 规划值对比图。

**数据流**：

```
query_parquet
    ↓ 随机/指定时间范围选取
df_selected（1天 × 1440分钟）
    ↓ 逐分钟 iterrows()
    ↓ engine.plan_one()
List[PlanResult]
    ↓ build_output_dataframe()
df_out（1354行 × 101列）
    ↓ build_plotly_figure()
Plotly Figure（8个子图）
    ↓ HTML 模板
validate_*.html
```

**输出 HTML 包含**：
- 7 个特征子图（主汽流量、床温、一次风流量、料层差压、炉膛差压、锅炉出口氧量、二次风风量）
- S/D 匹配度子图
- 交互式纵轴范围滑块

**引用关系**：
- 调用 `engine.plan_one()`
- 调用 `schemas.build_output_dataframe()`

---

### L. run_once.py — 单次查询示例

**职责**：读取查询数据的一行，执行查询，输出 Top-K 结果到 CSV。

**使用方式**：
```bash
python -m plan_center.run_once --row-index 0 --output result.csv --top-k 5
```

**输出 CSV 包含**：
- 输入信息（行号、时间、各特征值）
- Top-K 匹配结果（排名、稳定工况ID、相似度S、匹配度D、效率E、所有特征值）

**引用关系**：
- 调用 `engine.plan_one_no_continuity()`
- 调用 `similarity.compute_and_normalize_candidates()`（展示用重新计算）

---

### M. run_10day_sample.py — 批量随机采样查询

**职责**：从查询数据的最后30天中随机采样，执行查询，输出 Top-5 完整详情。

**使用方式**：
```bash
python -m plan_center.run_10day_sample.py --days 10 --points-per-day 10 --random-seed 42
```

**输出 CSV**：每个查询点展开为 top_k 行，包含查询输入和匹配详情。

---

## 归一化流程（当前）

### 训练阶段

`train_residual.py` **不再计算**归一化参数。

```
原始数据 → 残差模型训练 → 向量数据库 parquet（仅存原始值）
```

### 查询阶段（动态归一化）

```
查询向量 q_15d
    ↓
硬门控筛选候选（负荷 ±15 t/h）
    ↓
提取候选子集 df_candidates（M 行）
    ↓
compute_norm_stats_from_df(df_candidates)
    ↓
候选集 median/IQR（15 维）
    ↓
weighted_matrix(df_candidates) + weighted_vector_1d(q_15d)
    ↓
归一化 + 加权后的矩阵/向量
    ↓
cosine01() → 相似度 [0,1]
    ↓
Top-k 排序
```

**回退逻辑**：
- 候选集 < 5 条：回退全局统计（如果存在 `norm_stats.json`）
- 无候选通过门控：按最近负荷兜底

---

## 数据流总览

```
                        训练阶段
原始稳定工况数据 (#4_final_stable_df_mean_3_4_02.parquet)
    ↓
[train_residual.py] 训练残差模型 + 生成 resid_* 列
    ↓
vector_db.parquet（原始特征 + resid_* + 效率）
residual_models/（6个 .joblib）
    ↓
                        查询阶段
查询数据 (#4_df_all_1min.parquet)
    ↓
[engine.py] PlanningEngine 加载 vector_db + models
    ↓
[query.py] 单次查询：
  1. make_query_vector_15d() → 15维查询向量
  2. flow_gate_keep_mask() → 候选子集索引
  3. compute_and_normalize_candidates() → 动态归一化 + 相似度
  4. Top-k 排序 → PlanResult
    ↓
[continuity.py] 连续性处理 → final_plan_center
    ↓
[schemas.py] 装配输出 DataFrame
    ↓
[batch.py / validate_visual.py / run_once.py] 输出/可视化
```

---

## 配置参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `features.raw_features` | 9个特征 | 原始相似度特征 |
| `features.residual_targets` | 6个目标 | 残差模型预测目标 |
| `features.residual_weight_ratio` | 0.5 | 残差权重 = 原始权重 × ratio |
| `features.load_col` | 主汽流量 | 负荷列（硬门控） |
| `matching.d_weight_s` | 1.0 | 相似度权重 |
| `matching.d_weight_e` | 0.0 | 效率权重 |
| `matching.top_k` | 5 | Top-k 数量 |
| `flow_gate.abs_threshold` | 15.0 | 负荷偏差阈值（t/h） |
| `flow_gate.enable` | true | 是否启用硬门控 |

---

## 常见问题

### Q: 为什么训练阶段不计算归一化参数？

A: 改为查询时动态计算候选子集的 median/IQR，使归一化更贴合当前候选集的分布，避免全局统计量对局部搜索的 bias。

### Q: 候选集多大合适？

A: 通常几百到一千条（负荷 ±15 t/h 范围内）。极端情况下可能只有几十条，此时回退全局统计。

### Q: 如果候选集 < 5 条怎么办？

A: 自动回退全局统计量（如果存在 `norm_stats.json`）。如果也不存在，仍使用候选集统计量（可能不稳定）。

### Q: 如何启用/禁用硬门控？

A: 修改 `defaults.yaml → flow_gate.enable`：
```yaml
flow_gate:
  enable: false  # 禁用硬门控，所有样本参与匹配
```

### Q: 如何更换数据？

A: 修改 `defaults.yaml` 中的路径：
```yaml
paths:
  stable_parquet: "新向量数据库路径"
  query_parquet: "新查询数据路径"
train:
  input_parquet: "新原始数据路径"
```
