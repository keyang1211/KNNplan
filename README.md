# 规划中心模块化框架 — 使用说明

## 文件结构

```
plan_center/
├── __init__.py              # 包入口，导出主要类和函数
├── config.py                # 配置 dataclass + load_config(yaml)
├── defaults.yaml            # 默认配置文件（所有参数集中在此）
├── similarity.py            # 相似度计算纯函数（归一化、加权余弦、硬门控）
├── features.py              # 残差模型加载 + 残差特征计算（9原→15维）
├── standard_store.py        # 标准样本V加载（读parquet + 归一化参数）
├── query.py                 # 单次查询核心 query_one() → PlanResult
├── continuity.py            # 输出端连续性（时间间隔重置 + 变化率限幅）
├── engine.py                # PlanningEngine 类（组装模块，持有V+模型）
├── batch.py                 # 批量驱动 run_batch()：逐行查询 + parquet输出
├── schemas.py               # 列名前缀常量 + PlanResult dataclass
├── train_residual.py        # 残差特征训练模块（训练 + 归一化参数 + 向量数据库）
└── validate_visual.py       # 7天连续数据可视化
```

## 快速开始

### 1. 环境要求
```bash
conda activate AI_env
pip install pandas numpy scikit-learn joblib pyyaml plotly
```

### 2. 配置文件

编辑 `defaults.yaml`，填入实际路径：

```yaml
paths:
  stable_parquet: "向量数据库parquet路径"
  residual_model_dir: "残差模型目录"
  norm_stats_path: "归一化参数json路径"
  query_parquet: "分钟级查询数据路径"
```

## 核心功能

### A. 残差特征训练（从原始数据生成向量数据库）

如果拿到的是**原始稳定工况数据**（无resid_*列），需要先运行训练：

```bash
python -m plan_center.train_residual
```

**输入**：稳定工况parquet（只有原始特征 + 效率）
**输出**：
- `vector_db.parquet` — 向量数据库（原始特征 + resid_* + 效率）
- `residual_models/` — 6个残差模型（.joblib）
- `norm_stats.json` — 归一化参数
- `model_report.csv` / `residual_report.csv` — 训练报告

**配置**（defaults.yaml → train段）：
```yaml
train:
  input_parquet: "原始数据路径"
  output_dir: "输出目录"
  enable_filter: true        # 合理工况筛选开关
  filter_q_low: 0.02         # 下分位数
  filter_q_high: 0.98        # 上分位数
  enable_stratified_split: true  # 分层抽样开关
```

### B. 单次查询（给定一条工况向量，返回规划中心）

```python
from plan_center import PlanningEngine

engine = PlanningEngine("defaults.yaml")

result = engine.plan_one(
    raw_features={
        "主汽流量": 250.0, "主汽压力": 13.0, "吨煤产汽平均值": 7.3,
        "炉膛差压": 800.0, "一次风流量": 140000, "床温": 880.0,
        "料层差压": 6.3, "锅炉出口氧量": 4.6, "二次风风量": 57.0,
        "热值": 5000.0,
    },
    prev_center=None,      # 上一分钟中心（连续性）
    prev_time=None,
    current_time="2026-06-24 10:00:00",
)

print(result.final_plan_center)  # 规划中心
print(result.match_status)       # 匹配状态
print(result.similarity_best)    # 最佳相似度
```

### C. 批量查询（读取parquet，逐行查询，输出parquet）

```python
from plan_center import PlanningEngine, run_batch

engine = PlanningEngine("defaults.yaml")

df_out = run_batch(
    engine,
    query_parquet="查询数据.parquet",
    output_parquet="输出.parquet",
    time_col="时间",
)
```

**输出parquet列**：
- 原始数据列（透传）
- `规划中心_*` × 8（规划控制变量）
- `原始规划中心_*` × 8（连续性处理前）
- 匹配诊断：规划匹配状态、相似度S、匹配度D、TopK数量等
- 连续性诊断：连续性处理状态、限幅触发特征等
- 回退诊断：低相似度回退、规划中心来源

### D. 可视化验证（1天连续数据）

```bash
python -m plan_center.validate_visual
```

**输出**：`validate_visual_1day.html`，包含7个子图（实际值 vs 规划值）：
- 主汽流量（负荷）
- 床温
- 一次风流量
- 料层差压
- 炉膛差压
- 锅炉出口氧量
- 匹配度（相似度S）

## 配置参数说明

### features（特征配置）
- `raw_features`：9个原始相似度特征
- `residual_targets`：6个残差目标（模型预测目标）
- `residual_inputs`：3个残差模型输入（主汽流量、主汽压力、吨煤产气量）
- `weights`：原始特征权重（主汽流量=0硬门控，热值=0占位）
- `residual_weight_ratio`：残差权重 = 原始权重 × ratio（默认0.5）
- `plan_center_cols`：输出控制变量（8个）
- `column_aliases`：列别名映射（用于不同数据源的列名对齐）

### matching（匹配配置）
- `d_weight_s` / `d_weight_e`：D = a*S + b*E 中的权重
- `top_k`：Top-k数量（默认5）
- `low_sim_fallback_threshold`：低相似度回退阈值（默认0.97）

### flow_gate（硬门控）
- `enable`：是否启用主汽流量硬门控
- `mode`：absolute（绝对偏差）或 relative（相对偏差）
- `abs_threshold`：绝对偏差阈值（默认30.0 t/h）

### continuity（连续性）
- `enable_rate_limit`：是否启用变化率限幅
- `reset_on_time_gap`：时间间隔过大时是否重置
- `max_gap_minutes`：最大间隔分钟数（默认5）

### train（训练配置）
- `enable_filter`：合理工况筛选开关
- `filter_cols`：筛选特征列表
- `enable_stratified_split`：分层抽样开关

## 数据流

```
原始稳定工况数据
    ↓
[train_residual.py] 训练残差模型 + 计算归一化参数
    ↓
向量数据库(parquet) + 模型(.joblib) + 归一化参数(json)
    ↓
[engine.py] 加载数据 + 模型
    ↓
[query.py] 单次查询 → PlanResult
    ↓
[continuity.py] 连续性处理 → 最终规划中心
```
