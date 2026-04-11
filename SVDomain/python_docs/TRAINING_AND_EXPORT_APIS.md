# Training and Export APIs

这份文档对 SVDomain 相关的 Python 入口做一份人工索引。

---

## 1. `SVDomain/train_es_svd_ms_rr_r1.py`

### 作用

训练 canonical `r1` 的三类产物：

- `es_svd_math_rr_r1`
- `es_svd_science_rr_r1`
- `es_svd_ms_rr_r1`

### 关键职责

1. 构建 qualified feature store
2. 按 `dataset + problem_id` 做 holdout split
3. 针对每个 domain 的每个 anchor 训练 SVD route
4. 汇总 route summary / eval summary / artifacts
5. 输出 registry 与文档

### 你最该看的对象

- `FIXED_FEATURE_NAMES`
- `FIXED_REPRESENTATION`
- `METHOD_IDS`
- `_load_or_build_qualified_feature_store`
- `_subset_payload_by_problem_ids`

### 方法层含义

这支脚本实际上定义了 canonical `r1` 的训练协议本身，因此对 paper 来说，它相当于：

- method implementation reference
- reproducibility reference

---

## 2. `SVDomain/train_es_svd_coding_rr_r1.py`

### 作用

训练 coding 单域 canonical branch，并导出 merged blind submission。

### 关键职责

1. 只收集 coding 域 cache
2. 复用 `raw+rank + token_plus_traj_fixed`
3. 输出 coding holdout 结果
4. 生成 blind merge submission
5. 写入 `registry.json`

### 为什么重要

它提供了一个清晰的 negative result：

- 同样的 canonical family 不能自动迁移到 coding

对论文来说，这是一条很有价值的边界结论。

---

## 3. `scripts/export_svd_explanations.py`

### 作用

离线导出 canonical SVD 的解释性 artifact。

### 产出

- `manifest.json`
- `model_summary.json`
- `problem_top1_vs_top2.jsonl`
- `run_contributions/`
- `wrong_top1_cases.jsonl`
- `failure_mode_summary.json`
- `sanity_checks.json`

### 建议用途

- 论文图表
- appendix
- viewer data source

---

## 4. `registry.json`

### 作用

作为 SVD family 的轻量级机器可读索引，记录：

- method id
- kind
- domain
- anchors
- model path
- result / doc path

### 论文价值

它不是正文材料，但很适合作为：

- artifact manifest
- reproducibility appendix 指针

---

## 5. 与训练直接相关的底层模块

### `nad.ops.earlystop_svd`

这里是 canonical SVD route 的核心实现来源，包含：

- SVD bundle 的加载 / 保存
- 表示构建
- rank transform
- 训练与推理的共享逻辑

### `scripts/run_earlystop_prefix10_svd_round1.py`

虽然命名仍保留旧前缀，但当前 canonical `r1` 训练大量复用了它的：

- anchor 常量
- feature-store building
- baseline evaluation
- route summary rendering

### 写进论文时怎么描述

建议不要在正文里强调文件名历史，而是强调：

- 当前 canonical `r1` 是在现有 EarlyStop SVD 工程底座上重新收敛出的干净 family

---

## 6. 推荐引用顺序

如果 reviewer 或合作者要看代码，建议引导他们：

1. `SVDomain/train_es_svd_ms_rr_r1.py`
2. `nad.ops.earlystop_svd`
3. `scripts/export_svd_explanations.py`
4. `SVDomain/registry.json`
