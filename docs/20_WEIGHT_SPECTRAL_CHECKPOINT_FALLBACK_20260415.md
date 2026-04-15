# Weight Spectral Checkpoint Fallback (2026-04-15)

## Summary

- 目标：在 `checkpoint_ranking` 任务里补一条 **fallback / parallel branch**，用于“更接近 checkpoint 本体分析、而不是 response / activation 分析”的场景。
- 约束：**不替代** 现有 response / activation 主线；只在能访问 checkpoint 权重、甚至只能访问权重时提供额外 ranking 信号。
- 入口脚本：`exp/checkpoint_ranking/weight_spectral.py`
- 产物：
  - `outputs/weight_spectral_oof.csv`
  - `outputs/weight_spectral_feature_importance.csv`
  - `outputs/weight_spectral_report.md`

## What Was Added

### 1) 权重差分特征

对每个 checkpoint 提取了两套差分家族：

- `delta-to-base`
- `delta-to-prev`

并在 `global` / `per-layer` / `per-module` 三个粒度上汇总以下特征：

- Frobenius norm
- Frobenius ratio to reference
- mean absolute drift
- cosine to reference checkpoint
- update sparsity
- sign-flip ratio

模块级 drift 至少覆盖：

- `attention`
- `mlp`
- `embedding`
- `lm_head`
- `norm`

层级特征来自 `model.layers.{0..35}`，因此可以直接分析“哪些层最有预测力”。

### 2) 随机谱探测

在 principal 矩阵（如 `attn_q / attn_o / mlp_up / mlp_down / embedding / lm_head`）上补了随机谱探测：

- `randomized_svd`
- low-rank energy concentration
- singular value summary
- anisotropy
- stable rank
- effective rank
- random probe vector responses
- Hutchinson-style quadratic summaries

考虑到 full exact 扫描全部 4B 权重代价较高，脚本默认使用：

- **sampled drift / exact spectral sketches**

即：

- drift 侧用固定抽样估计全模型 drift 统计；
- spectral 侧仍对抽样 sketch 做精确的 `randomized_svd` / probe / trace summary。

如需完整逐元素 drift 扫描，可加：

```bash
python3 exp/checkpoint_ranking/weight_spectral.py --exact-drift
```

### 3) 轻量建模

脚本里实现了三段轻量建模：

- pointwise `Ridge` 回归头
- within-scenario pairwise `LogisticRegression` ranking head
- 对融合分数做 1-D temporal smoothing

其中 pairwise 分支只在同一 scenario 内比较 checkpoint，不跨场景泄漏。

### 4) 解释性输出

脚本会自动写出：

- OOF 分数与 full-fit 分数
- 每个特征的 importance
- 层重要性 / 模块重要性
- `delta-to-prev` vs `delta-to-base` 子集对比
- random probe feature 主导层

## Local Run Context

本地可直接使用的权重族为：

- `/home/jovyan/public-ro/NAD_RL/math5000RL_neuron_analysis/model`

对应 `Qwen3-4B-Base_base` 到 `Qwen3-4B-Base_math7500-step-1000` 的 11 个 checkpoint。

监督标签来源于：

- `results/scans/checkpoint_ranking/es_svd_math_rr_r1/es_svd_math_rr_r1__math5000rl_slot100_meanconf_eval.json`

其中 `true_accuracy` 被用作当前本地的 checkpoint-ranking supervision。

## Main Local Results

来自 `outputs/weight_spectral_report.md`：

### Full-fit

- Spearman ρ = `0.8364`
- Pearson r = `0.9693`
- Kendall τ = `0.7455`
- Top-1 hit = `1`
- Top-3 overlap = `2`

### OOF

- Spearman ρ = `0.6273`
- Pearson r = `0.9074`
- Kendall τ = `0.4545`
- Top-1 hit = `0`
- Top-3 overlap = `1`

### 当前读数

- 最有预测力的层集中在：`12 / 0 / 10 / 19 / 2 / 23 / 34`
- 随机 probe 响应最有区分度的层集中在：`19 / 5 / 31 / 30 / 35`
- 当前 math-heavy 场景下，模块重要性更偏：
  - `attention`
  - `embedding`
  - `lm_head`
  - `mlp`
- `delta-to-prev` 子集在 OOF 上略强于 `delta-to-base`
  - `delta_prev_only`: Spearman ρ = `0.6364`
  - `delta_base_only`: Spearman ρ = `0.5818`

## How To Run

默认运行：

```bash
python3 exp/checkpoint_ranking/weight_spectral.py
```

常用可调参数：

```bash
python3 exp/checkpoint_ranking/weight_spectral.py \
  --sketch-rows 192 \
  --sketch-cols 192 \
  --svd-rank 6 \
  --probe-count 4 \
  --feature-cap 96 \
  --smooth-lambda 0.35
```

完整 drift 扫描：

```bash
python3 exp/checkpoint_ranking/weight_spectral.py --exact-drift
```

## Notes

- 这条线当前只在本地可见的 `math5000rl_qwen3_4b` scenario 上产出结果；如果后续补到更多权重家族，脚本结构已经支持复用。
- 脚本依赖 `safetensors` 来流式读取 checkpoint 权重；若环境缺失，可先执行：

```bash
python3 -m pip install safetensors
```

- `outputs/weight_spectral_report.md` 是自动生成报告；本文件是人工整理后的说明文档。
- 当前仓库的 `bash cookbook/00_setup/verify.sh` 仍因环境缺少若干已有依赖失败，这不是本次 fallback 分支单独引入的问题。

## Why This Branch Exists

- response / activation 支线回答的是“模型在样本上的行为信号”
- weight spectral fallback 回答的是“checkpoint 本体在参数空间里的更新结构”

两条线的观察对象不同，因此应该并行保留，而不是互相覆盖。
