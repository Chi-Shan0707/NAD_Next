# SVD Interpretability R1 — 2026-04-11

## Summary

- 本轮只做 **SVD 可解释性线**：
  - explain core
  - API
  - `cot_viewer` 集成
  - 导出脚本与解释文档
- **不训练新模型**，**不改 leaderboard / submission 逻辑**，**不碰搜索空间**。
- 当前覆盖的 canonical models：
  - `es_svd_math_rr_r1`
  - `es_svd_science_rr_r1`
  - `es_svd_ms_rr_r1`
- 当前支持的 anchor：
  - `10 / 40 / 70 / 100`

## Deliverables

- explain core：`nad/explain/svd_explain.py`
- 导出脚本：`scripts/export_svd_explanations.py`
- viewer 集成：
  - `cot_viewer/app.py`
  - `cot_viewer/static/app.js`
  - `cot_viewer/static/styles.css`
  - `cot_viewer/templates/index.html`
- 结果产物目录：
  - `results/interpretability/es_svd_math_rr_r1/`
  - `results/interpretability/es_svd_science_rr_r1/`
  - `results/interpretability/es_svd_ms_rr_r1/`

## Interpretation Math

当前 canonical SVD route 的结构固定为：

- `StandardScaler`
- `TruncatedSVD`
- `LogisticRegression`

### Effective linear weight 回投影

令：

- `x_rep`：route 实际输入表示
- `x_scaled = (x_rep - mean) / scale`
- `z = SVD(x_scaled)`
- `beta = lr.coef_[0]`

若 `whiten=yes`，先做：

- `beta_latent = beta / singular_values`

否则：

- `beta_latent = beta`

再回投影到表示空间：

- `w_scaled = svd.components_.T @ beta_latent`
- `w_rep = w_scaled / scale`
- `b_eff = intercept - (mean / scale) · w_scaled`

于是单样本打分可以重写为：

- `score = b_eff + Σ_j x_rep[j] * w_rep[j]`

### Raw + Rank 折叠

对于 `raw+rank` 路径，解释层把一对表示维度折叠回同一个原始 feature：

- `raw_weight`
- `rank_weight`
- `signed_weight = raw_weight + rank_weight`
- `strength = |raw_weight| + |rank_weight|`

单样本贡献也对应拆成：

- `raw_contribution = raw_value * raw_weight`
- `rank_contribution = rank_value * rank_weight`
- `total_contribution = raw_contribution + rank_contribution`

### Reconstruction sanity check

每条 run 都输出：

- `score_reconstructed = b_eff + Σ feature_contribution`
- `reconstruction_error = |score_reconstructed - model_score|`

当前 smoke 导出中的重构误差稳定在数值精度范围内，`sanity_checks.json` 里可直接查看。

## Family Taxonomy

- `confidence`：`tok_conf_*`
- `uncertainty`：`tok_gini_*`、`tok_neg_entropy_*`
- `self_cert_logprob`：`tok_selfcert_*`、`tok_logprob_*`
- `trajectory`：`traj_*`
- `availability_meta`：`has_*`，并兼容未来 `nc_*` / `self_similarity`
- `terminal_tail`：`tail_q10`、`head_tail_gap`、`tail_variance`、`last_event_tail_conf`、`event_pre_post_delta`

说明：

- 当前 canonical 三个模型主路径实际只用到 `token_plus_traj_fixed`，因此 `terminal_tail` 主要是给未来 slot100 解释兼容预留。
- `availability_meta` 保留 `has_rows_bank` 这类 availability flag 的解释位。

## Output Types

### 1) Domain × Anchor 模型层解释

`model_summary` 输出：

- route 元信息：`representation / rank / whiten / class_weight / training_position`
- `top_positive_features`
- `top_negative_features`
- `family_strengths`

### 2) Problem Top1 vs Top2 决策解释

`problem_top1_vs_top2` 输出：

- `top1`
- `top2`
- `margin`
- `why_top1`
- `top_feature_deltas`
- `top_family_deltas`

### 3) Run 样本层解释

`run_contributions` 输出：

- 单条 run 的 `score`
- `intercept_effective`
- `feature_contributions`
- `family_contributions`
- `score_reconstructed`
- `reconstruction_error`

### 4) Wrong Top1 错例解释

离线导出会额外产出：

- `wrong_top1_cases.jsonl`
- `failure_mode_summary.json`

错例对比口径固定为：

- **错误 top1 vs 同题中模型分数最高的正确 run**

## Viewer Integration

`cot_viewer` 现在支持：

- `Method` 切到 canonical SVD methods
- `Anchor` 切到 `10 / 40 / 70 / 100`
- `SVD Domain` 切换总体模型解释视角

首页行为：

- Hero 图显示 4-anchor score trajectory
- Decision 卡显示：
  - 当前方法选中的 run
  - top1 / top2 分数与 margin
  - `why_top1`
  - top family deltas
- Feature Panel 同时显示：
  - `top1 vs top2` feature deltas
  - top family deltas
  - 当前选中 run 的 feature contributions
  - 当前 `domain × anchor` 的 model priors

当前 canonical SVD methods 在 coding cache 上的行为：

- problem-level explain 返回 `applicable=false`
- 不 fallback 到 bridge / leaderboard 方法
- 但 `es_svd_ms_rr_r1` 的 `model_summary` 仍可单独切到 `math` / `science` 视图看总体权重

## Viewer APIs

- `GET /api/svd/explain/model_summary`
  - query：`method`, `domain`, `anchor`
- `GET /api/svd/explain/problem_top1_vs_top2`
  - query：`cache`, `problem_id`, `method`, `anchor`
- `GET /api/svd/explain/run_contributions`
  - query：`cache`, `problem_id`, `sample_id`, `method`, `anchor`

## Export Artifacts

导出命令：

```bash
/home/jovyan/work/NAD_Next/.venv/bin/python scripts/export_svd_explanations.py
```

当前 checked-in 结果产物使用的是 **compact smoke export**：

```bash
/home/jovyan/work/NAD_Next/.venv/bin/python scripts/export_svd_explanations.py --max-problems 1
```

这样做的原因：

- API 与 dashboard 已支持任意 sample 的 on-demand explain
- checked-in artifact 保持紧凑，避免仓库追踪过大的解释 JSONL
- `manifest.json` 明确记录了 `max_problems_per_cache`

每个方法目录至少包含：

- `manifest.json`
- `model_summary.json`
- `problem_top1_vs_top2.jsonl`
- `run_contributions/anchorXXX/*.jsonl`
- `wrong_top1_cases.jsonl`
- `failure_mode_summary.json`
- `sanity_checks.json`

## Validation

已完成的验证：

- `python3 -m py_compile`：
  - `nad/explain/svd_explain.py`
  - `scripts/export_svd_explanations.py`
  - `cot_viewer/app.py`
- `node --check cot_viewer/static/app.js`
- canonical explain 数值 smoke：
  - `es_svd_math_rr_r1`
  - `es_svd_ms_rr_r1`
  - sample-level feature count = `22`
  - family count = `5`
  - reconstruction error ~= `1e-14`
- Flask test client API smoke：
  - `model_summary`
  - `problem_top1_vs_top2`
  - `run_contributions`
  - science anchor `40%`
  - coding cache `applicable=false`
- smoke 导出：
  - `scripts/export_svd_explanations.py --max-problems 1`
  - 三个方法目录均成功生成
  - `sanity_checks.json` 中 `max_abs_error` 为数值精度级别

### Environment note

仓库建议的：

```bash
bash cookbook/00_setup/verify.sh
```

已执行，但当前系统 Python 环境缺少全局 `flask / plotly / tokenizers / transformers / hmmlearn`，因此该脚本未全绿。

本次实现实际使用的是仓库 `.venv`，并且 `.venv` 下的：

- viewer/import smoke
- Flask test client
- exporter

均已通过。
