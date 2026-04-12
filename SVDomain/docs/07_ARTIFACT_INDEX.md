# Artifact Index

这份文档提供一份论文与复现实验最常用 artifact 的集中索引。

---

## 1. 模型文件

### Canonical `r1`

- `models/ml_selectors/es_svd_math_rr_r1.pkl`
- `models/ml_selectors/es_svd_science_rr_r1.pkl`
- `models/ml_selectors/es_svd_ms_rr_r1.pkl`
- `models/ml_selectors/es_svd_coding_rr_r1.pkl`

### Registry

- `SVDomain/registry.json`

---

## 2. 原始结果文档

- `docs/ES_SVD_MS_RR_R1.md`
- `docs/11_CROSS_ANCHOR_TRANSFER.md`
- `docs/ES_SVD_CODING_RR_R1.md`
- `docs/16_DENSE_ANCHOR_EARLYSTOP.md`
- `docs/17_DENSE_CROSS_ANCHOR_TRANSFER.md`
- `docs/SVD_INTERPRETABILITY_R1_20260411.md`
- `docs/BESTOFN_ES_SVD_MS_RR_R1_SLOT100_20260411.md`
- `docs/ES_SVD_MATH_RL_CHECKPOINT_RANKING.md`

---

## 3. 训练 / 评估汇总 JSON

### Canonical multi-domain

- `results/scans/earlystop/es_svd_ms_rr_r1_summary.json`
- `results/scans/earlystop/es_svd_ms_rr_r1_eval.json`

### Coding branch

- `results/scans/earlystop/es_svd_coding_rr_r1_summary.json`
- `results/scans/earlystop/es_svd_coding_rr_r1_eval.json`
- `results/scans/earlystop/es_svd_coding_rr_r1_blind_coding_scores.json`

### Checkpoint ranking

- `results/scans/checkpoint_ranking/es_svd_math_rr_r1/es_svd_math_rr_r1__math5000rl_slot100_meanconf_eval.json`

---

## 4. 提交文件

### EarlyStop

- `submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json`
- `submission/EarlyStop/es_svd_ms_rr_r1__coding_rr_r1.json`

### Best-of-N

- `submission/BestofN/extreme12/patches/es_svd_ms_rr_r1__coding_from_round1c__slot100.json`

### Checkpoint Ranking

- `submission/CheckpointRanking/es_svd_math_rr_r1/es_svd_math_rr_r1__math5000rl_slot100_meanconf.json`

---

## 5. Leaderboard 凭据 / receipt

### Checkpoint Ranking

- `submission/resultofleaderboard/Checkpoint Ranking — es_svd_math_rr_r1__math5000rl_slot100_meanconf.txt`

### 说明

EarlyStop 与 Best-of-N 的在线结果当前主要整理在：

- `docs/ES_SVD_MS_RR_R1.md`
- `docs/BESTOFN_ES_SVD_MS_RR_R1_SLOT100_20260411.md`

---

## 6. 解释性 artifact

### Math

- `results/interpretability/es_svd_math_rr_r1/manifest.json`
- `results/interpretability/es_svd_math_rr_r1/model_summary.json`
- `results/interpretability/es_svd_math_rr_r1/problem_top1_vs_top2.jsonl`
- `results/interpretability/es_svd_math_rr_r1/sanity_checks.json`

### Science

- `results/interpretability/es_svd_science_rr_r1/manifest.json`
- `results/interpretability/es_svd_science_rr_r1/model_summary.json`
- `results/interpretability/es_svd_science_rr_r1/problem_top1_vs_top2.jsonl`
- `results/interpretability/es_svd_science_rr_r1/sanity_checks.json`

### Multi-domain

- `results/interpretability/es_svd_ms_rr_r1/manifest.json`
- `results/interpretability/es_svd_ms_rr_r1/model_summary.json`
- `results/interpretability/es_svd_ms_rr_r1/problem_top1_vs_top2.jsonl`
- `results/interpretability/es_svd_ms_rr_r1/sanity_checks.json`

---

## 7. 关键代码入口

### 训练

- `SVDomain/train_es_svd_ms_rr_r1.py`
- `SVDomain/train_es_svd_coding_rr_r1.py`

### Explain core

- `nad/explain/svd_explain.py`

### 导出

- `scripts/export_svd_explanations.py`

### Viewer

- `cot_viewer/app.py`
- `cot_viewer/static/app.js`
- `cot_viewer/static/styles.css`
- `cot_viewer/templates/index.html`

---

## 8. 本目录里的论文友好入口

### 文档

- `SVDomain/docs/00_EXECUTIVE_SUMMARY.md`
- `SVDomain/docs/06_PAPER_OUTLINE.md`
- `SVDomain/docs/17_DENSE_CROSS_ANCHOR_TRANSFER.md`

### 结果整理

- `SVDomain/results/comparison_tables.md`
- `SVDomain/results/summary_metrics.json`
- `SVDomain/results/tables/*.csv`
- `SVDomain/docs/10_STRUCTURED_OOD.md`
- `SVDomain/results/tables/dense_cross_anchor_transfer_matrix.csv`
- `SVDomain/results/tables/dense_cross_anchor_transfer_deltas.csv`
- `SVDomain/results/tables/dense_cross_anchor_transfer_summary.csv`

### 环境与复现

- `SVDomain/env/README.md`
- `SVDomain/examples/reproduce_commands.sh`

---

## 9. 建议引用顺序

如果你在写正文：

1. 先从 `SVDomain/docs/03_RESULTS_AND_COMPARISONS.md` 找 narrative
2. 再到 `SVDomain/results/tables/*.csv` 取表格
3. 最后回到原始 `results/*.json` 做 appendix / reviewer response
