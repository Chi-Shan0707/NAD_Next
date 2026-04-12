# Results Package

本目录存放专门为论文写作整理的结果材料。

原则是：

- 结果表尽量机器可读
- narrative 尽量简洁
- 原始 source of truth 仍然保留在仓库原路径

---

## 内容说明

- `comparison_tables.md`
  - 人类可读版结果总结
- `summary_metrics.json`
  - 机器可读版核心指标
- `tables/*.csv`
  - 可以直接粘到 notebook / spreadsheet / plotting pipeline 的表格

---

## 表格索引

- `tables/earlystop_holdout_summary.csv`
- `tables/blind_leaderboard_summary.csv`
- `tables/bestofn_summary.csv`
- `tables/checkpoint_ranking_summary.csv`
- `tables/interpretability_sanity.csv`
- `tables/structured_ood_results.csv`
- `tables/id_vs_ood_summary.csv`
- `tables/dense_cross_anchor_transfer_matrix.csv`
- `tables/dense_cross_anchor_transfer_deltas.csv`
- `tables/dense_cross_anchor_transfer_summary.csv`

---

## 推荐用法

### 写正文

先看：

- `comparison_tables.md`

### 画图 / 做 appendix

直接读：

- `summary_metrics.json`
- `tables/*.csv`

### reviewer response

先引用本目录表格，再回到原始：

- `results/scans/*`
- `submission/*`
- `results/interpretability/*`
