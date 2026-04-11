# Python Docs

本目录不是自动生成的 API 文档，而是面向论文复现与代码阅读的 **人工整理版 Python 入口说明**。

适用对象：

- 需要快速理解训练脚本入口的人
- 需要把 explainability / viewer 写进方法或 appendix 的人
- 需要给 reviewer 指路的人

---

## 文档索引

- `TRAINING_AND_EXPORT_APIS.md`
  - 训练脚本
  - 导出脚本
  - registry / artifact 输出
- `VIEWER_AND_EXPLAIN_APIS.md`
  - explain core
  - Flask API
  - 前端 viewer 行为

---

## 建议阅读顺序

### 只关心训练

读：

1. `TRAINING_AND_EXPORT_APIS.md`
2. `SVDomain/train_es_svd_ms_rr_r1.py`
3. `SVDomain/train_es_svd_coding_rr_r1.py`

### 只关心解释与 dashboard

读：

1. `VIEWER_AND_EXPLAIN_APIS.md`
2. `nad/explain/svd_explain.py`
3. `cot_viewer/app.py`
