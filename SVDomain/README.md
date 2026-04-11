# SVDomain

`SVDomain/` 现在不仅是训练脚本目录，也是一套 **可直接支持论文写作、实验复现、结果引用、解释展示** 的 SVD 专题资料包。

这套目录的目标是：当你开始写 paper / tech report / appendix 时，不需要再到仓库各处翻找零散材料，直接从这里进入即可。

---

## 1. 当前主题范围

本目录聚焦 **EarlyStop / Best-of-N SVD family**，当前同时覆盖两条线：

- `r1` canonical EarlyStop 主线（论文主叙事）
- `r2` 10-anchor EarlyStop 扩展线
- `slot100` coding-only Best-of-N bridge 线

核心方法包括：

- `es_svd_math_rr_r1`
- `es_svd_science_rr_r1`
- `es_svd_ms_rr_r1`
- `es_svd_coding_rr_r1`
- `es_svd_math_rr_r2`
- `es_svd_science_rr_r2`
- `es_svd_ms_rr_r2`

命名含义：

- `es_svd`：EarlyStop SVD family
- `math / science / ms / coding`：适用域
- `rr`：`raw+rank`
- `r1`：第一版 canonical domain split bundle
- `r2`：第二版 10-anchor + per-position family search bundle

当前论文主线建议围绕：

- `es_svd_math_rr_r1`
- `es_svd_science_rr_r1`
- `es_svd_ms_rr_r1`

其中：

- `es_svd_ms_rr_r1__coding_from_round1c` 是当前主要 blind leaderboard 提交
- `es_svd_ms_rr_r2` 是当前 non-coding EarlyStop 扩展训练入口
- `es_svd_coding_rr_r1` 更适合作为 **负结果 / ablation / boundary case**
- `slot100` best-of-n coding 线适合作为 **桥接对照 / 补充实验**

---

## 2. 一屏结论

如果你现在要马上开始写论文，先记住这几条：

1. **方法层**：SVD 主干统一使用 `StandardScaler -> TruncatedSVD -> LogisticRegression`，表示固定为 `raw+rank`。
2. **训练层**：`r1` 使用 `10 / 40 / 70 / 100` 四锚点；`r2` 改为全部 `10` 个 early-stop positions，并在每个位置独立搜索 feature family。
3. **结果层**：`es_svd_ms_rr_r1__coding_from_round1c` 仍是 blind leaderboard 上的主 EarlyStop 提交；`es_svd_ms_rr_r2` 是新的 non-coding 扩展报告线。
4. **补充层**：`slot100` coding-only SVDomain 脚本与文档可作为 Best-of-N bridge 补充结果。
5. **解释层**：现在已经有完整的 SVD explain core、viewer API、dashboard 集成，可回答“为什么它是 top1”。

---

## 3. 推荐阅读顺序

如果你是为了写论文，建议按下面顺序读：

1. `SVDomain/docs/00_EXECUTIVE_SUMMARY.md`
2. `SVDomain/docs/01_METHOD_AND_MODELING.md`
3. `SVDomain/docs/02_DATA_PROTOCOL_AND_FEATURES.md`
4. `SVDomain/docs/03_RESULTS_AND_COMPARISONS.md`
5. `SVDomain/docs/04_INTERPRETABILITY_AND_VIEWER.md`
6. `SVDomain/docs/06_PAPER_OUTLINE.md`
7. `docs/ES_SVD_MS_RR_R2_REPORT.md`

如果你是为了复现：

1. `SVDomain/env/README.md`
2. `SVDomain/docs/05_REPRODUCTION_CHECKLIST.md`
3. `SVDomain/examples/reproduce_commands.sh`
4. `SVDomain/SUBMISSION_GUIDE.md`

如果你是为了查 artifact：

1. `SVDomain/docs/07_ARTIFACT_INDEX.md`
2. `SVDomain/results/README.md`
3. `SVDomain/registry.json`
4. `docs/SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md`

---

## 4. 当前目录结构

```text
SVDomain/
├── README.md
├── SUBMISSION_GUIDE.md
├── registry.json
├── train_es_svd_ms_rr_r1.py
├── train_es_svd_coding_rr_r1.py
├── train_es_svd_ms_rr_r2.py
├── train_slot100_svd_domain_r1.py
├── docs/
│   ├── 00_EXECUTIVE_SUMMARY.md
│   ├── 01_METHOD_AND_MODELING.md
│   ├── 02_DATA_PROTOCOL_AND_FEATURES.md
│   ├── 03_RESULTS_AND_COMPARISONS.md
│   ├── 04_INTERPRETABILITY_AND_VIEWER.md
│   ├── 05_REPRODUCTION_CHECKLIST.md
│   ├── 06_PAPER_OUTLINE.md
│   └── 07_ARTIFACT_INDEX.md
├── env/
│   ├── README.md
│   ├── requirements-paper.txt
│   ├── environment.yml
│   └── verify_imports.py
├── python_docs/
│   ├── README.md
│   ├── TRAINING_AND_EXPORT_APIS.md
│   └── VIEWER_AND_EXPLAIN_APIS.md
├── results/
│   ├── README.md
│   ├── comparison_tables.md
│   ├── summary_metrics.json
│   └── tables/
├── figures/
│   ├── README.md
│   ├── pipeline_overview.mmd
│   └── explainability_stack.mmd
└── examples/
    ├── README.md
    └── reproduce_commands.sh
```

---

## 5. 当前协议摘要

### Anchors / positions

- `EARLY_STOP_POSITIONS = [0.1, 0.2, ..., 1.0]`
- `EXTRACTION_POSITIONS = (0.05, 0.1, 0.15, 0.2, 0.3, ..., 1.0)`
- `ANCHOR_POSITIONS (r1) = (0.1, 0.4, 0.7, 1.0)`
- `ANCHOR_POSITIONS (r2) = (0.1, 0.2, ..., 1.0)`

含义：

- `EARLY_STOP_POSITIONS`：真实在线评测和早停决策会看的 10 个停点
- `EXTRACTION_POSITIONS`：离线提特征时额外看的控制点
- `ANCHOR_POSITIONS (r1)`：SVD canonical route 实际训练的 4 个锚点
- `ANCHOR_POSITIONS (r2)`：10 个官方停点全部独立建模

`r1` slot 映射：

- `20 / 30 -> 10`
- `50 / 60 -> 40`
- `80 / 90 -> 70`
- `100 -> 100`

`r2` slot 映射：

- identity（`10 -> 10`, `20 -> 20`, ..., `100 -> 100`）

### 特征和表示

- 表示：`raw+rank`
- `r1` 主特征组：`token_plus_traj_fixed`
- `r2` 主特征组：per-position family search over `token_only / token_plus_traj / all / strong_core3 / strong_event7 / token_plus_traj_global@100%`
- 核心 family：
  - `confidence`
  - `uncertainty`
  - `self_cert_logprob`
  - `trajectory`
  - `availability_meta`

### 数据和切分

- 数据来源：`MUI_HUB/cache` + `MUI_HUB/cache_train`
- holdout：`85 / 15`
- 切分单元：`dataset + problem_id`
- split seed：`42`

---

## 6. 论文可直接引用的主结果

### EarlyStop 主提交

- 方法：`es_svd_ms_rr_r1__coding_from_round1c`
- blind status：`CURRENT BEST`
- primary score：`3.8125`
- `auc_of_auroc = 0.7428`
- `auc_of_selacc = 0.8317`

### Best-of-N slot100 直抽

- 方法：`es_svd_ms_rr_r1__coding_from_round1c__slot100`
- status：`Not best`
- `avg rank = 3.6000`
- `hit@1 = 0.7299`
- `hit@3 = 0.8010`

### EarlyStop `r2` 扩展线

- 训练入口：`SVDomain/train_es_svd_ms_rr_r2.py`
- registry id：`es_svd_math_rr_r2` / `es_svd_science_rr_r2` / `es_svd_ms_rr_r2`
- 报告路径：`docs/ES_SVD_MS_RR_R2_REPORT.md`
- 目标定位：作为 `r1` canonical 线的 10-anchor 扩展与 feature-family search 对照

### Coding 单域分支

- `es_svd_coding_rr_r1` 在 coding holdout 和 blind leaderboard 上都没有超过现有主线
- 更适合作为：
  - 负结果
  - ablation
  - “domain mismatch / weak supervision” 讨论材料

---

## 7. 与解释性线的关系

SVD explainability 现在已经接入：

- `nad/explain/svd_explain.py`
- `scripts/export_svd_explanations.py`
- `cot_viewer/app.py`
- `cot_viewer/static/app.js`

支持输出：

- **模型层**：domain × anchor 权重与 family strength
- **样本层**：单条 run 的 feature contribution / family contribution
- **决策层**：top1 vs top2 的 feature delta / family delta

viewer 侧目前还支持：

- 切换 `Top1 / Top2 / Top3`
- 在 verifier trajectory 图 hover 时显示该 run 是否正确

---

## 8. 与原始仓库材料的关系

这里的内容是对原始材料的 **专题整理与再打包**，主要来源包括：

- `docs/ES_SVD_MS_RR_R1.md`
- `docs/ES_SVD_MS_RR_R2_REPORT.md`
- `docs/ES_SVD_CODING_RR_R1.md`
- `docs/SVD_INTERPRETABILITY_R1_20260411.md`
- `docs/BESTOFN_ES_SVD_MS_RR_R1_SLOT100_20260411.md`
- `docs/SVD_PERF_PLAN_20260411.md`
- `docs/SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md`
- `docs/ES_SVD_MATH_RL_CHECKPOINT_RANKING.md`

原则是：

- 训练与结果原始 artifact 仍保留在原路径
- `SVDomain/` 提供论文写作与复现用的 curated entrypoint
- 不改 leaderboard 逻辑，不制造第二套 source of truth

---

## 9. 你现在该从哪里开始

### 如果你要写论文

先看：

- `SVDomain/docs/00_EXECUTIVE_SUMMARY.md`
- `SVDomain/docs/06_PAPER_OUTLINE.md`
- `SVDomain/results/comparison_tables.md`
- `docs/ES_SVD_MS_RR_R2_REPORT.md`（作为扩展实验对照）

### 如果你要复现实验

先看：

- `SVDomain/env/README.md`
- `SVDomain/docs/05_REPRODUCTION_CHECKLIST.md`
- `SVDomain/examples/reproduce_commands.sh`
- `SVDomain/SUBMISSION_GUIDE.md`

### 如果你要看 slot100 / coding 补充线

先看：

- `docs/SVD_PERF_PLAN_20260411.md`
- `docs/SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md`
- `SVDomain/train_slot100_svd_domain_r1.py`

### 如果你要解释模型

先看：

- `SVDomain/docs/04_INTERPRETABILITY_AND_VIEWER.md`
- `SVDomain/python_docs/VIEWER_AND_EXPLAIN_APIS.md`

---

## 10. 备注

- `r1` 仍是最适合整理成论文主线的 canonical family
- `r2` 主要承担 10-anchor / feature-family-search 的扩展对照角色
- `slot100` coding 线更适合作为 bridge、negative result 或补充实验
- `SVDomain/SUBMISSION_GUIDE.md` 可作为操作补充，但论文核心叙事建议优先以本目录 curated 文档为准
