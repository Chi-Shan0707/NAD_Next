# 18. SVD Feature Complexity Study

这份文档定义一套**三层受控实验**，专门回答下面这个问题：

> 现在 `no-SVD` 已经接近 `SVD`，是不是因为当前 canonical feature bank 被挑得过于干净；一旦特征数增多、语义变宽、冗余和噪声变多，`SVD` 的低秩压缩和正则化价值才会系统性显现？

实现入口：

- `SVDomain/experiments/run_svd_feature_complexity_study.py`

自动生成的结果 note：

- `docs/18_SVD_FEATURE_COMPLEXITY_RESULTS.md`

---

## 1. 设计原则

- **比较对象固定**：只比较 `StandardScaler + LogisticRegression`（`no-SVD`）与 `StandardScaler + TruncatedSVD + LogisticRegression`（`SVD`）。
- **其他协议不变**：继续使用当前仓库 EarlyStop 主线的 grouped holdout、四锚点 `10/40/70/100`、`raw+rank` 表示、相同的 EarlyStop 汇总指标。
- **主结论双指标并列**：同时报告 `AUC of AUROC` 与 `AUC of SelAcc`；论文主线默认按前者组织。
- **种子重复**：默认多 seed 重复，减少“某个 seed 恰好有利于 SVD / no-SVD”的争议。
- **三层问题分离**：
  - 第一层问：在干净、小而可解释的 curated bundle 上，SVD 到底多强。
  - 第二层问：随着真实 observation language 变宽，SVD 是否开始更重要。
  - 第三层问：当 feature bank 被污染后，SVD 是否真的更像一种压缩 / 正则化器。

---

## 2. 三层实验

### Layer 1 — Canonical curated bundles

这一层只做**越来越接近 paper 主线**的干净 bundle。

| Condition | 特征数 | 定义 |
|---|---:|---|
| `trajectory_only` | 5 | `traj_*` 五个 trajectory 特征 |
| `uncertainty_only` | 7 | `tok_gini_*` + `tok_neg_entropy_*` + `tok_selfcert_*` |
| `token_only` | 11 | 全部 token-level 特征，不含 trajectory / availability |
| `token_plus_trajectory` | 16 | `token_only` + 五个 `traj_*` |
| `canonical_22` | 22 | `token + trajectory + availability flags` |

这一层回答：

- 在一个**小、干净、几乎不含冗余**的特征集合上，`no-SVD` 到底离 `SVD` 有多远？
- 如果这一层里二者差距很小，那么“low-rank 本身的价值”应当被理解为**温和但一致**，而不是巨大跃迁。

### Layer 2 — Real upstream expansion

这一层从 `canonical_22` 出发，只加入**仓库里已经可抽取、但 canonical 主线刻意没纳入**的真实上游特征。

按语义家族逐步扩展：

| Condition | 特征数 | 新加入特征 |
|---|---:|---|
| `canonical_plus_neuron_adjacent` | 25 | `nc_mean`, `nc_slope`, `self_similarity` |
| `canonical_plus_prefix_tail` | 28 | 上一行 + `tail_q10`, `head_tail_gap`, `tail_variance` |
| `canonical_plus_event_local` | 30 | 上一行 + `last_event_tail_conf`, `event_pre_post_delta` |

这一层回答：

- 当 observation language 从“精选 22 维”走向“真实 30 维上游 bank”时，`SVD` 的相对价值是否增大？
- 这种增大是所有 domain 都一致，还是主要发生在更难、更噪的 domain（尤其 `coding`）？

### Layer 3 — Noise / decoy control

这一层从最宽的真实 bank（`30` features）出发，再刻意加入三类 decoy：

| Family | 机制 | 作用 |
|---|---|---|
| `permutation` | 在 problem 内打乱真实特征值 | 保留边际分布，破坏语义对齐 |
| `duplicate` | 复制已有特征并加噪 | 制造冗余、共线、近重复特征 |
| `random` | 加入匹配尺度的随机控制特征 | 人为增加无信息维度 |

每个 family 都做三档剂量：

- `low`：新增特征数约为 base bank 的 `0.5×`
- `med`：新增特征数约为 base bank 的 `1.0×`
- `high`：新增特征数约为 base bank 的 `2.0×`

这一层回答：

- feature bank 一旦被污染，`no-SVD` 是否退化更快？
- `SVD` 的收益是否开始更像**压缩 / 正则化收益**，而不只是“另一个线性头”？

---

## 3. 横向特征数实验

核心输出不是只看单点，而是看**feature count 横轴**。

### Clean sweep

把 Layer 1 + Layer 2 串起来，形成：

`5 -> 7 -> 11 -> 16 -> 22 -> 25 -> 28 -> 30`

对每个点都同时画：

- `no-SVD` 曲线
- `best fixed-rank SVD` 曲线
- `Δ(best-SVD − no-SVD)` 曲线

这条曲线是主证明：

- 如果 `22` 维之前差距很小，而 `25/28/30` 之后差距拉开，那么就能直接支持：
  - **现在 no-SVD 接近 SVD，是因为 canonical bank 已经非常干净**
  - **一旦真实 feature language 变宽，SVD 就开始更需要**

### Noise sweep

在 Layer 3 中，横轴可以理解为：

- `30 + 15`
- `30 + 30`
- `30 + 60`

分别对应 `low / med / high` 污染强度。

这条曲线不是主结果，但它负责补上机制解释：

- SVD 在**被污染的高维 feature bank** 上更稳，不是偶然，而是因为它在做压缩和抗噪。

---

## 4. 评估与默认设置

- **Domains**：`math, science, coding`
- **Anchors**：`10, 40, 70, 100`
- **Representation**：`raw+rank`
- **Holdout**：按 `dataset + problem_id` 切分，默认 `85/15`
- **CV**：grouped CV，默认 `n_splits=3`
- **Seeds**：默认 `42, 43, 44`
- **SVD ranks**：默认 `4, 8, 12, 16, 24`
- **结果汇总**：
  - 单 model 明细：每个 seed、每个 domain、每个 condition、每个 rank
  - `best_svd`: 在固定 rank sweep 中按 holdout `AUC of AUROC` 选择最优 SVD
  - 对比表：`no-SVD` vs `best-SVD`
  - macro 平均：`math / science / coding` 三域简单平均

---

## 5. 产物

脚本默认会写出：

- `results/scans/feature_complexity/three_layer_summary.json`
- `results/tables/feature_complexity_conditions.csv`
- `results/tables/feature_complexity_model_rows.csv`
- `results/tables/feature_complexity_aggregate_rows.csv`
- `results/tables/feature_complexity_comparison.csv`
- `results/tables/feature_complexity_clean_sweep.csv`
- `results/tables/feature_complexity_noise_robustness.csv`
- `results/figures/feature_complexity/*.png`
- `docs/18_SVD_FEATURE_COMPLEXITY_RESULTS.md`

其中最关键的是：

- `feature_complexity_comparison.csv`：每个 condition 的 `no-SVD` vs `best-SVD`
- `feature_complexity_clean_sweep.csv`：Layer 1 + Layer 2 的横向特征数主表
- `feature_complexity_noise_robustness.csv`：Layer 3 的 decoy robustness 主表

---

## 6. 复现命令

### Full run

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 SVDomain/experiments/run_svd_feature_complexity_study.py \
  --domains math,science,coding \
  --layers canonical,expansion,noise \
  --seeds 42,43,44 \
  --svd-ranks 4,8,12,16,24 \
  --feature-workers 8 \
  --fit-workers 12
```

### Smoke run

```bash
python3 SVDomain/experiments/run_svd_feature_complexity_study.py \
  --smoke \
  --domains math,science,coding \
  --layers canonical,expansion,noise \
  --seeds 42,43 \
  --svd-ranks 8,12
```

---

## 7. 如何解释结果

最理想、也最符合这个设计初衷的结果模式是：

1. **Layer 1**：`no-SVD` 与 `SVD` 差距很小，但 SVD 仍有轻微正增益  
2. **Layer 2**：随着真实特征扩展到 `25/28/30` 维，SVD 增益明显放大  
3. **Layer 3**：在 decoy 污染下，`no-SVD` 退化更快，SVD 增益继续扩大  

如果看到这个模式，就可以用一句非常清楚的话总结：

> `no-SVD` 之所以在 canonical 路线上已经接近 `SVD`，主要是因为当前 feature bank 足够干净；当特征语言变宽、冗余变多、噪声变强时，SVD 的低秩压缩与正则化才真正显示出必要性。
