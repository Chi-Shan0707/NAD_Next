# 18. SVD Feature Complexity Results

这份结果 note 汇报三层 `SVD vs no-SVD` 特征复杂度实验的**正式全量结果**。核心问题是：

> 现在 `no-SVD` 已经接近 `SVD`，是不是因为 canonical feature bank 很干净；一旦特征语言变宽、变脏、变冗余，就需要 SVD？

这次全量实验的结论是：

> **在当前 EarlyStop 协议下，这个命题没有被支持。**
>
> `canonical_22` 上 `SVD` 只比 `no-SVD` 略好一点；加入真实上游扩展特征后，macro 平均反而变成 `no-SVD` 更强；再加入 decoy 噪声后，`SVD` 在所有 9 个噪声条件上都落后于 `no-SVD`。

但结果也不是“`SVD` 完全没用”：

> `SVD` 在**更瘦、更 token-centric 的 bundle** 上仍然有局部优势，最明显的是 `coding / token_only`，`ΔAUC=+8.09` AUC-pts。

所以更准确的表述是：

> **当前证据支持“`SVD` 对部分精瘦 token bundle 有帮助”，但不支持“特征一复杂起来就必须上 SVD”。**

---
第一层：canonical curated bundles

按“越来越接近 paper 主线”的方式扩展：

trajectory-only
uncertainty-only
token-only
token + trajectory
canonical 22 features

这层回答：
在精心设计的小而干净特征集上，no-SVD 到底和 SVD 差多少？

第二层：real upstream expansion

你 appendix 已经写了，上游其实还支持更多 neuron-adjacent / meta / prefix-local 特征，比如：

nc_mean
nc_slope
self_similarity
tail_q10
head_tail_gap
tail_variance
last_event_tail_conf
event_pre_post_delta

而 canonical route 目前故意把这些排除了。

所以第二层应该是真实地把这些可抽取但未纳入 canonical 的特征逐步加进来。
这层回答：

随着 observation language 从“精炼”走向“更广”，SVD 是否开始比 no-SVD 更有价值？

第三层：noise / decoy control

最后再加一组严格控制的 decoy：

permutation features
duplicated noisy variants
useless random controls
## 1. Protocol Snapshot

- `domains`: `math, science, coding`
- `layers`: `canonical, expansion, noise`
- `seeds`: `42, 43, 44`
- `svd ranks`: `4, 8, 12, 16, 24`
- `anchors`: `10, 40, 70, 100`
- `representation`: `raw+rank`
- `holdout split`: `0.15` with `split_seed=42`
- `grouped CV`: `n_splits=3`
- `feature_workers`: `8`
- `fit_workers`: `12`
- `run mode`: `full`
- `full wall time`: `614.77s`

数据与实现细节见：

- `docs/18_SVD_FEATURE_COMPLEXITY.md`
- `SVDomain/experiments/run_svd_feature_complexity_study.py`

---

## 2. Short Answer

### 2.1 对“横向特征数实验能否证明 SVD 更必要？”的直接回答

**不能，至少这次正式实验不能。**

最关键的三行是：

| Condition | #feat | macro no-SVD AUC | macro best-SVD AUC | Δ(best-SVD - no-SVD) |
|---|---:|---:|---:|---:|
| `canonical_22` | 22 | 75.74% | 76.05% | +0.31% |
| `canonical_plus_event_local` | 30 | 77.56% | 76.98% | -0.58% |
| `random_med` | 60 | 77.72% | 69.68% | -8.05% |

也就是说：

- 在 paper-facing `canonical_22` 上，`SVD` 只是**轻微领先**；
- 扩展到真实 30-feature upstream bank 之后，macro 上**不是 SVD 变更必要，而是 no-SVD 反超**；
- 加入显式 decoy 以后，差距进一步往 `no-SVD` 一边扩大。

### 2.2 这次结果真正支持什么

这次结果更支持下面这句话：

> `SVD` 的优势更像是**局部的、bundle-specific 的**：它在某些更瘦、更 token-centric 的表示上有帮助；但在当前这套 feature language、固定 rank sweep 和线性头设置下，它不是“特征一复杂就更优”的统一答案。

---

## 3. Macro Clean Sweep

这是 Layer 1 + Layer 2 的主表，也是“特征数横轴”实验最核心的结果。

| Condition | #feat | no-SVD AUC | best-SVD AUC | ΔAUC | no-SVD SelAcc | best-SVD SelAcc | ΔSelAcc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `trajectory_only` | 5 | 75.39% | 72.86% | -2.53% | 83.20% | 81.78% | -1.42% |
| `uncertainty_only` | 7 | 67.50% | 67.80% | +0.31% | 75.87% | 76.15% | +0.28% |
| `token_only` | 11 | 67.24% | 70.21% | +2.96% | 74.83% | 79.51% | +4.68% |
| `token_plus_trajectory` | 16 | 75.73% | 76.15% | +0.42% | 82.37% | 82.17% | -0.20% |
| `canonical_22` | 22 | 75.74% | 76.05% | +0.31% | 81.92% | 82.00% | +0.08% |
| `canonical_plus_neuron_adjacent` | 25 | 77.57% | 76.98% | -0.58% | 88.67% | 88.24% | -0.43% |
| `canonical_plus_prefix_tail` | 28 | 77.61% | 76.98% | -0.63% | 88.63% | 88.22% | -0.41% |
| `canonical_plus_event_local` | 30 | 77.56% | 76.98% | -0.58% | 88.76% | 88.22% | -0.53% |

### 3.1 这张表说明了什么

- `canonical_22` 的确体现了“**两者很接近**”：`SVD` 只领先 `+0.31` AUC-pts。
- 但真正把特征从 `22 -> 25 -> 28 -> 30` 扩开以后，宏观走势不是“`SVD` 变强”，而是**`no-SVD` 更好地吸收了这些新增真实特征**。
- 从 `25` 维开始，macro `ΔAUC` 连续三档为负：`-0.58 / -0.63 / -0.58`。

换句话说：

> **“canonical 很干净，所以 no-SVD 才显得接近；一旦扩到真实 upstream 特征，SVD 就会变得更重要”**——这句话没有在 macro 结果里成立。

### 3.2 Layer 1 仍然给了 SVD 一个局部胜场

Layer 1 里最能帮 `SVD` 的 bundle 是：

- `token_only`: macro `ΔAUC=+2.96`
- `uncertainty_only`: macro `ΔAUC=+0.31`
- `token_plus_trajectory`: macro `ΔAUC=+0.42`

最强正例不是 `canonical_22`，而是更瘦的 `token_only`。

这说明：

> `SVD` 的当前强项更像是“**在更稀、更纯的 token 统计子空间里做低秩压缩**”，而不是“随着 feature bank 变宽自动变得更必要”。

---

## 4. Domain Breakdown

为了回答“是不是只有某个更难 domain 才需要 SVD”，最有信息量的是下面这张压缩表：

| Domain | `canonical_22` ΔAUC | `canonical_plus_event_local` ΔAUC | `random_med` ΔAUC | 总结 |
|---|---:|---:|---:|---|
| `math` | +0.02% | -0.04% | -17.37% | canonical 近乎打平；扩展后不占优；强噪声下明显输给 no-SVD |
| `science` | +0.19% | -2.10% | -2.72% | canonical 略优；真实扩展与噪声条件都转负 |
| `coding` | +0.73% | +0.40% | -4.04% | clean / real-upstream 下仍有小幅正增益；噪声条件全部转负 |
| `macro` | +0.31% | -0.58% | -8.05% | 总体不支持“更复杂就更需要 SVD” |

### 4.1 `coding` 是唯一部分支持原假设的域

`coding` 上：

- `token_only`: `+8.09` AUC-pts
- `canonical_22`: `+0.73`
- `canonical_plus_neuron_adjacent`: `+0.40`
- `canonical_plus_prefix_tail`: `+0.23`
- `canonical_plus_event_local`: `+0.40`

所以如果一定要找“特征变宽后 SVD 仍有价值”的证据，**只能在 coding 找到弱证据**。

但这证据不够撑起总体结论，因为：

- `math` 的 25/28/30 三档都转负；
- `science` 的 25/28/30 三档也都转负；
- macro 平均因此明确转负。

### 4.2 sign-count 视角更直观

按 `ΔAUC > 0` 计数：

| Domain | Canonical 层 | Expansion 层 | Noise 层 |
|---|---:|---:|---:|
| `math` | 4 / 5 | 0 / 3 | 0 / 9 |
| `science` | 2 / 5 | 0 / 3 | 0 / 9 |
| `coding` | 4 / 5 | 3 / 3 | 0 / 9 |
| `macro` | 4 / 5 | 0 / 3 | 0 / 9 |

最关键的是最后两列：

- **Expansion：macro 0 / 3 正增益**
- **Noise：macro 0 / 9 正增益**

---

## 5. Noise Robustness

如果原命题成立，Layer 3 理论上应该最支持 `SVD`。但正式结果恰好相反。

| Family | Dose | #feat | no-SVD AUC | best-SVD AUC | ΔAUC |
|---|---|---:|---:|---:|---:|
| `duplicate` | low | 45 | 77.55% | 75.55% | -1.99% |
| `duplicate` | med | 60 | 77.74% | 74.85% | -2.89% |
| `duplicate` | high | 90 | 77.26% | 74.47% | -2.79% |
| `permutation` | low | 45 | 77.89% | 75.00% | -2.89% |
| `permutation` | med | 60 | 78.10% | 75.15% | -2.95% |
| `permutation` | high | 90 | 78.52% | 75.36% | -3.16% |
| `random` | low | 45 | 77.21% | 73.64% | -3.57% |
| `random` | med | 60 | 77.72% | 69.68% | -8.05% |
| `random` | high | 90 | 76.74% | 69.66% | -7.09% |

### 5.1 这层的结论非常明确

- macro 上 **9 / 9** 个噪声条件全部是负增益；
- 最接近打平的是 `duplicate / low`，也仍然是 `-1.99`；
- 最差的是 `random / med`，`-8.05`。

因此：

> 这组 decoy control **没有证明 SVD 提供更强的抗污染正则化**；在当前实现里，反而是 `no-SVD` 更稳。

这点在三个 domain 上也成立：`math / science / coding` 的噪声层都是 `0 / 9` 正增益。

---

## 6. What This Means

### 6.1 被支持的结论

- `canonical_22` 上，`no-SVD` 和 `SVD` 的确已经很接近；
- `SVD` 对某些**精瘦 token bundle** 有明显帮助，尤其是 `token_only`；
- `coding` 对 `SVD` 更友好，这和此前对 code domain 的直觉并不冲突。

### 6.2 没被支持的结论

下面这句**没有被正式结果支持**：

> “现在 no-SVD 接近 SVD，是因为特征已经被挑得很干净；特征一复杂起来，就需要 SVD。”

当前证据更像是：

- `no-SVD` 的 `StandardScaler + LogisticRegression` 在这套 feature bank 上已经有足够强的线性判别与正则化能力；
- 加入真实上游特征时，新增维度带来的有效信息比低秩压缩带来的收益更大；
- 加入 decoy 后，固定 rank 的 `SVD` 更像是在**过压缩**，而不是在“去噪后保留主信号”。

### 6.3 最可能的解释

比较保守、同时最符合现象的解释是：

1. **当前 feature bank 仍然高度语义对齐**  
   即使加到 30 real features，也不是一个真正失控的 feature soup。

2. **`no-SVD` 的线性头已经足够强**  
   标准化 + `LogisticRegression` 在这里并不脆弱，能直接把有用特征权重提上去、把噪声权重压下去。

3. **当前 fixed-rank sweep 可能对大 bank 偏激进**  
   `raw+rank` 会把 `30 / 45 / 90` 个原始特征扩成更高维表示，而 rank sweep 仍只到 `24`。这让 `SVD` 在更宽 bank 上更像“压掉信息”，而不是“只压掉噪声”。

不过要注意：

- 这不能单独解释全部现象，因为很多噪声条件的 `best rank` 并没有都顶到 `24`；
- 换句话说，**“rank cap 太低”是合理怀疑，但不是已经证实的唯一原因。**

---

## 7. Quality Checks

- `fallback_anchor_count`: `0 / 864` model rows  
  这次正式结果没有触发缺失 anchor 兜底，因此不是 routing artifact。
- 产物完整写出：
  - `results/scans/feature_complexity/three_layer_summary.json`
  - `results/tables/feature_complexity_conditions.csv`
  - `results/tables/feature_complexity_model_rows.csv`
  - `results/tables/feature_complexity_aggregate_rows.csv`
  - `results/tables/feature_complexity_comparison.csv`
  - `results/tables/feature_complexity_clean_sweep.csv`
  - `results/tables/feature_complexity_noise_robustness.csv`
  - `results/figures/feature_complexity/clean_feature_curves.png`
  - `results/figures/feature_complexity/clean_feature_gain.png`
  - `results/figures/feature_complexity/noise_robustness.png`

---

## 8. Takeaway for the Paper

如果现在就把这组实验写进论文，最稳妥的说法不是“证明 SVD 更优”，而是：

> 在当前 EarlyStop 线性路由设定下，`SVD` 对精瘦 token-centric bundle 仍有帮助，但并没有随着 feature bank 扩展或 decoy 污染而系统性变得更优。`canonical_22` 上的接近更多说明两者在当前主线 bank 上已经近似，而不是说明 SVD 在更宽更脏的 bank 上必然占优。

如果还想继续追“为什么 SVD 没有在更脏 bank 上赢”，最有价值的后续不是再重复同一结论，而是做下面两类补充：

1. **Adaptive rank sweep**  
   对 `30 / 45 / 90` feature banks 把 rank 扩到 `32 / 48 / 64`，检查当前负结果是否主要来自过低 rank。

2. **更真实的 correlated nuisance**  
   现在的 decoy 大多是“易被线性头忽略”的无关维度；可以进一步构造与标签弱相关、与真特征共线的 nuisance，测试 SVD 是否在那种污染下才真正占优。

---

## 9. Reproduction

```bash
bash cookbook/00_setup/verify.sh

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
