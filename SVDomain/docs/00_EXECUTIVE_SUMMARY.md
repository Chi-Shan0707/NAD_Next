# Executive Summary

这份文档是 `SVDomain/` 的一页式摘要，适合作为：

- 论文写作前的快速总览
- 开题 / 周报 / 项目复盘的开场材料
- paper introduction / contributions / discussion 的提纲底稿

---

## 1. 研究问题

我们关注的是一个 **early-stop best-of-N selection** 问题：

- 对同一道题有多条 candidate runs
- 每条 run 在多个 early-stop 位置都能提取 token / trajectory 类信号
- 我们希望在不等到完整最终输出的情况下，尽早判断哪条 run 更值得选

SVDomain 的核心问题不是“训练一个更大的模型”，而是：

1. 用一组稳定、低复杂度的特征表示 run
2. 在多个 early-stop anchor 上学习一个可泛化的选择器
3. 让该选择器具备 **可解释性**

---

## 2. 核心方法

当前 canonical family 采用：

- `StandardScaler`
- `TruncatedSVD`
- `LogisticRegression`

输入表示固定为：

- `raw+rank`

训练 anchor 固定为：

- `10 / 40 / 70 / 100`

域划分固定为：

- `math`
- `science`
- `coding`

主线方法是：

- math 单域：`es_svd_math_rr_r1`
- science 单域：`es_svd_science_rr_r1`
- noncoding 组合：`es_svd_ms_rr_r1`

---

## 3. 为什么这条线值得写进论文

### 3.1 方法足够简单

它不是复杂的深层 reranker，而是：

- 小特征集合
- 低秩线性结构
- 多 anchor 训练
- 显式 domain split

这让它非常适合写成：

- 一个清楚的 method section
- 一个可解释的 ablation story
- 一个可复现的 paper artifact

### 3.2 结果上有明确收益

`es_svd_ms_rr_r1__coding_from_round1c` 在 blind leaderboard 上给出：

- `primary score = 3.8125`
- `auc_of_auroc = 0.7428`
- `auc_of_selacc = 0.8317`

对比 previous best `earlystop_prefix10_svd_round1`：

- `primary score`：`4.0000 -> 3.8125`
- `auc_of_auroc`：`0.7379 -> 0.7428`
- `auc_of_selacc`：`0.8311 -> 0.8317`

也就是说，这条线不是只在离线 holdout 上看起来好看，而是在线 blind 榜上也有实际收益。

### 3.3 解释性已经补齐

当前已经可以回答三类问题：

1. **模型层**：这个 domain / anchor 主要依赖哪些特征？
2. **样本层**：某条 run 为什么得这个分？
3. **决策层**：为什么 top1 赢过 top2？

这使得 SVDomain 不只是“一个分数更好的 selector”，而是“一个可解释的 selector family”。

---

## 4. 当前最稳的论文叙事

建议把论文主叙事写成：

> 我们提出一个面向 early-stop selection 的 domain-aware low-rank linear routing family，
> 使用固定的 raw+rank 表示和四个 canonical anchors（10/40/70/100），
> 在保持模型简单、可解释、可复现的同时，
> 在 blind leaderboard 上优于旧版 early-stop SVD 主提交。

然后正文里依次讲：

1. 问题设置
2. 表示与特征
3. domain split + four-anchor routing
4. 结果
5. 解释性

---

## 5. 当前最重要的实验结论

### 5.1 Math

`es_svd_math_rr_r1` 在 math holdout 上：

- `AUC of AUROC = 95.81%`
- `AUC of SelAcc = 99.73%`

相对旧 `earlystop_prefix10_svd_round1`：

- `93.48% -> 95.81%`

### 5.2 Science

`es_svd_science_rr_r1` 在 science holdout 上：

- `AUC of AUROC = 79.85%`
- `AUC of SelAcc = 98.80%`

science 的收益没有 math 那么强，但仍给出更好的 selective-accuracy 曲线与更干净的 canonical 结构。

### 5.3 Combined noncoding

`es_svd_ms_rr_r1` 在 combined noncoding holdout 上：

- `AUC of AUROC = 92.26%`
- `AUC of SelAcc = 99.52%`

相对旧 `earlystop_prefix10_svd_round1`：

- `89.88% -> 92.26%`

### 5.4 Coding 是负结果

`es_svd_coding_rr_r1` 没有赢过：

- `tok_conf_prefix_mean_v1`
- `earlystop_prefix10_svd_round1`

这个结果并不尴尬，反而说明：

- 当前 canonical noncoding feature set 对 coding 并不充分
- coding 域需要单独的 inductive bias
- 这能自然引出 discussion / future work

---

## 6. 关于 slot100 Best-of-N 的结论

从 `es_svd_ms_rr_r1__coding_from_round1c` 直接抽 `100%` 槽位得到：

- `es_svd_ms_rr_r1__coding_from_round1c__slot100`

这个版本：

- 是合法、可复现的 best-of-n 提交
- 但不是当前 best

其主要结论是：

- `SelAcc@10%` 略有优势
- 但 `Hit@1 / Hit@3 / Pairwise Acc` 更弱

因此它更适合作为：

- bridge baseline
- pure slot100 extraction baseline
- patching / reranking 的底座

而不是正文里的主结果。

---

## 7. 关于 interpretability 的主结论

目前 smoke artifact 与 viewer 已显示出比较稳定的模式：

- `math` 与 `ms -> math` 主要由 `trajectory` family 主导
- 到 `anchor=100` 时，`uncertainty` family 权重会明显抬升
- `science` 在 `10/40/70` 主要依赖 `trajectory`
- 到 `100` 时，`uncertainty` 会反超 `trajectory`
- `self_cert_logprob` 经常出现在 top negative features 中，往往表现为“拉低分”的证据

这让论文可以不仅说“模型有效”，还可以说“模型到底在看什么”。

---

## 8. 论文里该怎么定位 coding 分支

建议不要把 `es_svd_coding_rr_r1` 写成主贡献，而是写成：

- **negative result**
- **boundary experiment**
- **evidence that domain-specific structure matters**

这样 coding 的失败不会伤害主线，反而强化“domain-aware design”的必要性。

---

## 9. 最终建议

如果你现在要发论文，最稳的组织方式是：

- 正文主线：`es_svd_math_rr_r1` + `es_svd_science_rr_r1` + `es_svd_ms_rr_r1`
- 在线主结果：`es_svd_ms_rr_r1__coding_from_round1c`
- 解释性章节：viewer + SVD explain core
- appendix：
  - coding negative result
  - best-of-n slot100 baseline
  - checkpoint ranking side task

这会让整篇 paper 结构清楚、叙事稳定、证据链完整。
