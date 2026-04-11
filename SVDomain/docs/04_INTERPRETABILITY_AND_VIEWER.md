# Interpretability and Viewer

这份文档把 SVD explainability 线整理成论文可写的结构化说明。

---

## 1. 解释性目标

SVDomain 当前的解释性分为三层：

1. **模型层**：某个 `domain × anchor` 主要依赖哪些特征
2. **样本层**：某条 run 为什么得到这个分数
3. **决策层**：为什么 top1 胜过 top2

对应产物已经全部落地：

- explain core
- export script
- artifact
- API
- viewer dashboard

---

## 2. 解释核心结构

当前解释层针对 canonical models：

- `es_svd_math_rr_r1`
- `es_svd_science_rr_r1`
- `es_svd_ms_rr_r1`

支持的 anchor：

- `10 / 40 / 70 / 100`

支持的结构：

- `StandardScaler -> TruncatedSVD -> LogisticRegression`

支持的情况：

- `whiten = yes/no`
- 多 anchors
- domain-specific feature subsets

---

## 3. 回投影解释

解释数学上依赖一个关键事实：

> route 虽然经过 `StandardScaler + TruncatedSVD`，但最终 decision head 仍然是线性的，
> 因此可以把权重回投影回表示空间。

若记：

- `x_scaled = (x - mean) / scale`
- `z = SVD(x_scaled)`
- `beta = lr.coef_[0]`

则可以恢复：

- 表示空间有效权重
- 有效截距
- 单样本 feature contribution

对于 `raw+rank` 表示，还可以进一步折叠为：

- `raw_weight`
- `rank_weight`
- `signed_weight`
- `strength`

这让解释输出能够回到“原始特征名”层，而不是停在 latent dimension。

---

## 4. Family taxonomy

当前 family 至少包括：

- `confidence`
- `uncertainty`
- `self_cert_logprob`
- `trajectory`
- `availability_meta`
- `terminal_tail`

这套 taxonomy 现在同时服务于：

- model summary
- per-run contributions
- top1 vs top2 decision deltas

---

## 5. 三类解释产物

### 5.1 Domain × Anchor 总体解释

输出内容：

- top positive features
- top negative features
- family strength

用于回答：

- 某个 anchor 主要在看什么
- 不同 domain 的 route 偏好有什么差异

### 5.2 Top1 vs Top2 决策解释

输出内容：

- top1 分数
- top2 分数
- margin
- top feature deltas
- top family deltas

用于回答：

- 为什么某题里 top1 胜过 top2

### 5.3 Run-level 样本解释

输出内容：

- feature contributions
- family contributions
- reconstructed score
- reconstruction error

用于回答：

- 某条 run 为什么会高分 / 低分

---

## 6. 当前数值 sanity

当前 interpretability artifact 是 compact smoke export，但数值链路已经验证通过：

| Model | Problems | Problem×Anchor | Runs | Max Abs Error | Mean Abs Error |
|---|---:|---:|---:|---:|---:|
| `es_svd_math_rr_r1` | 7 | 28 | 1792 | `3.64e-14` | `1.29e-14` |
| `es_svd_science_rr_r1` | 2 | 8 | 512 | `9.77e-15` | `2.26e-15` |
| `es_svd_ms_rr_r1` | 9 | 36 | 2304 | `3.64e-14` | `1.06e-14` |

这说明：

- contribution sum 能数值重构原模型分数
- 当前解释链路不是 heuristic approximation，而是严格对齐的

---

## 7. 当前解释层初步结论

从已导出的解释结果中，可以看到几个稳定模式：

### 7.1 Math / `ms -> math`

- `trajectory` family 长期主导
- 到 `anchor=100` 时，`uncertainty` 权重明显抬升
- `traj_continuity` 常见于 top positive features
- `traj_novelty` 常见于 top negative features

### 7.2 Science / `ms -> science`

- `10 / 40 / 70` 时以 `trajectory` 为主
- 到 `100` 时 `uncertainty` 会反超 `trajectory`

### 7.3 Negative evidence

- `self_cert_logprob` 相关特征经常落在 top negative features 中
- 它们更像“拉低分”的证据，而不只是简单的正向置信度信号

---

## 8. Viewer 集成

当前 `cot_viewer` 已支持 canonical SVD explainability：

- `GET /api/svd/explain/model_summary`
- `GET /api/svd/explain/problem_top1_vs_top2`
- `GET /api/svd/explain/run_contributions`

前端已支持：

- 切换 `Method`
- 切换 `Anchor`
- 切换 `SVD Domain`
- 查看 top1 / top2 / top3
- trajectory 图 hover 显示该 run 是否正确

### 当前首页能回答的问题

1. 当前方法选了哪条 run？
2. 为什么它是 top1？
3. top1 比 top2 赢在哪些 family？
4. 当前 margin 是多少？
5. 当前选中的 top1 / top2 / top3 在 trajectory 图上是不是正确 run？

---

## 9. 当前 viewer 的使用建议

### 看总体模型偏好

切到：

- `Method = es_svd_math_rr_r1 / es_svd_science_rr_r1 / es_svd_ms_rr_r1`
- `Anchor = 10 / 40 / 70 / 100`
- `SVD Domain = math / science`

然后看：

- `Model Anchor Priors`
- `top positive / negative features`
- `family strength`

### 看单题决策

切到某个 problem 后看：

- `Decision`
- `Top Family Deltas`
- `Feature Panel`

### 看单条 run

点击：

- `Top1`
- `Top2`
- `Top3`

即可切换当前 run 的：

- per-run contributions
- token evidence
- trajectory 高亮

---

## 10. 论文里怎么写 interpretability 部分

建议不要把解释性写成“静态 feature importance 表格”，而是写成：

1. **Method-level decomposition**
2. **Instance-level contribution**
3. **Decision-level contrast**

这三层正好对应本项目的实际实现。

正文里最值得展示的图/表：

- `domain × anchor` family strength 图
- 一个 top1 vs top2 的 feature delta case study
- 一个 run contribution case study

appendix 可以再补：

- wrong-top1 cases
- failure mode summary
- per-anchor sanity statistics
