# Method and Modeling

这份文档把 SVDomain 的方法层叙事整理成论文友好的版本。

---

## 1. 任务定义

对每个 `problem`，系统会产生多条 candidate `runs`。  
每条 run 在生成过程中有多个 early-stop 位置可被观测。

我们要做的是：

- 给每条 run 一个分数
- 在同题 runs 中选出最优 run
- 尽量在较早位置就形成可靠判断

在实现上，这对应一个 **group-local ranking / selection** 问题。

---

## 2. 三层位置定义

### 2.1 `EARLY_STOP_POSITIONS`

```text
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

这是实际评测和 early-stop 决策关心的 10 个位置。

### 2.2 `EXTRACTION_POSITIONS`

```text
(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
```

这是特征提取时看的位置集合。  
它比 `EARLY_STOP_POSITIONS` 多出一些控制点，例如 `0.05` 和 `0.15`。

作用是：

- 帮助构造 trajectory / prefix / recency 类特征
- 让特征不被 10 个官方停点完全束缚

### 2.3 `ANCHOR_POSITIONS`

```text
(0.1, 0.4, 0.7, 1.0)
```

这 4 个位置是 canonical SVD family 实际训练的锚点。

映射关系：

- `20 / 30 -> 10`
- `50 / 60 -> 40`
- `80 / 90 -> 70`
- `100 -> 100`

因此 canonical family 的思想不是“10 个点都训练 10 套模型”，而是：

- 用少量 anchor 学一套稳定 route
- 再把中间 slot 映射到最近 anchor

---

## 3. 模型结构

每个 anchor route 使用的结构固定为：

```text
StandardScaler -> TruncatedSVD -> LogisticRegression
```

它的优点是：

- 足够轻量
- 低秩表示明确
- 权重可回投影
- 便于做 feature-level contribution

这正是 SVDomain 能把“效果”和“解释性”同时保住的关键。

---

## 4. 表示：`raw+rank`

每个原始 feature 在 route 里不是只出现一次，而是以两种形式出现：

- `raw`
- `rank`

也就是：

- 原始数值保留强度信息
- 组内 rank 保留相对排序信息

这在 early-stop selection 里尤其重要，因为很多信号更像：

- “绝对值很大”
- 或 “虽然绝对值不大，但在同题 runs 中显著更高/更低”

canonical family 故意只保留 `raw+rank`，不继续引入更复杂表示，以降低搜索空间和叙事复杂度。

---

## 5. 特征设计

当前 canonical `r1` 使用固定特征组：

- `token_plus_traj_fixed`

包含的特征主要有：

### 5.1 token-level families

- `tok_conf_*`
- `tok_gini_*`
- `tok_neg_entropy_*`
- `tok_selfcert_*`
- `tok_logprob_*`

### 5.2 trajectory families

- `traj_continuity`
- `traj_reflection_count`
- `traj_novelty`
- `traj_max_reflection`
- `traj_late_convergence`

### 5.3 availability/meta

- `has_tok_conf`
- `has_tok_gini`
- `has_tok_neg_entropy`
- `has_tok_selfcert`
- `has_tok_logprob`
- `has_rows_bank`

明确排除的数值 row / tail 特征包括：

- `nc_mean`
- `nc_slope`
- `self_similarity`
- `tail_q10`
- `head_tail_gap`
- `tail_variance`
- `last_event_tail_conf`
- `event_pre_post_delta`

这样做的目标是让 canonical family 尽量聚焦于：

- token uncertainty / confidence
- global trajectory shape

---

## 6. Domain split 设计

SVDomain 的核心结构不是“一个模型打天下”，而是：

- `math` 单域训练
- `science` 单域训练
- 再把两者组织为 `ms` 多域 canonical bundle

这有两个作用：

1. 避免不同域的统计模式相互污染
2. 让解释性结果更清楚地呈现 domain-specific behavior

在当前证据里，这个设计是有价值的：

- math 的收益更强
- science 更依赖 uncertainty / trajectory 的组合
- coding 明显不适合直接复用 noncoding canonical family

---

## 7. 搜索空间

canonical `r1` 的训练搜索空间并不大：

- `rank ∈ {2, 4, 6, 8, 12, 16}`
- `C ∈ {0.05, 0.10, 0.20, 0.50, 1.00}`
- `whiten ∈ {False, True}`
- `class_weight ∈ {"none", "balanced"}`

搜索原则：

- 保持模型容量有限
- 不引入复杂非线性
- 让 route summary 可以写成简单表格

这对于论文很友好，因为每个 anchor 都可以给出：

- 最优 rank
- 最优 C
- 是否 whiten
- 是否 balanced
- 对应 baseline 与 CV AUROC

---

## 8. 训练与验证协议

### 8.1 数据

- `MUI_HUB/cache`
- `MUI_HUB/cache_train`

### 8.2 切分

- `85 / 15` holdout
- 按 `dataset + problem_id` 划分
- 跨 root 一致
- `split_seed = 42`

### 8.3 非目标

canonical `r1` 刻意不做：

- 不训练大模型
- 不引入复杂 reranker
- 不把 baseline route 混进最终 canonical bundle
- 不把 coding 逻辑强行塞回 noncoding 主线

---

## 9. 为什么这套方法适合论文

### 9.1 足够清楚

你可以很自然地把 method section 写成：

1. feature extraction positions
2. anchor routing
3. raw+rank representation
4. SVD low-rank projection
5. logistic decision head

### 9.2 足够可解释

因为结构是线性的，所以能回投影到原始特征空间，得到：

- effective weight
- feature contribution
- family contribution
- top1 vs top2 delta

### 9.3 足够可复现

训练脚本、结果 artifact、viewer API、解释导出都已经存在，并且在本目录中有统一索引。

---

## 10. 建议在正文中强调的 method takeaways

如果你要写论文的 method section，建议强调：

1. **Four-anchor training is a deliberate compression design**：用 4 个稳定锚点覆盖 10 个 official slots。
2. **Raw+rank is enough for a strong linear selector**：不需要复杂表示也能得到稳定收益。
3. **Domain-aware factorization matters**：math / science / coding 的统计结构不一样。
4. **Interpretability is native, not post-hoc only**：当前 explainability 直接建立在线性回投影上。
