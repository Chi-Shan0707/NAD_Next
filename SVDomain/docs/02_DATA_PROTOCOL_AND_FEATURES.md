# Data, Protocol, and Features

这份文档把与数据、特征、评估协议相关的内容整理成论文可直接引用的说明。

---

## 1. 数据来源

当前 canonical `r1` 使用两个 cache root：

- `MUI_HUB/cache`
- `MUI_HUB/cache_train`

这两个 root 一起构成训练与 holdout 评估的数据池。

---

## 2. 域划分

当前主要域是：

- `math`
- `science`
- `coding`

其中：

- `math` 主要对应 `aime24 / aime25 / brumo25 / hmmt25`
- `science` 主要对应 `gpqa`
- `coding` 主要对应 `lcb_v5`

对于论文正文，推荐把 `math` 和 `science` 作为主实验域，`coding` 作为边界案例。

---

## 3. 切分协议

### 3.1 Holdout split

- 训练 / holdout 比例：`85 / 15`
- 切分单元：`dataset + problem_id`
- seed：`42`

### 3.2 为什么按 `dataset + problem_id` 切分

这样做的理由是：

- 避免同一道题在不同 root 中泄漏到 train 和 holdout 两边
- 保证组内比较仍然是 problem-local
- 使得离线验证更接近 blind 测试的泛化场景

---

## 4. 位置协议

### 4.1 官方早停位置

```text
10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
```

### 4.2 特征提取位置

```text
5%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
```

### 4.3 训练锚点

```text
10%, 40%, 70%, 100%
```

这三层位置的分工是：

- 官方停点：定义实际评测接口
- 提取位置：定义可用信号
- 训练锚点：定义 canonical route

---

## 5. 特征家族

### 5.1 `confidence`

- `tok_conf_*`

含义：

- 模型对当前输出的置信强度

### 5.2 `uncertainty`

- `tok_gini_*`
- `tok_neg_entropy_*`

含义：

- 模型是否处在更不稳定、更分散、或更 uncertain 的状态

### 5.3 `self_cert_logprob`

- `tok_selfcert_*`
- `tok_logprob_*`

含义：

- 模型内部自证与 token log-prob 相关证据

### 5.4 `trajectory`

- `traj_continuity`
- `traj_reflection_count`
- `traj_novelty`
- `traj_max_reflection`
- `traj_late_convergence`

含义：

- 一条 run 的生成轨迹是否稳定、是否反复修改、是否后期才收敛

### 5.5 `availability_meta`

- `has_*`
- 当前也兼容未来 `nc_* / self_similarity` 扩展位

含义：

- 某些底层信号是否可用

### 5.6 `terminal_tail`

当前 canonical `r1` 主模型没有使用这一组，但解释层预留了兼容位：

- `tail_q10`
- `head_tail_gap`
- `tail_variance`
- `last_event_tail_conf`
- `event_pre_post_delta`

这方便未来把 slot100 / tail-sensitive family 纳入统一解释体系。

---

## 6. `raw+rank` 的意义

每个 feature 不只保留：

- 原始值 `raw`

还保留：

- 组内排序 `rank`

这意味着模型既能利用：

- absolute magnitude
- relative ordering within a problem

对 best-of-N 任务来说，这两个信息都很关键。

---

## 7. 当前 canonical `r1` 的特征选择原则

可以概括为三句话：

1. **少而稳定**
2. **尽量跨域共享 noncoding 主体**
3. **不要让复杂 row / tail 特征掩盖主线**

因此 canonical `r1` 刻意固定为：

- token uncertainty + confidence
- trajectory shape
- availability flags

而不是做大规模 feature search。

---

## 8. 评估指标

在 EarlyStop 主线里，最常看的指标包括：

- `AUC of AUROC`
- `AUC of SelAcc`
- `Earliest > 0.6`
- `AUROC@100%`
- `Stop Acc@100%`

在 Best-of-N 任务中，则主要看：

- `Average Rank`
- `AUROC`
- `Hit@1`
- `Hit@3`
- `SelAcc@10%`
- `Pairwise Acc`

在 RL checkpoint ranking side task 中，则主要看：

- `Spearman ρ`
- `Pearson r`
- `Kendall τ`
- `Top-1 hit`
- `Top-3 hit`

---

## 9. 为什么 coding 没有纳入主 canonical bundle

当前 `r1` 的一个重要设计决策是：

- **不让 coding 路由污染 noncoding canonical bundle**

原因包括：

1. `cache_train` 没有 coding cache，监督证据不足
2. noncoding 的 `token_plus_traj_fixed` 结构不一定适合 coding
3. 如果强行混入，会让主线解释变得混乱

因此当前更稳妥的做法是：

- noncoding 主线保持干净
- coding 单独训练、单独评估、单独报告

---

## 10. 写进论文时建议怎么说

建议在论文的 data/protocol section 里突出：

- 我们没有依赖更复杂的数据清洗或数据增强
- 我们采用 problem-consistent split，防止跨 root 泄漏
- 我们用少量 canonical anchors 覆盖整个 early-stop 轴
- 我们故意保持特征组简洁，以获得更强可解释性和可复现性
