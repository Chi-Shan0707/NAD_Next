# SVD Introspection Results — es_svd_ms_rr_r1

> Domains: math, science | Anchors: 10%, 40%, 70%, 100%
> Routes: 20 | Checked-in wrong-top1 cases: 0
> 本文档是在自动导出骨架基础上补充的中文长版解读，保留 `es_svd_ms_rr_r1` introspection 报告中的具体数字，并将其落档到主实验仓库。

---

## 模型结构摘要

当前 `es_svd_ms_rr_r1` 的每条 route 结构一致：

- `SVD rank = 16`
- `representation = raw+rank`
- `44` 个表示维度（`22` 个原始 feature × `2` 个 channel）
- `16` 个 latent components

训练集交叉验证上的 `CV-AUROC` 随 anchor 的变化如下：

| Domain | @10% | @40% | @70% | @100% |
|--------|------:|------:|------:|-------:|
| math | 0.895 | 0.935 | 0.945 | 0.957 |
| science | 0.713 | 0.766 | 0.764 | 0.778 |

`math` 在 `10%` 时就已经很可分（`0.895`），明显强于 `science` 的 `0.713`。`science` 从 `40%` 到 `70%` 基本是平台期，到 `100%` 才只有小幅提升。这意味着：模型在 `science` 上直到回答接近完成时，才真正获得足够稳定的判别信号。

---

## Q1：模型实际上在看什么？

按 family 汇总后的权重强度（对所有 rep-features 的 `|w_orig|` 求和，再按 anchor 取均值）如下：

| Family | Math（anchor 均值） | Science（anchor 均值） | Ratio |
|--------|--------------------:|-----------------------:|------:|
| trajectory | 98.4 | 5.1 | 19× |
| uncertainty | 18.3 | 5.1 | 3.6× |
| self_cert_logprob | 4.7 | 1.6 | 2.9× |
| confidence | 1.1 | 0.4 | 2.8× |
| availability_meta | 0.4 | 0.06 | — |

`math` 本质上是一个 **trajectory problem**；`science` 则更像 **mixed-signal problem**。对 `math` 而言，`trajectory` 一项就占据了总权重的大约 `85%`。对 `science` 而言，到 `100%` 时 `trajectory` 与 `uncertainty` 的贡献则接近并列。

---

## Q2：哪些具体特征在驱动决策？

### Math：主导模式是 Continuity vs Novelty 的对立

| Feature | @10% | @40% | @70% | @100% | Direction |
|--------|-----:|-----:|-----:|------:|-----------|
| `traj_continuity::raw` | +38.7 | +49.1 | +50.9 | +47.8 | ↑ 强正向 |
| `traj_novelty::raw` | -30.6 | -45.8 | -45.2 | -37.3 | ↓ 强负向 |
| `tok_gini_tail::raw` | -1.3 | -1.7 | -4.1 | -14.5 | ↓ 随 anchor 增强 |
| `tok_gini_slope::raw` | -3.7 | — | -6.2 | -13.8 | ↓ 随 anchor 增强 |

`math` 学到的是一条非常干净的规则：**好的 run 应该连续、顺滑，且不应过度新奇或乱跳**。到后期 anchor，基于 Gini 的 uncertainty 特征会进一步放大这条规则——在 `100%` 时，它们已经贡献了约 `28%` 的权重。

### Science：主导模式是 late-anchor 上的 Gini slope + Continuity

| Feature | @10% | @40% | @70% | @100% | Direction |
|--------|-----:|-----:|-----:|------:|-----------|
| `traj_continuity::raw` | +0.11 | +2.03 | +2.06 | +10.76 | ↑ 只在后期变重要 |
| `tok_gini_slope::raw` | +0.07 | +1.53 | +0.98 | +5.71 | ↑ 正向（与 math 相反） |
| `tok_gini_tail::raw` | — | — | — | +8.06 | ↑ 仅在 `100%` 明显出现 |
| `tok_logprob_prefix` | — | — | — | -2.02 | ↓ 惩罚过高 logprob |

`science` 模型在 `10%–40%` 基本接近“盲视”（权重量级约 `0.1`），然后在 `100%` 才突然形成明确判断。它偏好的是 **绝对 Gini / entropy 更高** 的 reasoning trace，这与 `math` 上 uncertainty 的负号方向正好相反。

---

## Q3：为什么 top1 会赢过 top2？

SVD component 直接告诉我们决策边界长什么样。`math` 的 component 几乎总是纯 trajectory（purity `0.56–0.76`），而 `science` 的 component 更常见 uncertainty + trajectory 的混合（purity `0.42–0.51`）。

### Math latent dimensions（每个 anchor 的 top 3）

- 所有 anchor 都存在一个 `trajectory(↑)` component，以 `traj_continuity::raw` 为主轴，代表“好推理”方向。
- 同时总会出现一个相反的 `trajectory(↓)` component，用来对比 `traj_continuity::raw` 高 vs `traj_novelty::raw` 高。
- 到 `70%–100%` 时，一个 `uncertainty(↑)` component 开始浮现，说明模型会把 token-level uncertainty 与 trajectory signal 结合起来看。

### Science latent dimensions（每个 anchor 的 top 3）

- `uncertainty(↑)` component（常出现在 `k = 12/13`）在 `40%–100%` 是 top 或 co-top，混合了 `tok_gini_slope::raw`、`traj_reflection_count::raw`、`tok_gini_tail::raw`。
- 这一个 component 就解释了为什么 `science` 中 `tok_gini_slope` 是正号：模型把 **高熵 + 反思更丰富** 的推理迹象视为更可能正确。
- 到 `100%` 时，最强 component 仍是一个 purity 只有 `0.46` 的 `trajectory(↓)`，说明 `science` 决策边界天生比 `math` 更模糊。

用更直白的话说：对于 `math`，`top1` 能赢 `top2`，是因为它有更强的连续推理信号、更低的新奇探索信号；对于 `science`，`top1` 能赢 `top2`，则是因为它在保持反思深度的同时，展现出更丰富、更多样的词汇分布（更高的 Gini）。

---

## Q4：常见失败模式

这一部分要加一个边界说明：**完整 wrong-top1 case 的统计需要 full-cache run 才能充分填满**；当前 checked-in smoke / compact artifact 中 wrong-top1 样本过少，所以这里更适合作为“风险指向”，而不是完整失败谱系。

推理时按 family 统计的平均绝对贡献如下：

| Domain | Family | @10% | @40% | @70% | @100% |
|--------|--------|-----:|-----:|-----:|------:|
| math | trajectory | 19.3 | 20.8 | 17.6 | 15.9 |
| math | self_cert_logprob | 8.5 | 12.2 | 36.4 | 63.8 |
| math | confidence | 7.3 | 5.0 | 10.5 | 17.8 |
| science | self_cert_logprob | 0.08 | 0.84 | 0.68 | 10.5 |
| science | confidence | 0.03 | 1.30 | 1.72 | 10.2 |

`math` 中 `self_cert/logprob` 的影响会爆炸式增长（`8 → 64`），到 `100%` 已经超过 `trajectory`。这意味着模型在推理后期越来越依赖“自证正确”信号；如果模型对错误答案表现出强烈自信，这就会成为典型失败模式。

---

## Q5：哪些解释稳定，哪些不稳定？

相邻 anchor 之间的符号一致率如下：

| Family | Math | Science |
|--------|-----:|--------:|
| confidence | 100% | 100% |
| trajectory | 95.6% | 97.8% |
| availability_meta | 94.4% | 100% |
| uncertainty | 91.1% | 96.7% |
| self_cert_logprob | 88.9% | 97.2% |

整体上，**95.8%** 的 feature weight 在跨 anchor 时保持符号不变。稳定的是 `raw` channel：绝对信号方向在四个 anchor 上大体保留。

真正不稳定的是 `::rank` channel：

- `math`：`tok_gini_slope::rank`（`2` 次翻转）、`tok_logprob_prefix::rank`（`2` 次翻转）
- `science`：`tok_neg_entropy_prefix::rank`（`2` 次翻转）、`tok_selfcert_prefix::rank`（`2` 次翻转）

解释上，这非常合理：`raw` 表示的是 feature 的绝对值，因此跨时间更稳定；`rank` 表示的是它在同题不同 run 里的相对位置，而随着 reasoning 继续推进，run 之间的排序本来就会重新洗牌。某条 run 在 `10%` 时 uncertainty 排名第一，到 `100%` 可能只排中游，因为其他 run 也逐步发展出类似的不确定性迹象。

---

## Q6：域间差异是一种结构性分歧

跨 domain 的符号一致率只有 **46.0%**。这不是噪声，而是 `uncertainty` family 在两个域上的语义发生了反转：

| Feature | Math weight | Science weight | Interpretation |
|--------|-------------|----------------|----------------|
| `tok_gini_slope::raw` | `-3.7` 到 `-13.8` | `+0.07` 到 `+5.7` | `math` 惩罚 token entropy；`science` 奖励它 |
| `tok_gini_tail::raw` | `-1.3` 到 `-14.5` | `+0.02` 到 `+8.1` | 同样反向 |
| `tok_neg_entropy_*::raw` | negative | positive | 同样反向 |
| `tok_conf_prefix::raw` | positive（confidence is good） | near-zero / ambiguous | 无统一结论 |
| `traj_continuity::raw` | always positive | always positive | ✓ 一致 |
| `traj_novelty::raw` | always negative | near-zero | 部分一致 |

`uncertainty` family 在两个域上的语义是反过来的。在 `math` 中，高 token Gini 往往意味着“下一步不确定”，因此是预警信号；而在 `science`（GPQA）中，高 Gini 反而可能表示模型在认真探索多个假设、真正处理难题，这与答对的概率正相关。

`trajectory` feature 尤其是 `traj_continuity`，是跨域唯一稳定而统一的信号：始终为正，在 `math` 上有 `95.6%` 稳定性，在 `science` 上有 `97.8%` 稳定性。这使它成为最接近“普适”解释特征的一类。

---

## 总结：三层解释

| Layer | Math | Science |
|------|------|---------|
| Feature level | 由 `traj_continuity (+)` 与 `traj_novelty (-)` 主导，late anchor 上 uncertainty 作为放大器 | `≤40%` 时几乎没有强信号，`100%` 主要由 `tok_gini (+)` + `traj_continuity (+)` 驱动 |
| Component level | `3` 个高纯度 trajectory component（`↑/↓` 成对）捕捉核心决策 | mixed-purity component（uncertainty + trajectory）更常见，top component 也没那么干净 |
| Family level | `trajectory > uncertainty > self_cert > confidence` | `uncertainty ≈ trajectory > self_cert`，且 uncertainty 符号与 `math` 相反 |

综合来看，模型实际上学到了两套不同的“好推理轨迹”理论：

- 在 `math` 中，**good = 连续 / 不新奇**
- 在 `science` 中，**good = 探索性更强 / 带不确定性但有反思**

而这两套理论又被打包在同一个 `ms_rr_r1` bundle 中（每个 anchor 共享 scaler / SVD 范式）。这说明该 bundle 编码的不是单一的 universal decision boundary，而是一个 **domain-conditional decision boundary**。

---

## 对应产物

本说明主要对应以下导出产物：

- `results/interpretability_smoke/es_svd_ms_rr_r1/family_summary.csv`
- `results/interpretability_smoke/es_svd_ms_rr_r1/effective_weights.csv`
- `results/interpretability_smoke/es_svd_ms_rr_r1/component_table.csv`
- `results/interpretability_smoke/es_svd_ms_rr_r1/stability_report.csv`
- `results/interpretability_smoke/es_svd_ms_rr_r1/failure_modes.csv`
- `results/interpretability/es_svd_ms_rr_r1/problem_top1_vs_top2.jsonl`

自动导出骨架来自 `scripts/export_svd_introspection.py`；本页是在其基础上补充的人工解读版。
