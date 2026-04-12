# 16. Dense-Anchor EarlyStop

这份 note 只回答一个问题：把 EarlyStop 从粗粒度 `10 / 40 / 70 / 100` 扩展到完整 `10 / 20 / ... / 100` 之后，主论文关于“预测信号何时出现”的结论会不会改变。

## 1. Setup

- 任务范围：只做 `EarlyStop`，不扩展到 `Best-of-N` 或 `RL ranking`。
- Anchor grid：`10 / 20 / 30 / 40 / 50 / 60 / 70 / 80 / 90 / 100`。
- 域：`math / science / coding`。
- B0：`legacy canonical`（22 特征 = token + trajectory + availability）。
- B1：`canonical + neuron_meta_min`（B0 + `nc_mean` / `nc_slope`）。
- 实现：复用现有 prefix-safe feature store 与 offline holdout split，只把评估网格从四锚点细化到十锚点。
- 协议：沿用当前 offline holdout + grouped CV；不改 split / grouping / metrics。

## 2. Headline answers

- **当前 4 anchors 是否足够支撑主论文？** 是。Dense-anchor 结果没有推翻粗锚点结论：`math` 仍然最早、最强，`science` 早期已有可用信号但中后段继续抬升，`coding` 仍然最弱。
- **Dense anchors 是否改变主结论？** 主要没有改结论，而是在 timing 上给出更细定位。B0 的 AUC-of-AUROC 分别为 `math=0.968`、`science=0.827`、`coding=0.432`。
- **Dense anchors 是否应进入正文？** 如果正文要和 neuron-agreement / early-correctness 文献对话，建议至少放一张 dense-anchor timing 图；否则四锚点主表仍然够用，dense 结果适合作为 timing-focused 主文图或 appendix 主证据。

## 3. Domain readout

- `math` 基本属于早期强信号：在 `10%` 就达到 final-AUROC 的 95%，并在 `50%` 左右进入平台；dense anchors 更像是在细化“多早就够了”，而不是改写结论。
- `science` 不是纯 late-onset：`10%` 已越过固定阈值，`20%` 已达到 final-AUROC 的 95%，但 AUROC 仍从 `10%` 的 `0.767` 继续抬升到 `100%` 的 `0.841`，并在 `40%` 左右进入 `±0.01` 平台；dense anchors 更支持“早期已有 coarse signal、后期继续抬升”而不是“只在 completion 才首次出现信号”。
- `coding` 更接近噪声主导：直到 `100%` 也没有稳定越过固定阈值，final-anchor AUROC 只有 `0.407`。

## 4. Neuron-meta-min comparison

- 当前仓库里没有已经固化的更强 neuron bundle，因此这里按约定使用 `canonical + neuron_meta_min` fallback。
- `math`: B0 AUC-of-AUROC=`0.968` → B1=`0.966`。 `math` 上，B1−B0 的 mean ΔAUROC 在 early anchors 为 `-0.000`，在 late anchors 为 `-0.004`。
- `science`: B0 AUC-of-AUROC=`0.827` → B1=`0.829`。 `science` 上，B1−B0 的 mean ΔAUROC 在 early anchors 为 `+0.000`，在 late anchors 为 `+0.006`。
- `coding`: B0 AUC-of-AUROC=`0.432` → B1=`0.442`。 `coding` 上，B1−B0 的 mean ΔAUROC 在 early anchors 为 `-0.007`，在 late anchors 为 `+0.023`。

## 5. Paper-facing interpretation

- `math`: dense anchors show that the predictive signal is already strong by `10%` and largely stabilizes by the mid trajectory.
- `science`: dense anchors argue for early coarse signal plus late refinement, not a purely completion-only onset story.
- `coding`: dense anchors help distinguish a weak noisy curve from a true onset event; in the current family it never clears the fixed threshold and remains the weakest slice.
- 对弱域来说，`95% of final AUROC` 不是最稳健的 onset 指标，因为 final AUROC 本身可能偏低；fixed-threshold onset 与 plateau 更值得引用。
- 对论文正文最稳的写法是：dense anchors refine **when** predictability appears and plateaus; they do not overturn the coarse-anchor story.

## 6. Recommended sentence

> The coarse 10/40/70/100 anchors are directionally correct, but denser anchors reveal where each domain’s predictive signal actually emerges and stabilizes.

## 7. Artifacts

- `results/tables/dense_anchor_earlystop.csv`
- `results/tables/onset_of_signal.csv`
- `results/tables/plateau_of_signal.csv`
- `results/tables/dense_anchor_neuron_vs_legacy.csv`
