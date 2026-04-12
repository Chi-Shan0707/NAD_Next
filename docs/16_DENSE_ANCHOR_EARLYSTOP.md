# 16. Dense-Anchor EarlyStop

这份 note 的目标，是把 dense-anchor EarlyStop 结果压缩成一页内可直接转写到论文正文的主结论、主表与主图口径。

## 1. Main paper assets

- 主表：`results/tables/dense_anchor_main_table.csv`
- 主图：`results/figures/dense_anchor_earlystop_main.png`
- 完整曲线表：`results/tables/dense_anchor_earlystop.csv`
- onset / plateau 表：`results/tables/onset_of_signal.csv`、`results/tables/plateau_of_signal.csv`
- neuron-vs-legacy 对照：`results/tables/dense_anchor_neuron_vs_legacy.csv`

## 2. Main table

| Domain | B0 AUC-AUROC | B1-B0 ΔAUC | B0 onset95 | B0 fixed onset | B0 plateau | Reading |
|---|---:|---:|---:|---:|---:|---|
| `math` | 0.968 | -0.002 | 10% | 10% | 50% | very early strong signal; B0 reaches 95%-of-final at 10% and plateaus by 50%; neuron add-on is not helpful overall (-0.002 AUC) |
| `science` | 0.827 | +0.002 | 20% | 10% | 40% | usable signal already appears by 10%, but the curve keeps improving into later anchors; neuron add-on helps modestly overall (+0.002 AUC) |
| `coding` | 0.432 | +0.010 | 10%* | — | 100% | weak curve with no fixed-threshold onset; neuron add-on helps mainly late anchors (+0.010 AUC overall) |

* `coding` 这类弱域上，`95% of final AUROC` 不是最稳健的 onset 指标；fixed-threshold onset 与 plateau 更值得优先引用。

## 3. Main claims

- `math` 基本属于早期强信号：在 `10%` 就达到 final-AUROC 的 95%，并在 `50%` 左右进入平台；dense anchors 更像是在细化“多早就够了”，而不是改写结论。
- `science` 不是纯 late-onset：`10%` 已越过固定阈值，`20%` 已达到 final-AUROC 的 95%，但 AUROC 仍从 `10%` 的 `0.767` 继续抬升到 `100%` 的 `0.841`，并在 `40%` 左右进入 `±0.01` 平台；dense anchors 更支持“早期已有 coarse signal、后期继续抬升”而不是“只在 completion 才首次出现信号”。
- `coding` 更接近噪声主导：直到 `100%` 也没有稳定越过固定阈值，final-anchor AUROC 只有 `0.407`。

## 4. Neuron add-on readout

- `math`: B1−B0 的 mean ΔAUROC 为 early `-0.000` / late `-0.004` / overall `-0.002`。
- `science`: B1−B0 的 mean ΔAUROC 为 early `+0.000` / late `+0.006` / overall `+0.002`。
- `coding`: B1−B0 的 mean ΔAUROC 为 early `-0.007` / late `+0.023` / overall `+0.010`。

## 5. Figure caption candidate

> Dense-anchor EarlyStop curves show that math is already strongly predictable at very early prefixes, science has usable early signal but continues to improve through later anchors, and coding remains weak under the current feature family; neuron features help mainly on science/coding late anchors rather than shifting the earliest onset.

## 6. Table sentence candidate

> The coarse 10/40/70/100 anchors are directionally correct, but denser anchors reveal that math saturates very early, science exhibits early coarse signal followed by late refinement, and coding never develops a comparably stable early-stop signal.
