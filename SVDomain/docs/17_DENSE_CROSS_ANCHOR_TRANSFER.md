# Dense Cross-Anchor Transfer

This note compresses the dense all-to-all cross-anchor transfer results for `es_svd_ms_rr_r2` into a paper-ready summary that can be cited in the main text or appendix.

## 1. Main paper assets

- Main note: `docs/17_DENSE_CROSS_ANCHOR_TRANSFER.md`
- Paper-package tables:
  - `SVDomain/results/tables/dense_cross_anchor_transfer_matrix.csv`
  - `SVDomain/results/tables/dense_cross_anchor_transfer_deltas.csv`
  - `SVDomain/results/tables/dense_cross_anchor_transfer_summary.csv`
- Source-of-truth tables:
  - `results/tables/dense_cross_anchor_transfer_matrix.csv`
  - `results/tables/dense_cross_anchor_transfer_deltas.csv`
  - `results/tables/dense_cross_anchor_transfer_summary.csv`
- Experiment script: `SVDomain/experiments/run_dense_cross_anchor_transfer.py`

## 2. Dense headline summary

| Domain | Diagonal Δ(Frozen−Task) | Offdiag-all Δ | Near-gap Δ (`10/20`) | Far-gap Δ (`50–90`) | Best source anchor |
|---|---:|---:|---:|---:|---:|
| `math` | -0.13 pts | -0.34 pts | -0.16 pts | -0.62 pts | `30%` |
| `science` | -0.09 pts | -0.54 pts | -0.23 pts | -0.97 pts | `50%` |

Main takeaways:

- In `math`, the shared basis remains close to the task-specific one even under dense all-to-all transfer, and degradation grows only gradually with anchor distance.
- In `science`, transfer is still real and not confined to `100%`, but distance-decay is stronger, especially for early-to-late forward reuse.
- The most stable narrative is therefore not “transfer only at slot-100”, but rather **math = broadly reusable across trajectory; science = reusable but direction- and distance-sensitive**.

## 3. Distance-decay readout

### Math

- forward all mean gap：`-0.11 pts`
- backward all mean gap：`-0.56 pts`
- worst pair：`100→10`，`-2.51 pts`
- best off-diagonal pair：`60→100`，`+0.01 pts`

Interpretation:

- The frozen basis in `math` remains very stable when transferred to later targets; most of the degradation appears in **late-to-early backward** transfer.
- This matches the dense-anchor timing result: `math` already reaches 95%-of-final AUROC at `10%`, so later anchors mostly refine the boundary rather than replacing it.

### Science

- forward all mean gap：`-0.98 pts`
- backward all mean gap：`-0.10 pts`
- worst pair：`10→50`，`-4.17 pts`
- best off-diagonal pair：`100→90`，`+0.26 pts`

Interpretation:

- Dense transfer in `science` is clearly asymmetric: **late-to-earlier** reuse remains fairly stable, while **early-to-later** reuse degrades rapidly with distance.
- This suggests that early science anchors already contain useful signal, but their basis has not yet matured into a late-stage representation that can be reused without loss.

## 4. Source-anchor ranking

### Math source-anchor ranking by mean off-diagonal Δ(Frozen−Task)

`30% (-0.14)` > `40% (-0.17)` > `20% (-0.18)` > `50% (-0.22)` > `60% (-0.26)` > `10% (-0.27)` > `70% (-0.34)` > `80% (-0.41)` > `90% (-0.52)` > `100% (-0.87)`

### Science source-anchor ranking by mean off-diagonal Δ(Frozen−Task)

`50% (-0.02)` > `60% (-0.02)` > `70% (-0.03)` > `40% (-0.03)` > `80% (-0.05)` > `100% (-0.09)` > `90% (-0.10)` > `30% (-0.10)` > `20% (-1.37)` > `10% (-3.60)`

Interpretation:

- In `math`, the best source anchor sits in the early-middle regime (`30–40%`), not at the latest `100%` anchor.
- In `science`, the best source anchor is no longer the isolated `100%` anchor seen in the sparse 4-anchor study, but a cluster of **mid/late-middle anchors (`50–80%`)**.
- The dense grid therefore sharpens the sparse `10/40/70/100` story: **the most transferable basis is often not the latest anchor, but a mature mid/late anchor that has not yet been overly shaped by completion-specific information**.

## 5. Recommended paper sentence

> Dense cross-anchor transfer shows that reuse is not a slot-100-only effect. In math, the frozen basis remains close to the task-specific one across the full `10/20/.../100` trajectory, with a diagonal mean gap of only `-0.13` AUC points and an all off-diagonal mean gap of `-0.34`; the best reusable source anchor is `30%`. In science, transfer remains real but more direction-sensitive: the diagonal mean gap is `-0.09`, the all off-diagonal mean gap is `-0.54`, and forward early-to-late transfer degrades much more than backward late-to-early transfer, with the best source anchors concentrated around `50–70%`.

## 6. Link to dense timing

This result should be cited together with `docs/16_DENSE_ANCHOR_EARLYSTOP.md`:

- `math`: reaches 95%-of-final at `10%` and plateaus around `50%`, consistent with weak distance-decay under dense transfer.
- `science`: reaches 95%-of-final at `20%` and plateaus around `40%`, consistent with an “early coarse signal + late refinement” transfer pattern.

The two results can be combined into one sentence:

> Dense anchors show that math saturates early and remains cross-anchor reusable throughout the trajectory, whereas science exhibits early usable signal but a more maturity-dependent reusable basis.
