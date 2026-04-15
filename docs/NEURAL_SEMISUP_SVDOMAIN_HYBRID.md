# Neural Semi-Supervised SVDomain: Coding Hybrid Sweep

## Goal

- Test a more aggressive coding-only follow-up without changing the paper-facing protocol.
- Keep the same structured input space: `22` canonical features, `raw+rank`, domain conditioning, anchors `10/40/70/100`, grouped `85/15` holdout, no raw CoT, no hidden states.
- Let coding choose, per anchor, between the frozen neural head and a coding-specific low-rank latent adapter using grouped CV on the training split.

## Main Result

- The hybrid route is **not** a better default than the simpler frozen neural route.
- It helps some low-/mid-label coding settings, but it does not improve the full-label endpoint that matters most for a reviewer-safe paper artifact:
  - `neural_semisup_coding_hybrid`: coding `AUC of AUROC = 0.4742` at `100%` labels
  - `neural_semisup_freeze`: coding `AUC of AUROC = 0.5237` at `100%` labels
  - `current_svdomain_coding`: coding `AUC of AUROC = 0.4924`

## Requested Answers

1. **Does self-supervised bottleneck learning help beyond fixed SVD?** Not in this hybrid variant at the full-label endpoint. The hybrid route underperforms the simpler frozen neural head and also underperforms the current coding SVDomain route at `100%` labels.
2. **Does it mainly help coding or not?** Only coding was changed here, and the effect is mixed even within coding. The hybrid route helps around some mid-label regimes, but not enough to justify replacing the mainline.
3. **Does it improve label efficiency?** Sometimes. The hybrid coding curve reaches `0.4938 / 0.4719 / 0.5266 / 0.5931 / 0.5412 / 0.4742` at `1/5/10/20/50/100%` labels. The `20%` and `50%` points are competitive, but the `100%` point regresses too much.
4. **Does it preserve enough interpretability to be worth the extra complexity?** Only partially. It still keeps the same compact latent and linear decoder path, but the extra coding adapter and per-anchor selection logic make the method harder to explain than the plain frozen-latent route.
5. **Is the gain coming from representation learning, objective change, or both?** This sweep mainly tests objective and adaptation changes on top of the learned latent. The results suggest that extra supervised adaptation can help selected coding anchors, but the benefit is not stable enough to treat as a stronger overall representation.

## Where It Helped

- Mid-label coding:
  - `20%`: `0.5931`
  - `50%`: `0.5412`
- Late anchor `70%` at full labels:
  - hybrid `AUROC = 0.5284`
  - frozen neural `AUROC = 0.5218`

## Where It Hurt

- Full-label aggregate coding:
  - hybrid `0.4742`
  - frozen neural `0.5237`
- Late anchor `100%` at full labels:
  - hybrid `AUROC = 0.4780`
  - frozen neural `AUROC = 0.4936`

## Recommendation

- Keep `neural_semisup_freeze` as the main semi-supervised result.
- Treat the hybrid route as exploratory evidence that coding-specific adaptation may help some regimes, especially around the `70%` anchor, but is not yet stable enough for the main claim.

## Files

- Hybrid table: `results/tables/neural_semisup_svdomain_hybrid.csv`
- Hybrid figures: `results/figures/neural_semisup_svdomain_hybrid`
