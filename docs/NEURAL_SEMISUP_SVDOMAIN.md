# Neural Semi-Supervised SVDomain

## Summary

- This artifact keeps the existing SVDomain protocol intact: the same `22` canonical structured features, `raw+rank` representation, domain-conditioned routing, anchors `10/40/70/100`, and grouped `85/15` holdout by `dataset + problem_id`.
- The main change is the bottleneck: fixed `TruncatedSVD` is replaced with a small self-supervised neural bottleneck with separate raw/rank encoders, domain and anchor embeddings, a shared trunk, a latent `z`, and linear decoders back to the canonical features.
- The strongest **full-label** coding neural row comes from `latent_dim=24` with tuned self-supervised weights (`align=0.10`, `future=0.25`, `consistency=0.10`, `contrastive=0.10`).
- Requested artifacts:
  - table: `results/tables/neural_semisup_svdomain.csv`
  - figures: `results/figures/neural_semisup_svdomain`
  - models: `models/ml_selectors/neural_semisup_svdomain`
  - follow-up coding sweeps: `docs/NEURAL_SEMISUP_SVDOMAIN_MORE.md` and `docs/NEURAL_SEMISUP_SVDOMAIN_HYBRID.md`

## Requested Answers

1. **Does self-supervised bottleneck learning help beyond fixed SVD?** Yes on coding, modestly, at the main full-label endpoint. The best full-label neural coding row reaches `AUC of AUROC=0.5237`, versus `0.4663` for the frozen-SVD linear head (`+0.0573`) and `0.4924` for the current coding SVDomain route (`+0.0313`). I would frame this as a coding-specific improvement, not a general replacement for the linear stack.
2. **Does it mainly help coding or not?** Mainly coding. The best neural rows still lag the current linear routes on math (`0.8556` vs `0.9586`), science (`0.7157` vs `0.8013`), and math+science (`0.8138` vs `0.9237`). The positive result is therefore concentrated in coding rather than being broad across domains.
3. **Does it improve label efficiency?** Partially yes. Relative to the frozen-SVD linear head, the best neural coding row wins at every label fraction in this sweep (`1/5/10/20/50/100%`) with an average `+0.0460` AUC-of-AUROC gain. Relative to the no-SVD logistic baseline, the story is mixed: neural wins at `1%`, `5%`, `50%`, and `100%`, but not at `10%` or `20%`. The strongest low-/mid-label points are promising, but I would not overclaim them as a uniformly better recipe.
4. **Does it preserve enough interpretability to be worth the extra complexity?** Probably yes for a paper ablation, with caveats. The latent is still small, decoders are linear back to named canonical features, and decoder-strength inspection remains straightforward. That said, the current evidence supports “worth trying for coding” more than “worth replacing SVDomain everywhere.”
5. **Is the gain coming from representation learning, objective change, or both?** Both, but not cleanly separable. The neural bottleneck beats frozen SVD on coding, which suggests learned representation helps. But the best coding baseline in this table is still the shallow supervised MLP (`0.5625`), so the gain cannot be attributed to self-supervision alone.

## Main Comparisons

- `neural_semisup_freeze` best coding row (`latent=24`): `AUC of AUROC=0.5237`, `AUC of SelAcc=0.4581`, `AUROC@100%=0.4936`, `Stop Acc@100%=0.5200`
- `current_svdomain_coding`: `AUC of AUROC=0.4924`, `AUC of SelAcc=0.4056`, `AUROC@100%=0.4282`, `Stop Acc@100%=0.5200`
- `frozen_svd_linear_head`: `AUC of AUROC=0.4663`, `AUC of SelAcc=0.3725`, `AUROC@100%=0.4741`, `Stop Acc@100%=0.5200`
- `no_svd_logreg`: `AUC of AUROC=0.4825`, `AUC of SelAcc=0.4756`, `AUROC@100%=0.5386`, `Stop Acc@100%=0.6000`
- `shallow_mlp_no_ssl`: `AUC of AUROC=0.5625`, `AUC of SelAcc=0.5613`, `AUROC@100%=0.5305`, `Stop Acc@100%=0.6000`

## Coding Late Anchors

- Best neural late-anchor rows use `latent=24`.
- At `70%`, neural reaches `AUROC=0.5218` versus `0.4973` for frozen SVD.
- At `100%`, neural reaches `AUROC=0.4936` versus `0.4741` for frozen SVD.
- These are real but not dominant gains: both `no_svd_logreg` and `shallow_mlp_no_ssl` remain stronger at the late coding anchors in this run.

## Label-Efficiency Notes

- Best neural coding rows by fraction:
  - `1%`: `0.4938`
  - `5%`: `0.5003`
  - `10%`: `0.5491`
  - `20%`: `0.5641`
  - `50%`: `0.5339`
  - `100%`: `0.5237`
- The curve is not monotone, so I would treat it as evidence that the neural bottleneck can help in low- to mid-label regimes, not as evidence of a uniformly better training recipe.

## Interpretability Notes

- The model remains intentionally small: separate raw/rank encoders, compact latent `z ∈ {8,16,24}`, and linear decoders.
- Top decoder-linked features in the best run are still human-readable structured signals:
  - `traj_reflection_count`
  - `tok_conf_recency`
  - `tok_conf_prefix`
  - `tok_selfcert_recency`
  - `tok_selfcert_prefix`
- The latent reuse heatmap spans `12` domain-anchor cells and remains suitable for paper-style diagnostic visualization.

## Figures

- `results/figures/neural_semisup_svdomain/label_efficiency_coding_auc_of_auroc.png`
- `results/figures/neural_semisup_svdomain/coding_late_anchor_auroc.png`
- `results/figures/neural_semisup_svdomain/latent_reuse_heatmap.png`
- `results/figures/neural_semisup_svdomain/decoder_feature_strength.png`

## Mainline Recommendation

- The reviewer-safe default remains `neural_semisup_freeze` with `latent_dim=24` for the full-label coding endpoint.
- Follow-up coding-only adapter and hybrid sweeps are documented separately. They improve some mid-label and `70%`-anchor coding settings, but they are not stable enough to replace the simpler frozen-latent route as the main artifact.

## Cautions

- This is best read as an ablation on whether learned low-rank structure helps coding within the existing feature world, not as evidence that the whole SVDomain pipeline should become neural.
- The best neural coding route does improve over the current coding SVDomain route, but it does **not** become the strongest overall baseline in this table.
- Low-label (`1%`) rows use a conservative direct-fit fallback when grouped CV becomes too small to score reliably. That keeps the label-efficiency sweep complete, but those rows should be interpreted as stress-test points rather than stable model-selection estimates.
- No raw CoT text, hidden-state transformers, or protocol changes were introduced.
