# Neural Semi-Supervised SVDomain: Extra Coding Sweeps

## Goal

- Push the semi-supervised route harder on coding without changing the paper-facing protocol.
- Keep the same structured feature world: `22` canonical features, `raw+rank`, grouped `85/15` holdout, anchor routing, no raw text, no hidden-state models.
- Test whether small supervised adaptation on top of the self-supervised latent can improve coding more than the plain frozen-latent head.

## Extra Variants Tried

- `neural_semisup_coding_adapter`
  - Same pretrained neural bottleneck as before.
  - Same frozen neural head for math/science.
  - Coding-only supervised low-rank latent adapter on top of `z`, trained with pointwise loss plus optional pairwise loss on late anchors.
- `neural_semisup_coding_hybrid`
  - Same coding adapter candidate as above.
  - Per coding anchor, choose between the plain frozen neural head and the coding adapter using grouped CV on the training split.

## Main Result

- The original frozen neural route remains the strongest neural choice at full-label coding:
  - `neural_semisup_freeze`: `AUC of AUROC = 0.5237`
  - `neural_semisup_coding_adapter`: `0.4742`
  - `neural_semisup_coding_hybrid`: `0.4742`
- So the additional supervised adaptation did **not** improve the full-label coding endpoint that matters most for the current artifact.

## Where The Extra Variants Helped

- The coding adapter **did** help in some intermediate-label regimes:
  - `20%` labels: adapter `0.5931` vs frozen neural `0.5560`
  - `50%` labels: adapter `0.5367` vs frozen neural `0.5339`
- The adapter also improved the late `70%` coding anchor at full labels:
  - frozen neural `AUROC = 0.5218`
  - coding adapter `AUROC = 0.5284`
- But that came with a worse `100%` anchor:
  - frozen neural `AUROC = 0.4936`
  - coding adapter `AUROC = 0.4780`

## Interpretation

- The extra supervised flexibility is not useless; it seems able to sharpen the representation for some coding subproblems, especially around the `70%` anchor and some mid-label regimes.
- But the current adapter objective is not stable enough across anchors. In practice it trades away too much of the `100%` anchor to be a better default than the simpler frozen neural head.
- The hybrid selector did not rescue this because its grouped-CV choice still leaned toward the worse late-anchor configuration at full labels.

## Practical Takeaway

- **Keep** the core frozen neural bottleneck path as the main semi-supervised result.
- **Do not promote** the coding adapter or coding hybrid as the new mainline yet.
- The strongest reviewer-safe claim after these extra sweeps is:
  - self-supervised bottleneck learning can improve coding relative to fixed SVD,
  - but the best evidence still comes from the simpler frozen-latent neural head,
  - and extra supervised adaptation on top of that latent is currently mixed rather than clearly beneficial.

## Files

- Main frozen-neural sweep: `results/tables/neural_semisup_svdomain.csv`
- Extra adapter sweep: `results/tables/neural_semisup_svdomain_more.csv`
- Extra hybrid sweep: `results/tables/neural_semisup_svdomain_hybrid.csv`
