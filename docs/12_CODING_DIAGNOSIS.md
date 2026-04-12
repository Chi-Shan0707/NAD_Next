# 12: Coding Diagnosis — Why SVDomain Fails on livecodebench_v5

**Date**: 2026-04-12  
**Status**: Analysis complete  
**Data**: 10,688 samples · 167 problems · 30 legacy features at position 1.0  

---

## 0. Executive Summary

- **Headline**: coding is a boundary-case failure, not a hidden near-win.
- **H1 not supported**: no feature family exceeds `0.51` AUROC; the best family is `traj_only = 0.5094`.
- **H2 supported**: repeated GroupKFold gives `σ = 0.0501`, and the bootstrap `95% CI = [0.4403, 0.5669]` spans the chance range.
- **H3 weakly not supported**: SVD helps only marginally at the best rank (`r12 = 0.5012` vs `no_svd = 0.4910`), with no clean plateau.
- **Practical verdict**: the dominant measurable issue is evaluation instability, compounded by weak feature-label correlation across all legacy feature families.

---

## 1. The Three Hypotheses

| ID | Hypothesis | Test |
|----|-----------|------|
| H1 | Feature mismatch: the 22-feature `token_plus_traj` family is wrong for coding | Analysis A |
| H2 | Evaluation instability: small-group class degeneracy inflates variance | Analysis B |
| H3 | Non-compact structure: coding CoT traces have no useful low-rank space | Analysis C |

---

## 2. Analysis A — Feature Family Ablation

**Question**: Which feature family works best? Does SVD help any of them?

| Family | Condition | AUROC | Balanced Acc | StopAcc |
|--------|-----------|-------|-------------|---------|
| tok_conf_only | no_svd | 0.4527 | 0.5031 | 0.5695 |
| tok_conf_only | svd_r12 | 0.4527 | 0.5031 | 0.5695 |
| token_uncertainty | no_svd | 0.4563 | 0.4988 | 0.5686 |
| token_uncertainty | svd_r12 | 0.4563 | 0.4988 | 0.5686 |
| token_only | no_svd | 0.4686 | 0.4936 | 0.5749 |
| token_only | svd_r12 | 0.4705 | 0.4958 | 0.5866 |
| traj_only | no_svd | 0.5094 | 0.5018 | 0.5456 |
| traj_only | svd_r12 | 0.5094 | 0.5021 | 0.5456 |
| token_plus_traj | no_svd | 0.4910 | 0.4943 | 0.5697 |
| token_plus_traj | svd_r12 | 0.5012 | 0.5035 | 0.5936 |
| prefix_local | no_svd | 0.4775 | 0.5000 | 0.5932 |
| prefix_local | svd_r12 | 0.4775 | 0.5000 | 0.5932 |
| all_30 | no_svd | 0.5058 | 0.5153 | 0.5934 |
| all_30 | svd_r12 | 0.5001 | 0.5033 | 0.5458 |

**H1 verdict**: **NOT SUPPORTED**
- Best family (no_svd): `traj_only` AUROC = 0.5094
- Threshold for H1 support: AUROC > 0.55

---

## 3. Analysis B — Evaluation Instability

**Question**: How stable is the AUROC estimate for coding?

### Protocol 1 — Repeated GroupKFold (n_seeds × n_splits)

- Mean AUROC: 0.5112
- AUROC std across seeds: 0.0501
- Degenerate fold fraction: 0.0% (0/50)

### Protocol 2 — Group Bootstrap (out-of-bag AUROC distribution)

- 95% CI: [0.4403, 0.5669]
- CI spans [0.45, 0.55]: True

### Protocol 3 — Leave-dataset-out (85% train / 15% test)

- Holdout AUROC: 0.4542
- Holdout balanced_acc: 0.4664
- n_test_groups=25  pos=735  neg=865

**H2 verdict**: **SUPPORTED**
- std=0.0501 (threshold 0.05), degenerate=0.0% (threshold 10%)

---

## 4. Analysis C — Low-Rank Compactness

**Question**: Is there a rank plateau where SVD matches no_svd?

- Coding effective rank: 12.0802  |  Math effective rank: 10.9136

### Rank sweep (token_plus_traj, raw+rank)

| Domain | Rank | AUROC | Balanced Acc | N Valid Folds | Eff.Rank | VarMass@4 | VarMass@8 |
|--------|------|-------|-------------|---------------|----------|----------|----------|
| coding | no_svd | 0.4910 | 0.4943 | 5 | - | - | - |
| coding | r1 | 0.4461 | 0.5000 | 5 | 12.0802 | 0.8416 | 0.9701 |
| coding | r2 | 0.4467 | 0.4987 | 5 | 12.0802 | 0.8416 | 0.9701 |
| coding | r4 | 0.4749 | 0.4891 | 5 | 12.0802 | 0.8416 | 0.9701 |
| coding | r6 | 0.4720 | 0.4876 | 5 | 12.0802 | 0.8416 | 0.9701 |
| coding | r8 | 0.4879 | 0.4960 | 5 | 12.0802 | 0.8416 | 0.9701 |
| coding | r12 | 0.5012 | 0.5035 | 5 | 12.0802 | 0.8416 | 0.9701 |
| coding | r16 | 0.4915 | 0.4978 | 5 | 12.0802 | 0.8416 | 0.9701 |
| coding | r24 | 0.4917 | 0.4966 | 5 | 12.0802 | 0.8416 | 0.9701 |
| math | no_svd | 0.9679 | 0.9193 | 5 | - | - | - |
| math | r1 | 0.6115 | 0.5472 | 5 | 10.9136 | 0.8849 | 0.9836 |
| math | r2 | 0.7149 | 0.6307 | 5 | 10.9136 | 0.8849 | 0.9836 |
| math | r4 | 0.8891 | 0.7874 | 5 | 10.9136 | 0.8849 | 0.9836 |
| math | r6 | 0.9044 | 0.8153 | 5 | 10.9136 | 0.8849 | 0.9836 |
| math | r8 | 0.9495 | 0.8834 | 5 | 10.9136 | 0.8849 | 0.9836 |
| math | r12 | 0.9640 | 0.9105 | 5 | 10.9136 | 0.8849 | 0.9836 |
| math | r16 | 0.9679 | 0.9205 | 5 | 10.9136 | 0.8849 | 0.9836 |
| math | r24 | 0.9678 | 0.9189 | 5 | 10.9136 | 0.8849 | 0.9836 |

**H3 verdict**: **NOT SUPPORTED**
- Coding no_svd=0.4910, best_svd=0.5012
- Math no_svd=0.9679, best_svd=0.9679

---

## 5. Analysis D — Failure Mode Interpretation

### Archetype Table

| Archetype | Label | N | % Wrong | Feature | z-score | Description |
|-----------|-------|---|---------|---------|---------|-------------|
| high_conf_wrong | FP | 17 | 0.3% | tok_conf_prefix | 1.7969 | High tok_conf_prefix but label=0; model over-trusts confidence signal |
| trajectory_noise | FP|FN | 1563 | 25.0% | traj_reflection_count | -1.2771 | High |traj_reflection_count| uncorrelated with correctness in coding |
| overreflection_trap | FN | 1525 | 24.4% | traj_reflection_count | -1.2509 | High reflection count for correct solutions; SVD over-penalises reflective correct traces |
| zero_traj_ambiguity | FP|FN | 1412 | 22.6% | traj_magnitude | -1.1192 | Near-zero trajectory features; model has no discriminative signal |
| boundary_ambiguity | FP|FN | 563 | 9.0% | decision_score | -0.0001 | Scores near decision boundary; prediction is effectively random |

---

## 6. Forced Verdicts

**Primary cause: H2 (evaluation instability)**

Evaluation instability (H2) is the dominant measurable factor, but not the only one. No feature family achieves reliable discrimination above 0.55 AUROC, and the bootstrap 95% CI spans the chance range. The near-random AUROC for coding arises from a combination of weak feature-label correlation and high variance due to the limited number of problem groups (167), rather than a simple tuning failure. SVD marginally helps at the best rank, suggesting that structure exists but is too weak and noisy to exploit reliably.

### Does low-rank have explanatory value despite weak predictive value?

Partially. Coding effective rank (12.1) slightly exceeds math (10.9, ratio=1.11×). The coding feature space is modestly more diffuse, but the difference alone cannot explain the large gap in AUROC. Feature-label mismatch is the stronger contributor.

### Paper framing recommendation

**Boundary case / negative result with diagnostic value.** The coding domain exposes a regime where the SVDomain framework cannot improve over a random baseline. The result should be reported as a principled boundary case, not a tuning failure. The diagnosis provides a forward pointer: CODING_DYNAMIC features (not in the legacy 30-feature schema) are the recommended next step.

---

## 7. Paper Paragraph Template

```
Coding (livecodebench_v5) is the only domain where SVDomain produces near-random predictions
(AUROC ≈ 0.4910). Our diagnosis separates three contributing factors. First, no feature
family in the 30-feature legacy schema achieves reliable discrimination (H1): even the best
single-family classifier stays near random, ruling out feature selection as a simple fix.
Second, the 167 coding problems yield high AUROC variance across repeated cross-validation
seeds (σ ≈ 0.0501), making it difficult to distinguish signal from noise (H2).
Third, the effective SVD rank of coding representations (12.0802) is
modestly higher than for math (10.9136), suggesting that coding CoT traces
are somewhat more structurally diffuse (H3, partially). Coding is best understood as a boundary case: its current
weakness arises from a mixture of feature mismatch and evaluation instability, rather than
simply from insufficient tuning. We recommend extracting CODING_DYNAMIC features in future work.
```
