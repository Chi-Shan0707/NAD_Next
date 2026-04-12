# 17: NAD-Lite Neuron Agreement Features

**Date**: 2026-04-12  
**Status**: Analysis complete  
**Data**: cache_all_547b9060debe139e.pkl · 30 legacy features · position 1.0  
**ViewSpec**: `agg=MAX, cut=MASS(0.95), order=BY_KEY` — full-sequence neuron sets from base CSR  
**Note**: Prefix-level agreement features infeasible (rows/bank unavailable). All agreement operates at full-sequence position only.

---

## 1. Conditions

| ID | Name | Features | N feat |
|----|------|----------|--------|
| C0 | `canonical` | TOKEN[0-10] + TRAJ[11-15] + AVAIL[19-24] | 22 |
| C1 | `canonical_plus_nc` | C0 + nc_mean[16] + nc_slope[17] | 24 |
| C2 | `canonical_plus_nad_scalar` | C0 + 4 NAD scalars [30-33] | 26 |
| C3 | `canonical_plus_nad_agree` | C0 + 5 agreement features [34-38] | 27 |
| C4 | `canonical_plus_all_nad` | C0 + 4 scalar + 5 agreement | 31 |

**NAD Scalar features** (indices 30-33):
- `activated_neuron_count`: unique neurons in full run (base CSR)
- `neuron_density`: activated_neuron_count / max within problem group
- `topk10_neuron_count`: neurons with weight ≥ 90th percentile of this run
- `neuron_weight_entropy`: Shannon entropy of normalized weight distribution

**NAD Agreement features** (indices 34-38, full-sequence Jaccard, n_peers ≥ 2):
- `mean_jaccard_sim`: mean(1 - D[i,j≠i]) — mean similarity to all peers
- `max_jaccard_sim`: max(1 - D[i,j≠i]) — most similar peer
- `knn_agree_score`: mean of top-k similarities (k=5)
- `medoid_distance`: mean(D[i,j≠i]) — lower = more central
- `minact_rank`: rank of activated_neuron_count within group [0=smallest]

Degenerate groups (n_runs < 2): agreement features = NaN → imputed with column mean.

Modeling: `no_svd` (StandardScaler+LR) and `svd_r12` (Scaler+TruncatedSVD+LR).  
Representation: always `raw+rank`. Evaluation: 5-fold GroupKFold.

---

## 2. Main Ablation Results

### AUROC by Domain and Condition (no_svd)

| Domain | C0 | C1 | C2 | C3 | C4 |
|--------|----|----|----|----|----|
| math | 0.9679 | 0.9678 | 0.9679 | 0.9688 | 0.9711 |
| science | 0.7798 | 0.7757 | 0.7824 | 0.7803 | 0.7862 |
| coding | 0.4910 | 0.5062 | 0.5051 | 0.4876 | 0.5000 |

### AUROC by Domain and Condition (svd_r12)

| Domain | C0 | C1 | C2 | C3 | C4 |
|--------|----|----|----|----|----|
| math | 0.9640 | 0.9618 | 0.9616 | 0.9581 | 0.9593 |
| science | 0.7698 | 0.7606 | 0.7640 | 0.7606 | 0.7538 |
| coding | 0.5012 | 0.5001 | 0.4996 | 0.4694 | 0.5005 |

---

## 3. NAD Feature Statistics

| Domain | Feature | Mean | Std | Pct valid |
|--------|---------|------|-----|-----------|
| math | activated_neuron_count | 17341.7477 | 7141.0313 | 100.0% |
| math | neuron_density | 0.8277 | 0.1284 | 100.0% |
| math | topk10_neuron_count | 1738.9285 | 714.8521 | 100.0% |
| math | neuron_weight_entropy | 8.7710 | 0.3088 | 100.0% |
| math | mean_jaccard_sim | 0.4641 | 0.0452 | 100.0% |
| math | max_jaccard_sim | 0.5217 | 0.0492 | 100.0% |
| math | knn_agree_score | 0.5133 | 0.0487 | 100.0% |
| math | medoid_distance | 0.5359 | 0.0452 | 100.0% |
| math | minact_rank | 0.5000 | 0.2932 | 100.0% |
| science | activated_neuron_count | 8228.0063 | 3336.9426 | 100.0% |
| science | neuron_density | 0.7274 | 0.1563 | 100.0% |
| science | topk10_neuron_count | 826.3699 | 334.5896 | 100.0% |
| science | neuron_weight_entropy | 8.3811 | 0.2385 | 100.0% |
| science | mean_jaccard_sim | 0.4878 | 0.0571 | 100.0% |
| science | max_jaccard_sim | 0.5652 | 0.0459 | 100.0% |
| science | knn_agree_score | 0.5557 | 0.0456 | 100.0% |
| science | medoid_distance | 0.5122 | 0.0571 | 100.0% |
| science | minact_rank | 0.5000 | 0.2932 | 100.0% |
| coding | activated_neuron_count | 19170.4583 | 8298.5251 | 100.0% |
| coding | neuron_density | 0.7318 | 0.1629 | 100.0% |
| coding | topk10_neuron_count | 1922.5624 | 831.2593 | 100.0% |
| coding | neuron_weight_entropy | 8.9526 | 0.4430 | 100.0% |
| coding | mean_jaccard_sim | 0.4562 | 0.0484 | 100.0% |
| coding | max_jaccard_sim | 0.5286 | 0.0601 | 100.0% |
| coding | knn_agree_score | 0.5193 | 0.0575 | 100.0% |
| coding | medoid_distance | 0.5438 | 0.0484 | 100.0% |
| coding | minact_rank | 0.5000 | 0.2932 | 100.0% |

---

## 4. Unsupervised Baselines (StopAcc@100%)

| Domain | Selector | StopAcc@100% |
|--------|----------|-------------|
| math | MinAct | 0.6833 |
| math | Medoid | 0.7167 |
| math | kNN_agree | 0.7500 |
| science | MinAct | 0.6212 |
| science | Medoid | 0.6515 |
| science | kNN_agree | 0.6515 |
| coding | MinAct | 0.5689 |
| coding | Medoid | 0.5808 |
| coding | kNN_agree | 0.6048 |

---

## 5. Coding Domain Analysis

| Condition | Coding AUROC | Math AUROC | Science AUROC | Coding Δ vs C0 | Interpretation |
|-----------|-------------|-----------|--------------|----------------|----------------|
| canonical | 0.4910 | 0.9679 | 0.7798 | 0.0000 | within noise |
| canonical_plus_nc | 0.5062 | 0.9678 | 0.7757 | 0.0152 | meaningful gain |
| canonical_plus_nad_scalar | 0.5051 | 0.9679 | 0.7824 | 0.0141 | meaningful gain |
| canonical_plus_nad_agree | 0.4876 | 0.9688 | 0.7803 | -0.0034 | within noise |
| canonical_plus_all_nad | 0.5000 | 0.9711 | 0.7862 | 0.0090 | marginal gain |

---

## 6. Forced Verdict

**Q1 — Are NAD-lite agreement features (C3) better than NAD-lite scalars (C2)?**  
AUROC delta C3 − C2 (no_svd):
- math: Δ = +0.0009 → MARGINAL
- science: Δ = -0.0021 → NO
- coding: Δ = -0.0175 → NO

**Q2 — Does C4 (all NAD) beat C1 (nc only) on any domain by > 0.005 AUROC?**  
→ YES
- math: Δ = +0.0033
- science: Δ = +0.0105
- coding: Δ = -0.0062

**Q3 — Which domain benefits most from agreement features (C0 → C3)?**  
- math: Δ = +0.0009 ← BEST
- science: Δ = +0.0005
- coding: Δ = -0.0034

**Q4 — Do agreement features (C3) reduce coding instability vs C0?**  
C0 coding AUROC: 0.4910  
C3 coding AUROC: 0.4876  
Delta: -0.0034  
→ NO IMPROVEMENT

**Q5 — Is unsupervised Medoid competitive with C0 supervised on math/science?**  
Medoid StopAcc (math): 0.7167  vs  C0 AUROC (math): 0.9679  
Medoid StopAcc (science): 0.6515  vs  C0 AUROC (science): 0.7798  
→ (Note: StopAcc and AUROC are different metrics; direct comparison is approximate)

**Q6 — Paper framing**:  
→ future work: NAD agreement features do not improve over canonical at this threshold; revisit with rows/bank prefix-level Jaccard when available.

---

## 7. Limitations

- Agreement features operate at **full-sequence** position only.
  Prefix-level agreement (e.g. Jaccard at 50%) requires rows/bank keys, which
  are unavailable (`has_rows_bank=0` in all current caches).
- Singleton groups (n_runs < 2) have NaN agreement features, imputed with column mean.
- Full-sequence Jaccard may conflate domain-level neuron patterns with run-level signals.
- The roaring bitmap backend is not installed; NumPy backend is used (slower for large sets).

---

*Generated by SVDomain/run_nad_lite_agreement.py*
