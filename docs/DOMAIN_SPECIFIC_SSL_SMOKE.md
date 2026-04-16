# Domain-Specific Contrastive SSL for SVDomain

Generated: 2026-04-15T08:54:46.891825+00:00

## Training Configuration

- Epochs: 20  |  Temperature τ: 0.07
- Optimizer: Adam with cosine LR annealing (lr_max=0.01, lr_min=1e-4)
- Gradient clipping: Frobenius norm = 1.0
- Warm-start: top-r SVD singular vectors per domain

## Results

| Domain | Condition | SSL r | Label % | AUROC | Top1 Acc | AUC SelAcc |
|--------|-----------|:-----:|--------:|------:|---------:|-----------:|
| coding | domain_ssl | 4 | 10% | 0.5059 | 0.4800 | 0.5269 |
| coding | domain_ssl | 4 | 50% | 0.5081 | 0.6000 | 0.4919 |
| coding | domain_ssl | 4 | 100% | 0.5109 | 0.5600 | 0.5106 |
| coding | domain_ssl | 8 | 10% | 0.4927 | 0.6000 | 0.4800 |
| coding | domain_ssl | 8 | 50% | 0.4910 | 0.5200 | 0.4694 |
| coding | domain_ssl | 8 | 100% | 0.4978 | 0.4800 | 0.4912 |
| coding | domain_ssl_weak | 4 | 10% | 0.5061 | 0.4800 | 0.5212 |
| coding | domain_ssl_weak | 4 | 50% | 0.5075 | 0.6000 | 0.4881 |
| coding | domain_ssl_weak | 4 | 100% | 0.5115 | 0.5600 | 0.5075 |
| coding | domain_ssl_weak | 8 | 10% | 0.4923 | 0.6000 | 0.4781 |
| coding | domain_ssl_weak | 8 | 50% | 0.4923 | 0.4800 | 0.4687 |
| coding | domain_ssl_weak | 8 | 100% | 0.4990 | 0.4400 | 0.4869 |
| coding | no_svd_lr | -1 | 10% | 0.5321 | 0.4400 | 0.5331 |
| coding | no_svd_lr | -1 | 50% | 0.4955 | 0.6400 | 0.5175 |
| coding | no_svd_lr | -1 | 100% | 0.4797 | 0.6000 | 0.4725 |
| coding | shared_ssl_r16 | 16 | 10% | 0.5009 | 0.4800 | 0.5325 |
| coding | shared_ssl_r16 | 16 | 50% | 0.5012 | 0.5200 | 0.5144 |
| coding | shared_ssl_r16 | 16 | 100% | 0.5020 | 0.5200 | 0.5269 |
| math | domain_ssl | 4 | 10% | 0.8769 | 0.7500 | 0.9890 |
| math | domain_ssl | 4 | 50% | 0.8857 | 0.7857 | 0.9857 |
| math | domain_ssl | 4 | 100% | 0.8844 | 0.7857 | 0.9868 |
| math | domain_ssl | 8 | 10% | 0.8357 | 0.7143 | 0.9940 |
| math | domain_ssl | 8 | 50% | 0.8735 | 0.7143 | 0.9841 |
| math | domain_ssl | 8 | 100% | 0.8746 | 0.7143 | 0.9835 |
| math | frozen_svd | -1 | 10% | 0.9460 | 0.7143 | 0.9934 |
| math | frozen_svd | -1 | 50% | 0.9569 | 0.7500 | 0.9978 |
| math | frozen_svd | -1 | 100% | 0.9582 | 0.7500 | 0.9978 |
| math | no_svd_lr | -1 | 10% | 0.9506 | 0.7500 | 0.9978 |
| math | no_svd_lr | -1 | 50% | 0.9589 | 0.7143 | 0.9973 |
| math | no_svd_lr | -1 | 100% | 0.9594 | 0.7143 | 0.9973 |
| math | shared_ssl_r16 | 16 | 10% | 0.6577 | 0.7500 | 0.8929 |
| math | shared_ssl_r16 | 16 | 50% | 0.7283 | 0.7500 | 0.9423 |
| math | shared_ssl_r16 | 16 | 100% | 0.7794 | 0.7500 | 0.9555 |
| science | domain_ssl | 4 | 10% | 0.8221 | 0.7167 | 0.9852 |
| science | domain_ssl | 4 | 50% | 0.8141 | 0.6833 | 0.9641 |
| science | domain_ssl | 4 | 100% | 0.8173 | 0.6667 | 0.9708 |
| science | domain_ssl | 8 | 10% | 0.7891 | 0.7333 | 0.9701 |
| science | domain_ssl | 8 | 50% | 0.7996 | 0.7000 | 0.9620 |
| science | domain_ssl | 8 | 100% | 0.8023 | 0.6667 | 0.9716 |
| science | frozen_svd | -1 | 10% | 0.8021 | 0.7167 | 0.9807 |
| science | frozen_svd | -1 | 50% | 0.7893 | 0.7000 | 0.9885 |
| science | frozen_svd | -1 | 100% | 0.7944 | 0.7333 | 0.9846 |
| science | no_svd_lr | -1 | 10% | 0.7868 | 0.7167 | 0.9484 |
| science | no_svd_lr | -1 | 50% | 0.8426 | 0.6833 | 0.9763 |
| science | no_svd_lr | -1 | 100% | 0.8320 | 0.7000 | 0.9852 |
| science | shared_ssl_r16 | 16 | 10% | 0.6639 | 0.7000 | 0.9508 |
| science | shared_ssl_r16 | 16 | 50% | 0.7602 | 0.7000 | 0.9771 |
| science | shared_ssl_r16 | 16 | 100% | 0.7876 | 0.6833 | 0.9872 |

## Research Questions

### Q1: Was the failure due to objective, cross-domain mixing, or both?
Compare `domain_ssl` vs `shared_ssl_r16` at 100% labels per domain.
If `domain_ssl` substantially outperforms → cross-domain mixing was primary cause.
If both still trail `no_svd_lr` → linear-basis capacity is the primary cause.

### Q2: Does contrastive/domain-specific SSL help at low labels?
Compare `domain_ssl_r{4,8,16}` vs `no_svd_lr` vs `frozen_svd` at 1–10%.
If `domain_ssl` ≥ `frozen_svd` at ≤5% labels → SSL improves label efficiency.

### Q3: Does coding benefit from pairwise weak supervision?
Compare `domain_ssl_weak` vs `domain_ssl` on coding across all label fractions.
Positive delta → pairwise hinge during pre-training helps.

### Q4: Is the per-domain basis more useful than the shared basis?
AUROC gap `domain_ssl_r16 − shared_ssl_r16` at 100% labels per domain.
Positive gap → per-domain training improves basis quality.

## Conditions

| Condition | Description |
|-----------|-------------|
| `domain_ssl` | Per-domain NT-Xent basis + supervised head |
| `domain_ssl_weak` | Same + pairwise hinge during coding pre-training |
| `shared_ssl_r16` | Old shared SSL basis (r=16) loaded from pkl |
| `no_svd_lr` | StandardScaler → LR, no dimensionality reduction |
| `frozen_svd` | Frozen supervised SVD + new LR head |

## Domain Configuration

| Domain | NT-Xent views (early, late) | Pairwise hinge | Batch |
|--------|----------------------------|----------------|-------|
| math    | 10%, 100% | No  | 256 |
| science | 40%, 100% | No  | 256 |
| coding  | 40%, 70%  | Yes (domain_ssl_weak only) | 384 |

## SSL Losses

1. `L_ntxent` (primary): symmetric NT-Xent, τ=0.07, positives = same run two anchors
2. `L_pred` (λ=0.3): MSE future-anchor prediction Z_early @ W_future ≈ Z_late
3. `L_view` (λ=0.1): raw-vs-rank view consistency ||X_rw@B[:22] − X_rk@B[22:]||²
4. `L_pair` (λ=0.5, coding weak): pairwise hinge on 40% anchor within problem groups
5. `L_proxy` (λ=0.1, coding, optional): MSE decode tier-1 features from Z
