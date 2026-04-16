# Domain-Specific Contrastive SSL for SVDomain

Generated: 2026-04-15T09:12:20.327996+00:00

## Training Configuration

- Epochs: 300  |  Temperature τ: 0.07
- Optimizer: Adam with cosine LR annealing (lr_max=0.01, lr_min=1e-4)
- Gradient clipping: Frobenius norm = 1.0
- Warm-start: top-r SVD singular vectors per domain

## Results

| Domain | Condition | SSL r | Label % | AUROC | Top1 Acc | AUC SelAcc |
|--------|-----------|:-----:|--------:|------:|---------:|-----------:|
| coding | domain_ssl | 4 | 1% | 0.5092 | 0.5600 | 0.5356 |
| coding | domain_ssl | 4 | 5% | 0.5054 | 0.5600 | 0.5331 |
| coding | domain_ssl | 4 | 10% | 0.5017 | 0.6400 | 0.5169 |
| coding | domain_ssl | 4 | 20% | 0.4978 | 0.4000 | 0.5237 |
| coding | domain_ssl | 4 | 50% | 0.5026 | 0.4800 | 0.5050 |
| coding | domain_ssl | 4 | 100% | 0.5093 | 0.5200 | 0.5025 |
| coding | domain_ssl | 8 | 1% | 0.5040 | 0.4800 | 0.5169 |
| coding | domain_ssl | 8 | 5% | 0.5024 | 0.4800 | 0.5012 |
| coding | domain_ssl | 8 | 10% | 0.5009 | 0.5200 | 0.5081 |
| coding | domain_ssl | 8 | 20% | 0.4920 | 0.4800 | 0.4681 |
| coding | domain_ssl | 8 | 50% | 0.4936 | 0.4800 | 0.5038 |
| coding | domain_ssl | 8 | 100% | 0.4967 | 0.6000 | 0.4787 |
| coding | domain_ssl | 16 | 1% | 0.5045 | 0.4800 | 0.5319 |
| coding | domain_ssl | 16 | 5% | 0.5003 | 0.4800 | 0.5319 |
| coding | domain_ssl | 16 | 10% | 0.4944 | 0.5600 | 0.4912 |
| coding | domain_ssl | 16 | 20% | 0.4891 | 0.5600 | 0.4838 |
| coding | domain_ssl | 16 | 50% | 0.4900 | 0.5600 | 0.4881 |
| coding | domain_ssl | 16 | 100% | 0.4950 | 0.4000 | 0.4744 |
| coding | domain_ssl_weak | 4 | 1% | 0.5093 | 0.5600 | 0.5356 |
| coding | domain_ssl_weak | 4 | 5% | 0.5054 | 0.5600 | 0.5312 |
| coding | domain_ssl_weak | 4 | 10% | 0.5018 | 0.6400 | 0.5188 |
| coding | domain_ssl_weak | 4 | 20% | 0.4979 | 0.4000 | 0.5262 |
| coding | domain_ssl_weak | 4 | 50% | 0.5025 | 0.5200 | 0.5062 |
| coding | domain_ssl_weak | 4 | 100% | 0.5093 | 0.5200 | 0.5012 |
| coding | domain_ssl_weak | 8 | 1% | 0.5035 | 0.4800 | 0.5169 |
| coding | domain_ssl_weak | 8 | 5% | 0.5026 | 0.4800 | 0.4975 |
| coding | domain_ssl_weak | 8 | 10% | 0.5013 | 0.5200 | 0.5056 |
| coding | domain_ssl_weak | 8 | 20% | 0.4918 | 0.4800 | 0.4650 |
| coding | domain_ssl_weak | 8 | 50% | 0.4923 | 0.4800 | 0.4994 |
| coding | domain_ssl_weak | 8 | 100% | 0.4960 | 0.5600 | 0.4731 |
| coding | domain_ssl_weak | 16 | 1% | 0.5042 | 0.4800 | 0.5294 |
| coding | domain_ssl_weak | 16 | 5% | 0.5005 | 0.4800 | 0.5306 |
| coding | domain_ssl_weak | 16 | 10% | 0.4940 | 0.5600 | 0.4869 |
| coding | domain_ssl_weak | 16 | 20% | 0.4893 | 0.5600 | 0.4869 |
| coding | domain_ssl_weak | 16 | 50% | 0.4907 | 0.5600 | 0.4856 |
| coding | domain_ssl_weak | 16 | 100% | 0.4951 | 0.4000 | 0.4681 |
| coding | no_svd_lr | -1 | 1% | 0.4753 | 0.4800 | 0.4088 |
| coding | no_svd_lr | -1 | 5% | 0.5573 | 0.6000 | 0.4400 |
| coding | no_svd_lr | -1 | 10% | 0.5911 | 0.5200 | 0.5331 |
| coding | no_svd_lr | -1 | 20% | 0.5706 | 0.5200 | 0.5019 |
| coding | no_svd_lr | -1 | 50% | 0.4672 | 0.5600 | 0.3525 |
| coding | no_svd_lr | -1 | 100% | 0.4797 | 0.6000 | 0.4725 |
| coding | shared_ssl_r16 | 16 | 1% | 0.4983 | 0.5200 | 0.5175 |
| coding | shared_ssl_r16 | 16 | 5% | 0.4982 | 0.5200 | 0.5169 |
| coding | shared_ssl_r16 | 16 | 10% | 0.5041 | 0.5200 | 0.5162 |
| coding | shared_ssl_r16 | 16 | 20% | 0.5056 | 0.5200 | 0.5125 |
| coding | shared_ssl_r16 | 16 | 50% | 0.5011 | 0.5200 | 0.5287 |
| coding | shared_ssl_r16 | 16 | 100% | 0.5020 | 0.5200 | 0.5269 |
| math | domain_ssl | 4 | 1% | 0.6986 | 0.7500 | 0.9533 |
| math | domain_ssl | 4 | 5% | 0.8867 | 0.7857 | 0.9808 |
| math | domain_ssl | 4 | 10% | 0.8797 | 0.7857 | 0.9841 |
| math | domain_ssl | 4 | 20% | 0.8807 | 0.7857 | 0.9896 |
| math | domain_ssl | 4 | 50% | 0.8760 | 0.7857 | 0.9934 |
| math | domain_ssl | 4 | 100% | 0.8855 | 0.7857 | 0.9857 |
| math | domain_ssl | 8 | 1% | 0.5881 | 0.7500 | 0.9456 |
| math | domain_ssl | 8 | 5% | 0.7583 | 0.7500 | 0.9951 |
| math | domain_ssl | 8 | 10% | 0.8843 | 0.7500 | 0.9868 |
| math | domain_ssl | 8 | 20% | 0.8824 | 0.7857 | 0.9929 |
| math | domain_ssl | 8 | 50% | 0.8720 | 0.7500 | 0.9940 |
| math | domain_ssl | 8 | 100% | 0.8803 | 0.7857 | 0.9879 |
| math | domain_ssl | 16 | 1% | 0.5897 | 0.7500 | 0.9544 |
| math | domain_ssl | 16 | 5% | 0.7629 | 0.7500 | 0.9951 |
| math | domain_ssl | 16 | 10% | 0.8859 | 0.7500 | 0.9857 |
| math | domain_ssl | 16 | 20% | 0.8812 | 0.7500 | 0.9912 |
| math | domain_ssl | 16 | 50% | 0.8688 | 0.7143 | 0.9918 |
| math | domain_ssl | 16 | 100% | 0.8757 | 0.7500 | 0.9841 |
| math | frozen_svd | -1 | 1% | 0.7223 | 0.7500 | 0.9637 |
| math | frozen_svd | -1 | 5% | 0.9396 | 0.7143 | 0.9962 |
| math | frozen_svd | -1 | 10% | 0.9450 | 0.7143 | 0.9912 |
| math | frozen_svd | -1 | 20% | 0.9529 | 0.7143 | 0.9951 |
| math | frozen_svd | -1 | 50% | 0.9574 | 0.7500 | 0.9973 |
| math | frozen_svd | -1 | 100% | 0.9582 | 0.7500 | 0.9978 |
| math | no_svd_lr | -1 | 1% | 0.8824 | 0.7500 | 0.9923 |
| math | no_svd_lr | -1 | 5% | 0.9468 | 0.7143 | 0.9962 |
| math | no_svd_lr | -1 | 10% | 0.9506 | 0.7143 | 0.9940 |
| math | no_svd_lr | -1 | 20% | 0.9553 | 0.7143 | 0.9967 |
| math | no_svd_lr | -1 | 50% | 0.9590 | 0.7143 | 0.9978 |
| math | no_svd_lr | -1 | 100% | 0.9594 | 0.7143 | 0.9973 |
| math | shared_ssl_r16 | 16 | 1% | 0.5504 | 0.7143 | 0.7643 |
| math | shared_ssl_r16 | 16 | 5% | 0.6047 | 0.7500 | 0.8429 |
| math | shared_ssl_r16 | 16 | 10% | 0.6523 | 0.7500 | 0.8824 |
| math | shared_ssl_r16 | 16 | 20% | 0.6708 | 0.7500 | 0.9022 |
| math | shared_ssl_r16 | 16 | 50% | 0.7117 | 0.7500 | 0.9319 |
| math | shared_ssl_r16 | 16 | 100% | 0.7794 | 0.7500 | 0.9555 |
| science | domain_ssl | 4 | 1% | 0.6417 | 0.6500 | 0.7839 |
| science | domain_ssl | 4 | 5% | 0.7729 | 0.6167 | 0.9156 |
| science | domain_ssl | 4 | 10% | 0.7966 | 0.6500 | 0.9513 |
| science | domain_ssl | 4 | 20% | 0.8073 | 0.6500 | 0.9648 |
| science | domain_ssl | 4 | 50% | 0.8106 | 0.6500 | 0.9661 |
| science | domain_ssl | 4 | 100% | 0.8175 | 0.6500 | 0.9753 |
| science | domain_ssl | 8 | 1% | 0.6063 | 0.6833 | 0.7609 |
| science | domain_ssl | 8 | 5% | 0.7383 | 0.7000 | 0.8857 |
| science | domain_ssl | 8 | 10% | 0.7671 | 0.6833 | 0.9513 |
| science | domain_ssl | 8 | 20% | 0.7786 | 0.6833 | 0.9680 |
| science | domain_ssl | 8 | 50% | 0.7839 | 0.6667 | 0.9609 |
| science | domain_ssl | 8 | 100% | 0.7962 | 0.6833 | 0.9708 |
| science | domain_ssl | 16 | 1% | 0.6214 | 0.6833 | 0.7695 |
| science | domain_ssl | 16 | 5% | 0.7698 | 0.6167 | 0.8898 |
| science | domain_ssl | 16 | 10% | 0.8096 | 0.6000 | 0.9477 |
| science | domain_ssl | 16 | 20% | 0.8289 | 0.6333 | 0.9531 |
| science | domain_ssl | 16 | 50% | 0.8340 | 0.6333 | 0.9555 |
| science | domain_ssl | 16 | 100% | 0.8452 | 0.6667 | 0.9664 |
| science | frozen_svd | -1 | 1% | 0.5930 | 0.6667 | 0.8391 |
| science | frozen_svd | -1 | 5% | 0.7519 | 0.7167 | 0.9589 |
| science | frozen_svd | -1 | 10% | 0.7769 | 0.7167 | 0.9667 |
| science | frozen_svd | -1 | 20% | 0.7700 | 0.7000 | 0.9755 |
| science | frozen_svd | -1 | 50% | 0.7853 | 0.7000 | 0.9776 |
| science | frozen_svd | -1 | 100% | 0.7944 | 0.7333 | 0.9846 |
| science | no_svd_lr | -1 | 1% | 0.4422 | 0.6667 | 0.7146 |
| science | no_svd_lr | -1 | 5% | 0.7895 | 0.6667 | 0.9680 |
| science | no_svd_lr | -1 | 10% | 0.7962 | 0.7000 | 0.9773 |
| science | no_svd_lr | -1 | 20% | 0.8014 | 0.7000 | 0.9784 |
| science | no_svd_lr | -1 | 50% | 0.8161 | 0.7167 | 0.9807 |
| science | no_svd_lr | -1 | 100% | 0.8320 | 0.7000 | 0.9852 |
| science | shared_ssl_r16 | 16 | 1% | 0.4147 | 0.6167 | 0.6357 |
| science | shared_ssl_r16 | 16 | 5% | 0.7346 | 0.7000 | 0.9661 |
| science | shared_ssl_r16 | 16 | 10% | 0.6735 | 0.7000 | 0.9508 |
| science | shared_ssl_r16 | 16 | 20% | 0.7416 | 0.7167 | 0.9716 |
| science | shared_ssl_r16 | 16 | 50% | 0.7760 | 0.6833 | 0.9828 |
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
