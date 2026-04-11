# BESTOFN ES SVD MS RR R1 SLOT100 (2026-04-11)

## 目标

- 直接复用 `EarlyStop` 提交 `es_svd_ms_rr_r1__coding_from_round1c` 的 `100%` 槽位。
- 不训练新模型，不做 rank-scale，不做 cache patch。
- 产出一份纯 `slot100` 直抽的 `Best-of-N` submission，并记录盲榜结果。

## 输入

- `submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json`
  - 对每个 `cache / problem / sample` 直接提取第 `10` 个槽位，即 `slot_index=9`。
  - 该文件覆盖全部 `12` 个 blind-test caches。

## 输出

- 导出脚本：
  - `scripts/export_bestofn_from_earlystop_slot_with_bestofn_overrides.py`
- 导出文件：
  - `submission/BestofN/extreme12/patches/es_svd_ms_rr_r1__coding_from_round1c__slot100.json`
- `method_name`：
  - `es_svd_ms_rr_r1__coding_from_round1c__slot100`

## 规则

- 全部 `12` 个 cache 都直接来自源 `EarlyStop` 文件的 `100%` 槽位。
- `override-bestofn-json` 明确关闭：
  - `--override-bestofn-json none`
  - `--override-cache-keys none`
- 输出分数与源文件逐样本满足：
  - `best_of_n_score(cache, problem, sample) = early_stop_scores[cache][problem][sample][9]`

## 校验

- 源文件通过 `validate_earlystop_payload`
- 导出文件通过 `validate_submission_payload`
- 当前验证结果：
  - `cache_keys=12`
  - `problems=970`
  - `samples=62080`
- spot-check：
  - `DS-R1/aime24 / 60 / 0`
  - `DS-R1/gpqa / gpqa-0 / 0`
  - `DS-R1/lcb_v5 / livecodebench_v5-0 / 0`
  - `Qwen3-4B/hmmt25 / hmmt25-0 / 0`
  - 上述样本的导出分数均与源 `slot100` 完全一致。

## 盲榜结果

- `task`：`best_of_n`
- `method_name`：`es_svd_ms_rr_r1__coding_from_round1c__slot100`
- `submitted at`：`2026-04-11 02:44:11 UTC`
- `status`：`Not best`
- `avg rank`：`3.6000`
- `auroc`：`0.8468`
- `hit@1`：`0.7299`
- `hit@3`：`0.8010`
- `selacc@10%`：`0.9313`
- `pairwise acc`：`0.7396`
- `samples`：`62080`
- `problems`：`970`

### 当前 Per-Cache Breakdown

| Cache | AUROC | Hit@1 | Hit@3 | SelAcc@10% | Pairwise Acc | Problems | Samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| DS-R1/aime24 | 0.9639 | 0.8667 | 0.9000 | 1.0000 | 0.9157 | 30 | 1920 |
| DS-R1/aime25 | 0.9117 | 0.6333 | 0.7667 | 0.9948 | 0.6642 | 30 | 1920 |
| DS-R1/brumo25 | 0.9731 | 0.8667 | 0.9333 | 0.9948 | 0.8949 | 30 | 1920 |
| DS-R1/gpqa | 0.7776 | 0.6465 | 0.7626 | 0.9953 | 0.6230 | 198 | 12672 |
| DS-R1/hmmt25 | 0.9572 | 0.7333 | 0.7667 | 1.0000 | 0.8546 | 30 | 1920 |
| DS-R1/lcb_v5 | 0.5272 | 0.5988 | 0.7066 | 0.5796 | 0.5141 | 167 | 10688 |
| Qwen3-4B/aime24 | 0.9196 | 0.8667 | 0.9000 | 1.0000 | 0.8785 | 30 | 1920 |
| Qwen3-4B/aime25 | 0.9468 | 0.8000 | 0.8667 | 1.0000 | 0.8249 | 30 | 1920 |
| Qwen3-4B/brumo25 | 0.9367 | 0.8333 | 0.9000 | 1.0000 | 0.8387 | 30 | 1920 |
| Qwen3-4B/gpqa | 0.8009 | 0.6818 | 0.7576 | 0.9653 | 0.5848 | 198 | 12672 |
| Qwen3-4B/hmmt25 | 0.9396 | 0.6333 | 0.6333 | 1.0000 | 0.7822 | 30 | 1920 |
| Qwen3-4B/lcb_v5 | 0.5079 | 0.5988 | 0.7186 | 0.6461 | 0.5000 | 167 | 10688 |

## 对比历史最佳 Best-of-N

- `historical best`：`extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb`
- `historical best submitted at`：`2026-04-09 02:43 UTC`
- `historical best avg rank`：`2.4000`
- `historical best metrics`：
  - `hit@1 = 0.7504`
  - `hit@3 = 0.8049`
  - `selacc@10% = 0.9286`
  - `pairwise acc = 0.7510`

| Metric | Historical Best | Current Slot100 Export | Delta |
|---|---:|---:|---:|
| Avg Rank | 2.4000 | 3.6000 | +1.2000 |
| Hit@1 | 0.7504 | 0.7299 | -0.0205 |
| Hit@3 | 0.8049 | 0.8010 | -0.0039 |
| SelAcc@10% | 0.9286 | 0.9313 | +0.0027 |
| Pairwise Acc | 0.7510 | 0.7396 | -0.0114 |

### Historical Best Per-Cache Breakdown

| Cache | AUROC | Hit@1 | Hit@3 | SelAcc@10% | Pairwise Acc | Problems | Samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| DS-R1/aime24 | 0.9415 | 0.8667 | 0.9000 | 1.0000 | 0.9060 | 30 | 1920 |
| DS-R1/aime25 | 0.9288 | 0.7333 | 0.7333 | 0.9896 | 0.7018 | 30 | 1920 |
| DS-R1/brumo25 | 0.9478 | 0.9000 | 0.9000 | 0.9948 | 0.8973 | 30 | 1920 |
| DS-R1/gpqa | 0.7694 | 0.6667 | 0.7424 | 0.9803 | 0.6294 | 198 | 12672 |
| DS-R1/hmmt25 | 0.9383 | 0.7333 | 0.8000 | 0.9844 | 0.8710 | 30 | 1920 |
| DS-R1/lcb_v5 | 0.5043 | 0.5988 | 0.7066 | 0.6067 | 0.5141 | 167 | 10688 |
| Qwen3-4B/aime24 | 0.9687 | 0.9000 | 0.9000 | 1.0000 | 0.8957 | 30 | 1920 |
| Qwen3-4B/aime25 | 0.9558 | 0.8333 | 0.9000 | 1.0000 | 0.8137 | 30 | 1920 |
| Qwen3-4B/brumo25 | 0.9582 | 0.8333 | 0.9333 | 1.0000 | 0.8792 | 30 | 1920 |
| Qwen3-4B/gpqa | 0.8068 | 0.7071 | 0.7576 | 0.9755 | 0.5937 | 198 | 12672 |
| Qwen3-4B/hmmt25 | 0.9398 | 0.6333 | 0.6667 | 1.0000 | 0.8103 | 30 | 1920 |
| Qwen3-4B/lcb_v5 | 0.5001 | 0.5988 | 0.7186 | 0.6114 | 0.5000 | 167 | 10688 |

## 结论

- 这份 `slot100` 直抽版本是一个合法且可复现的 `Best-of-N` 提交。
- 相比历史最佳，它只在 `SelAcc@10%` 上略有优势。
- 真正拖累总排名的是：
  - `Hit@1`
  - `Hit@3`
  - `Pairwise Acc`
- 因此它更适合作为：
  - `EarlyStop -> Best-of-N` 的桥接基线
  - `slot100` 纯抽取对照
  - 后续带 `lcb bridge` 或其他 patch 的比较底座

## 复跑

```bash
python3 scripts/export_bestofn_from_earlystop_slot_with_bestofn_overrides.py \
  --earlystop-json submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json \
  --slot-index 9 \
  --override-bestofn-json none \
  --override-cache-keys none \
  --out submission/BestofN/extreme12/patches/es_svd_ms_rr_r1__coding_from_round1c__slot100.json \
  --method-name es_svd_ms_rr_r1__coding_from_round1c__slot100
```
