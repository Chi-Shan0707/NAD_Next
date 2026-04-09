# BESTOFN EARLYSTOP SVD100 + SVM-LCB BRIDGE (2026-04-09)

## 目标

- 不训练新模型。
- 直接复用现有 `EarlyStop` 导出物中的 `100%` 槽位，生成一份 `Best-of-N` 提交文件。
- `lcb_v5` 不使用 `SVD100` 分数，直接沿用现有 `SVM bridge` 的 `Best-of-N` 分数。

## 输入

- `submission/EarlyStop/earlystop_prefix10_svd_round1.json`
  - 用作非 `lcb` cache 的分数来源。
  - 对每个 `cache / problem / sample` 提取第 `10` 个槽位，即 `slot_index=9`。
- `submission/BestofN/extreme12/patches/extreme12_svm_bridge_bestofn_v1.json`
  - 只用来覆盖：
    - `DS-R1/lcb_v5`
    - `Qwen3-4B/lcb_v5`

## 输出

- 导出脚本：
  - `scripts/export_bestofn_from_earlystop_slot_with_bestofn_overrides.py`
- 导出文件：
  - `submission/BestofN/extreme12/patches/extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb.json`
- `method_name`：
  - `extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb`

## 规则

- 非 `lcb` 的 `10` 个 cache：
  - 直接使用 `earlystop_prefix10_svd_round1` 的 `100%` 分数。
- `lcb` 的 `2` 个 cache：
  - 直接整块复制 `extreme12_svm_bridge_bestofn_v1.json` 中的对应分数。
- 不做额外 rank-scale，不做再训练，不改其它 patch 逻辑。

## 校验

- `EarlyStop` 源文件先过 `validate_earlystop_payload`
- `SVM bridge Best-of-N` 源文件先过 `validate_submission_payload`
- 新导出文件再过 `validate_submission_payload`
- 当前验证结果：
  - `cache_keys=12`
  - `problems=970`
  - `samples=62080`

## 复跑

```bash
python3 scripts/export_bestofn_from_earlystop_slot_with_bestofn_overrides.py
```
