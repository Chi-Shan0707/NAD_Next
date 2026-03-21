"""
ShardReader: 读取单个 NPZ shard，并按 **sample_id** 聚合所有 slices，
返回"按 sample 分段、每段内 key 升序去重"的 partial-CSR：
  row_ptr_s: int64 [num_samples+1]
  keys_s:   uint32 [total_K_in_this_shard]
  wmax_s:   float32 同上
  wsum_s:   float32 同上

修复说明：
- 原实现错误地把每个 (sample_id, slice_id) 当作独立的 run
- 正确做法：把同一 sample_id 的所有 slices 聚合成一个 run
- 数据层次：30 problems → 1920 samples → 每个sample约460 slices
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, List
import os


def _dedup_agg_sorted(keys: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    要求 keys 已按升序排序；返回唯一 key 及对应 wmax/wsum。

    Args:
        keys: 升序排列的键数组
        scores: 对应的分数数组

    Returns:
        (unique_keys, wmax, wsum) - 去重后的键和聚合的max/sum权重
    """
    if keys.size == 0:
        return (keys.astype(np.uint32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    # 找到边界
    diff = np.empty_like(keys, dtype=bool)
    diff[0] = True
    diff[1:] = keys[1:] != keys[:-1]
    idx = np.nonzero(diff)[0]

    # 聚合
    wsum = np.add.reduceat(scores, idx)
    wmax = np.maximum.reduceat(scores, idx)
    uniq = keys[idx].astype(np.uint32, copy=False)

    return uniq, wmax.astype(np.float32, copy=False), wsum.astype(np.float32, copy=False)


def _flatten_rows(keys_list: List[np.ndarray], scores_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """合并多个keys/scores列表"""
    if not keys_list:
        return (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32))
    keys = np.concatenate(keys_list).astype(np.int64, copy=False)
    scores = np.concatenate(scores_list).astype(np.float32, copy=False)
    return keys, scores


def read_shard_grouped_by_sample(npz_path: str, num_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 shard，输出 (row_ptr_s, keys_s, wmax_s, wsum_s)

    核心逻辑：
    1. 读取NPZ文件中的所有数据行（每行是一个slice）
    2. 按 sample_id 分组收集所有slices的neuron数据
    3. 对每个sample：合并所有slices → 排序 → 去重聚合
    4. 输出sample级别的partial-CSR格式

    Args:
        npz_path: NPZ shard文件路径
        num_samples: 总样本数（通常是1920）

    Returns:
        row_ptr_s: [num_samples+1] 每个sample的起始位置
        keys_s: [total_keys] 所有packed keys (layer<<16 | neuron)
        wmax_s: [total_keys] 对应的max权重
        wsum_s: [total_keys] 对应的sum权重

    注意：
    - 若某 sample 在该 shard 无数据，则对应段长为 0
    - keys 在每个sample内部是升序唯一的
    """
    data = np.load(npz_path, allow_pickle=False)

    # 尝试识别字段（兼容多种格式）
    # 期望字段：idx_samples（sample IDs）, slice_ids, layers, neurons, scores

    sample_ids = None
    if "idx_samples" in data:
        sample_ids = data["idx_samples"].astype(np.int64)
    elif "sample_id" in data:
        sample_ids = data["sample_id"].astype(np.int64)
    elif "sample_ids" in data:
        sample_ids = data["sample_ids"].astype(np.int64)

    if sample_ids is None:
        raise ValueError(f"Cannot find sample_id field in {npz_path}. Available: {list(data.keys())}")

    # 检查是否已经是sample级聚合格式（理想情况，直接返回）
    if "row_ptr_s" in data and "keys_s" in data and "wmax_s" in data and "wsum_s" in data:
        row_ptr_s = data["row_ptr_s"].astype(np.int64)
        keys_s = data["keys_s"].astype(np.uint32)
        wmax_s = data["wmax_s"].astype(np.float32)
        wsum_s = data["wsum_s"].astype(np.float32)

        # 确保长度匹配
        if row_ptr_s.size != num_samples + 1:
            pad = np.zeros((num_samples + 1,), dtype=np.int64)
            m = min(row_ptr_s.size, pad.size)
            pad[:m] = row_ptr_s[:m]
            row_ptr_s = pad

        return row_ptr_s, keys_s, wmax_s, wsum_s

    # 处理原始slice级数据
    # 需要字段：layers, neurons, scores（每个都是 [num_rows, 500] 或类似维度）
    if "layers" not in data or "neurons" not in data or "scores" not in data:
        raise ValueError(f"Missing required fields (layers/neurons/scores) in {npz_path}")

    layers = data["layers"]  # [N, K] where K could be variable (500 typical)
    neurons = data["neurons"]  # [N, K]
    scores = data["scores"].astype(np.float32)  # [N, K]

    # 汇聚到 sample：收集每个 sample 的所有 slices 的 keys/scores
    per_sample_keys: List[List[np.ndarray]] = [[] for _ in range(num_samples)]
    per_sample_scores: List[List[np.ndarray]] = [[] for _ in range(num_samples)]

    num_rows = sample_ids.size

    for i in range(num_rows):
        s = int(sample_ids[i])

        # 跳过超出范围的sample_id
        if s < 0 or s >= num_samples:
            continue

        # 提取该行的有效neuron数据
        # 通常 layers/neurons/scores 的第二维度是固定的（如500），但可能有填充的-1
        layer_row = layers[i]
        neuron_row = neurons[i]
        score_row = scores[i]

        # 过滤有效的entries（layer >= 0 表示有效）
        valid_mask = layer_row >= 0

        if not np.any(valid_mask):
            continue

        valid_layers = layer_row[valid_mask].astype(np.int64)
        valid_neurons = neuron_row[valid_mask].astype(np.int64)
        valid_scores = score_row[valid_mask]

        # Pack keys: (layer << 16) | neuron
        # 假设layer < 65536, neuron < 65536
        packed_keys = (valid_layers << 16) | valid_neurons

        per_sample_keys[s].append(packed_keys)
        per_sample_scores[s].append(valid_scores)

    # 对每个sample做排序、去重与聚合
    row_ptr_s = np.zeros((num_samples + 1,), dtype=np.int64)
    keys_blocks: List[np.ndarray] = []
    wmax_blocks: List[np.ndarray] = []
    wsum_blocks: List[np.ndarray] = []

    acc = 0

    for s in range(num_samples):
        keys_list = per_sample_keys[s]
        scores_list = per_sample_scores[s]

        if not keys_list:
            # 该sample在此shard中无数据
            row_ptr_s[s + 1] = acc
            continue

        # 合并该sample的所有slices
        kcat, scat = _flatten_rows(keys_list, scores_list)

        # 按 key 排序
        order = np.argsort(kcat, kind="mergesort")
        k_sorted = kcat[order]
        s_sorted = scat[order]

        # 去重聚合
        uniq, wmax, wsum = _dedup_agg_sorted(k_sorted, s_sorted)

        if uniq.size > 0:
            keys_blocks.append(uniq)
            wmax_blocks.append(wmax)
            wsum_blocks.append(wsum)
            acc += uniq.size

        row_ptr_s[s + 1] = acc

    # 合并所有blocks
    if acc == 0:
        return (row_ptr_s,
                np.empty((0,), dtype=np.uint32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    keys_s = np.concatenate(keys_blocks).astype(np.uint32, copy=False)
    wmax_s = np.concatenate(wmax_blocks).astype(np.float32, copy=False)
    wsum_s = np.concatenate(wsum_blocks).astype(np.float32, copy=False)

    return row_ptr_s, keys_s, wmax_s, wsum_s


def read_shard_tokens_by_sample(npz_path: str, num_samples: int) -> Tuple[np.ndarray, ...]:
    """
    Extract token metadata from a shard, grouped by sample.

    Returns:
        Tuple of (tok_row_ptr_s, token_ids_s, tok_conf_s, tok_neg_entropy_s,
                  tok_gini_s, tok_selfcert_s, tok_logprob_s)

        - tok_row_ptr_s: int64 [num_samples+1] - cumulative token counts
        - token_ids_s: int32 [total_tokens] or None
        - tok_conf_s: float32 [total_tokens] or None
        - tok_neg_entropy_s: float32 [total_tokens] or None
        - tok_gini_s: float32 [total_tokens] or None
        - tok_selfcert_s: float32 [total_tokens] or None
        - tok_logprob_s: float32 [total_tokens] or None

    If token data is not available, returns (zero_row_ptr, None, None, None, None, None, None)
    """
    data = np.load(npz_path, allow_pickle=False)

    # Check if token data exists
    required_fields = ["idx_samples", "slice_ids", "token_row_ptr", "token_ids"]
    if not all(f in data for f in required_fields):
        # No token data available
        zero_ptr = np.zeros((num_samples + 1,), dtype=np.int64)
        return (zero_ptr, None, None, None, None, None, None)

    sample_ids = data["idx_samples"].astype(np.int64)
    slice_ids = data["slice_ids"].astype(np.int64)
    row_ptr_tok = data["token_row_ptr"].astype(np.int64)
    tok_ids_all = data["token_ids"].astype(np.int32)

    # Optional token features (may not exist in all NPZ files)
    tok_conf_all = data.get("tok_conf")
    if tok_conf_all is not None:
        tok_conf_all = tok_conf_all.astype(np.float32)

    # Try tok_neg_entropy first, fallback to tok_entropy
    tok_neg_ent_all = data.get("tok_neg_entropy")
    if tok_neg_ent_all is None:
        tok_neg_ent_all = data.get("tok_entropy")
    if tok_neg_ent_all is not None:
        tok_neg_ent_all = tok_neg_ent_all.astype(np.float32)

    tok_gini_all = data.get("tok_gini")
    if tok_gini_all is not None:
        tok_gini_all = tok_gini_all.astype(np.float32)

    tok_self_all = data.get("tok_selfcert")
    if tok_self_all is not None:
        tok_self_all = tok_self_all.astype(np.float32)

    tok_logp_all = data.get("tok_logprob")
    if tok_logp_all is not None:
        tok_logp_all = tok_logp_all.astype(np.float32)

    # Group rows by sample and sort by slice_id to maintain token order
    per_sample_rows: List[List[int]] = [[] for _ in range(num_samples)]
    num_rows = sample_ids.size

    for i in range(num_rows):
        s = int(sample_ids[i])
        if 0 <= s < num_samples:
            per_sample_rows[s].append(i)

    # Sort each sample's rows by slice_id to preserve token sequence order
    for s in range(num_samples):
        if per_sample_rows[s]:
            per_sample_rows[s].sort(key=lambda i: int(slice_ids[i]))

    # Count total tokens per sample to build row_ptr
    tok_row_ptr_s = np.zeros((num_samples + 1,), dtype=np.int64)
    lengths = np.zeros((num_samples,), dtype=np.int64)
    total_tokens = 0

    for s in range(num_samples):
        n = 0
        for i in per_sample_rows[s]:
            a, b = int(row_ptr_tok[i]), int(row_ptr_tok[i + 1])
            n += max(0, b - a)
        lengths[s] = n
        total_tokens += n
        tok_row_ptr_s[s + 1] = total_tokens

    if total_tokens == 0:
        # No tokens in this shard
        return (tok_row_ptr_s, None, None, None, None, None, None)

    # Allocate output arrays
    token_ids_s = np.empty((total_tokens,), dtype=np.int32)

    def maybe_alloc(src, dtype=np.float32):
        return np.empty((total_tokens,), dtype=dtype) if src is not None else None

    tok_conf_s = maybe_alloc(tok_conf_all)
    tok_neg_ent_s = maybe_alloc(tok_neg_ent_all)
    tok_gini_s = maybe_alloc(tok_gini_all)
    tok_self_s = maybe_alloc(tok_self_all)
    tok_logp_s = maybe_alloc(tok_logp_all)

    # Fill arrays by concatenating token segments for each sample
    for s in range(num_samples):
        if lengths[s] == 0:
            continue

        a = int(tok_row_ptr_s[s])
        b = int(tok_row_ptr_s[s + 1])
        pos = a

        for i in per_sample_rows[s]:
            u, v = int(row_ptr_tok[i]), int(row_ptr_tok[i + 1])
            if u == v:
                continue

            seg_len = v - u
            token_ids_s[pos:pos + seg_len] = tok_ids_all[u:v]

            if tok_conf_s is not None:
                tok_conf_s[pos:pos + seg_len] = tok_conf_all[u:v]
            if tok_neg_ent_s is not None:
                tok_neg_ent_s[pos:pos + seg_len] = tok_neg_ent_all[u:v]
            if tok_gini_s is not None:
                tok_gini_s[pos:pos + seg_len] = tok_gini_all[u:v]
            if tok_self_s is not None:
                tok_self_s[pos:pos + seg_len] = tok_self_all[u:v]
            if tok_logp_s is not None:
                tok_logp_s[pos:pos + seg_len] = tok_logp_all[u:v]

            pos += seg_len

    return tok_row_ptr_s, token_ids_s, tok_conf_s, tok_neg_ent_s, tok_gini_s, tok_self_s, tok_logp_s
