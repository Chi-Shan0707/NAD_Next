"""
Fast cache builder using two-stage Map-Reduce with spawn context.
v4.1: optional Row-CSR Bank for window queries (position-based)

修复说明：
- Issue 1 (数据结构): 处理 1920 samples (not 883K slices)
- Issue 2 (并行挂起): 使用 spawn context，worker 写临时文件
- Issue 3 (性能): 两阶段 Map-Reduce，目标 <15 分钟

两阶段流程：
  Stage-A (Map): 60 workers 并行，每个输出 partial_{shard_id}.npz
  Stage-B (Reduce): 主进程k路归并 → 最终cache
"""
from __future__ import annotations
import os
import json
import logging
import tempfile
import shutil
from typing import Tuple, List, Dict
from glob import glob
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Setup logger for this module
logger = logging.getLogger(__name__)

from ..core.adapters.shard_reader import read_shard_grouped_by_sample, read_shard_tokens_by_sample
from ..core.storage.cache_paths import CachePaths
from ..core.storage.binary_io import create_memmap, write_array_atomic
from ..core.schema.manifest import Manifest, MANIFEST_VERSION, sha256_file


def _set_worker_env():
    """
    设置worker环境变量，避免BLAS线程风暴。
    在ProcessPoolExecutor的initializer中调用。
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _process_shard(args) -> str:
    """
    Worker：把一个 shard 转成 sample 级 partial-CSR + token metadata，
    以及（可选）行级 partial，并落成临时 npz。

    Args:
        args: (shard_path, tmp_dir, num_samples, row_bank)

    Returns:
        临时 npz 文件路径
    """
    shard_path, tmp_dir, num_samples, row_bank = args

    # 调用新的sample聚合函数 (neuron data)
    row_ptr_s, keys_s, wmax_s, wsum_s = read_shard_grouped_by_sample(shard_path, num_samples)

    # 提取token metadata (按sample聚合)
    (tok_row_ptr_s, token_ids_s, tok_conf_s, tok_neg_ent_s,
     tok_gini_s, tok_self_s, tok_logp_s) = read_shard_tokens_by_sample(shard_path, num_samples)

    # 写临时文件（小而集中，避免把大数组从子进程回传导致挂起）
    base = os.path.basename(shard_path)
    out = os.path.join(tmp_dir, base.replace(".npz", "_partial.npz"))

    # Build data dict
    data = {
        "row_ptr_s": row_ptr_s, "keys_s": keys_s, "wmax_s": wmax_s, "wsum_s": wsum_s,
        "tok_row_ptr_s": tok_row_ptr_s,
        "token_ids_s": (token_ids_s if token_ids_s is not None else np.empty((0,), np.int32)),
        "tok_conf_s": (tok_conf_s if tok_conf_s is not None else np.empty((0,), np.float32)),
        "tok_neg_entropy_s": (tok_neg_ent_s if tok_neg_ent_s is not None else np.empty((0,), np.float32)),
        "tok_gini_s": (tok_gini_s if tok_gini_s is not None else np.empty((0,), np.float32)),
        "tok_selfcert_s": (tok_self_s if tok_self_s is not None else np.empty((0,), np.float32)),
        "tok_logprob_s": (tok_logp_s if tok_logp_s is not None else np.empty((0,), np.float32)),
    }

    # 行级 partial：对每个行（slice）做一次升序去重（max/sum 都输出，减少后续重复工作）
    if row_bank:
        d = np.load(shard_path, mmap_mode="r")
        if all(k in d for k in ("idx_samples", "slice_ids", "layers", "neurons", "scores", "token_row_ptr")):
            idx_samples = d["idx_samples"].astype(np.int64)
            slice_ids   = d["slice_ids"].astype(np.int32)
            layers      = d["layers"]      # (N,K) uint8
            neurons     = d["neurons"]     # (N,K) uint16
            scores      = d["scores"].astype(np.float32)  # (N,K) fp16->fp32
            tok_ptr     = d["token_row_ptr"].astype(np.int64)
            N, K = layers.shape
            # 行级去重：对每行 pack 为 uint32 key = (layer<<16)|neuron，升序去重，输出 wmax/wsum
            row_ptr = np.zeros((N+1,), dtype=np.int64)
            keys_buf: List[np.ndarray] = []
            wmx_buf:  List[np.ndarray] = []
            wsm_buf:  List[np.ndarray] = []
            tok_len = (tok_ptr[1:] - tok_ptr[:-1]).astype(np.int64)
            for r in range(N):
                lay = layers[r].astype(np.uint32, copy=False)
                neu = neurons[r].astype(np.uint32, copy=False)
                sc  = scores[r].astype(np.float32, copy=False)
                # 有效激活（layer>0）——与规范一致；保持 1:1 对齐
                m = (lay > 0)
                if not np.any(m):
                    row_ptr[r+1] = row_ptr[r]
                    continue
                pk = (lay[m] << np.uint32(16)) | neu[m]
                sv = sc[m]
                order = np.argsort(pk, kind="mergesort")
                pk = pk[order]; sv = sv[order]
                # reduceat
                diff = np.empty_like(pk, dtype=bool); diff[0]=True; diff[1:] = pk[1:]!=pk[:-1]
                idx = np.nonzero(diff)[0]
                k_uniq = pk[idx].astype(np.uint32, copy=False)
                w_sum  = np.add.reduceat(sv, idx).astype(np.float32, copy=False)
                w_max  = np.maximum.reduceat(sv, idx).astype(np.float32, copy=False)
                keys_buf.append(k_uniq); wmx_buf.append(w_max); wsm_buf.append(w_sum)
                row_ptr[r+1] = row_ptr[r] + k_uniq.size
            data.update({
                "rows_idx_samples_s": idx_samples,
                "rows_slice_ids_s":   slice_ids,
                "rows_row_ptr_s":     row_ptr,
                "rows_keys_s":        (np.concatenate(keys_buf).astype(np.uint32) if row_ptr[-1]>0 else np.empty((0,),np.uint32)),
                "rows_wmax_s":        (np.concatenate(wmx_buf).astype(np.float32)  if row_ptr[-1]>0 else np.empty((0,),np.float32)),
                "rows_wsum_s":        (np.concatenate(wsm_buf).astype(np.float32)  if row_ptr[-1]>0 else np.empty((0,),np.float32)),
                "rows_token_len_s":   tok_len,
            })
    np.savez(out, **data)
    return out


def _merge_union_count(parts: List[Tuple[np.ndarray, np.ndarray]]) -> int:
    """
    仅计算该 sample 的最终唯一 key 数量。

    Args:
        parts: list of (keys_segment, weights_segment) - 每段keys已升序唯一

    Returns:
        最终唯一key的数量
    """
    if not parts:
        return 0

    # 把所有 keys 段串起来后排序去重
    # 由于每段内部已唯一，concat + sort + uniq 的开销较低
    arrs = [p[0] for p in parts if p[0].size > 0]
    if not arrs:
        return 0

    cat = np.concatenate(arrs).astype(np.uint32, copy=False)
    cat.sort(kind="mergesort")

    if cat.size == 0:
        return 0

    # 计数唯一值
    diff = np.empty_like(cat, dtype=bool)
    diff[0] = True
    diff[1:] = cat[1:] != cat[:-1]

    return int(diff.sum())


def _merge_union_values(parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算最终 (uniq_keys, wmax, wsum)。

    Args:
        parts: list of (keys, wmax, wsum) - 每段keys已升序唯一

    Returns:
        (uniq_keys, wmax, wsum) - 全局唯一并聚合
    """
    if not parts:
        return (np.empty((0,), dtype=np.uint32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    arr_k = [p[0] for p in parts if p[0].size > 0]
    arr_wmax = [p[1] for p in parts if p[1].size > 0]
    arr_wsum = [p[2] for p in parts if p[2].size > 0]

    if not arr_k:
        return (np.empty((0,), dtype=np.uint32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    kcat = np.concatenate(arr_k).astype(np.uint32, copy=False)
    vmax = np.concatenate(arr_wmax).astype(np.float32, copy=False)
    vsum = np.concatenate(arr_wsum).astype(np.float32, copy=False)

    # 按 key 排序
    order = np.argsort(kcat, kind="mergesort")
    k_sorted = kcat[order]
    max_sorted = vmax[order]
    sum_sorted = vsum[order]

    # reduce by key
    diff = np.empty_like(k_sorted, dtype=bool)
    diff[0] = True
    diff[1:] = k_sorted[1:] != k_sorted[:-1]
    idx = np.nonzero(diff)[0]

    out_keys = k_sorted[idx]
    out_wsum = np.add.reduceat(sum_sorted, idx)
    out_wmax = np.maximum.reduceat(max_sorted, idx)

    return out_keys, out_wmax, out_wsum


def build_cache_fast(raw_dir: str, cache_root: str, meta_json: str,
                     dataset_id: str = "v4", model_id: str = "v4",
                     workers: int = 32, include_token: bool = True,
                     row_bank: bool = False, pos_size: int = 32) -> None:
    """
    两阶段 Map-Reduce 构建（v4.1支持可选Row-CSR Bank）：
      - Map: 60 个 shard 并行 → partial_{shard}.npz
      - Reduce: 按 sample 归并 60 份 partial 段，计数→row_ptr，再归并写 base/index
      - Optional: 构建rows/银行用于任意窗口查询

    Args:
        raw_dir: shard 目录 (包含 *.npz)
        cache_root: 输出缓存目录
        meta_json: metadata JSON 文件路径 (包含samples信息)
        dataset_id: 数据集ID
        model_id: 模型ID
        workers: 并行worker数量
        include_token: 是否包含token metadata
        row_bank: 是否构建Row-CSR Bank（可选，用于窗口查询）
        pos_size: Position大小（tokens），默认32
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Building NAD Cache v4.1 (Map-Reduce with Spawn)")
    logger.info(f"{'='*80}")
    if row_bank:
        logger.info(f"Row-CSR Bank: ENABLED (pos_size={pos_size})")
    else:
        logger.info(f"Row-CSR Bank: DISABLED")

    # 读取metadata获取num_samples
    logger.info("[1/6] Loading metadata...")
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 解析样本/题信息
    if "problems" in meta:
        problems = meta["problems"]
        num_samples = sum(len(v) for v in problems.values())
    elif "samples" in meta:
        num_samples = len(meta["samples"])
    else:
        raise RuntimeError("meta.json 缺少 problems/samples")

    logger.info(f"      Total samples (runs): {num_samples}")

    # 枚举 shard 文件
    shard_files = sorted(glob(os.path.join(raw_dir, "*.npz")))
    if not shard_files:
        raise FileNotFoundError(f"No shards found under {raw_dir}")

    logger.info(f"      Shards found: {len(shard_files)}")

    # 创建缓存目录
    paths = CachePaths(cache_root)
    os.makedirs(paths.base_dir, exist_ok=True)
    os.makedirs(paths.index_dir, exist_ok=True)
    os.makedirs(paths.token_dir, exist_ok=True)  # Token metadata directory

    # 临时目录（优先使用 /dev/shm 内存文件系统以提升性能）
    shm_dir = "/dev/shm"
    if os.path.isdir(shm_dir) and os.access(shm_dir, os.W_OK):
        tmp_dir = tempfile.mkdtemp(prefix="nad_v4_partial_", dir=shm_dir)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="nad_v4_partial_")
    logger.info(f"      Temp dir: {tmp_dir}")

    # ---- Map：并行把 shard → partial-CSR
    logger.info(f"\n[2/6] Map stage: Processing {len(shard_files)} shards in parallel (workers={workers})...")

    ctx = mp.get_context("spawn")  # 防止 fork 带来的死锁/挂起
    partial_paths: List[str] = []

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_set_worker_env) as ex:
        futures = [ex.submit(_process_shard, (sf, tmp_dir, num_samples, row_bank)) for sf in shard_files]

        # 简化进度显示（避免tqdm在主进程和子进程交织）
        logger.debug(f"      Submitted {len(futures)} tasks...")
        for i, fu in enumerate(futures):
            result = fu.result()
            partial_paths.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == len(futures):
                logger.debug(f"      Progress: {i+1}/{len(futures)} shards completed")

    logger.info(f"      Map stage completed. Generated {len(partial_paths)} partial files.")

    # 读取所有 partial 的 row_ptr_s，方便快速定位每个 sample 在各 partial 的切片
    logger.info("\n[3/6] Reduce stage - Phase A: Loading partials and counting...")
    parts_row_ptr = []
    parts_keys = []
    parts_wmax = []
    parts_wsum = []

    # Token partials
    tok_parts_row_ptr = []
    tok_parts_token_ids = []
    tok_parts_conf = []
    tok_parts_neg_ent = []
    tok_parts_gini = []
    tok_parts_self = []
    tok_parts_logp = []
    has_token_data = False

    for pp in partial_paths:
        d = np.load(pp)
        parts_row_ptr.append(d["row_ptr_s"].astype(np.int64))
        parts_keys.append(d["keys_s"].astype(np.uint32))
        parts_wmax.append(d["wmax_s"].astype(np.float32))
        parts_wsum.append(d["wsum_s"].astype(np.float32))

        # Load token partials if available
        if "tok_row_ptr_s" in d.files:
            tok_parts_row_ptr.append(d["tok_row_ptr_s"].astype(np.int64))
            tok_parts_token_ids.append(d["token_ids_s"].astype(np.int32))
            tok_parts_conf.append(d["tok_conf_s"].astype(np.float32))
            tok_parts_neg_ent.append(d["tok_neg_entropy_s"].astype(np.float32))
            tok_parts_gini.append(d["tok_gini_s"].astype(np.float32))
            tok_parts_self.append(d["tok_selfcert_s"].astype(np.float32))
            tok_parts_logp.append(d["tok_logprob_s"].astype(np.float32))
            if d["token_ids_s"].size > 0:
                has_token_data = True

    if has_token_data:
        logger.info(f"      Token metadata detected in shards.")

    # ---- Reduce‑A：只做计数，得到每个 sample 的最终长度，以构造 row_ptr
    logger.info("      Counting unique keys per sample...")
    lengths = np.zeros((num_samples,), dtype=np.int64)
    tok_lengths = np.zeros((num_samples,), dtype=np.int64) if has_token_data else None

    for s in range(num_samples):
        if (s + 1) % 500 == 0 or (s + 1) == num_samples:
            logger.debug(f"      Counting: {s+1}/{num_samples} samples")

        segments = []
        for i in range(len(partial_paths)):
            rp = parts_row_ptr[i]
            a, b = int(rp[s]), int(rp[s + 1])
            if a < b:
                segments.append((parts_keys[i][a:b], parts_wsum[i][a:b]))  # wsum 随便拿一个权重用于占位

        lengths[s] = _merge_union_count(segments)

        # Count tokens for this sample
        if tok_lengths is not None:
            n = 0
            for i in range(len(tok_parts_row_ptr)):
                rp = tok_parts_row_ptr[i]
                aa, bb = int(rp[s]), int(rp[s + 1])
                n += max(0, bb - aa)
            tok_lengths[s] = n

    logger.info(f"\n      Counted all samples.")

    # 构建 row_ptr
    row_ptr = np.zeros((num_samples + 1,), dtype=np.int64)
    np.cumsum(lengths, out=row_ptr[1:])
    total_K = int(row_ptr[-1])

    logger.info(f"      Total unique keys after merge: {total_K:,}")

    # Token row_ptr
    if tok_lengths is not None:
        tok_row_ptr = np.zeros((num_samples + 1,), dtype=np.int64)
        np.cumsum(tok_lengths, out=tok_row_ptr[1:])
        total_T = int(tok_row_ptr[-1])
        logger.info(f"      Total tokens: {total_T:,}")

    # ---- Row-CSR Bank 统计与分配 (optional, NEW in v4.1)
    if row_bank:
        logger.info("\n[3b/6] Row-CSR Bank - Statistics phase...")

        # 加载所有 partial 的行级数据
        parts_rows_idx_samples = []
        parts_rows_slice_ids = []
        parts_rows_row_ptr = []
        parts_rows_keys = []
        parts_rows_wmax = []
        parts_rows_wsum = []
        parts_rows_token_len = []

        has_row_data = False
        for p_path in partial_paths:
            d = np.load(p_path, mmap_mode='r')
            if 'rows_idx_samples_s' in d:
                has_row_data = True
                parts_rows_idx_samples.append(d['rows_idx_samples_s'])
                parts_rows_slice_ids.append(d['rows_slice_ids_s'])
                parts_rows_row_ptr.append(d['rows_row_ptr_s'])
                parts_rows_keys.append(d['rows_keys_s'])
                parts_rows_wmax.append(d['rows_wmax_s'])
                parts_rows_wsum.append(d['rows_wsum_s'])
                parts_rows_token_len.append(d['rows_token_len_s'])

        if not has_row_data:
            logger.warning("      No row-level data found in partials, skipping rows/ bank")
            row_bank = False
        else:
            # 统计每个 sample 的总行数
            rows_per_sample = np.zeros((num_samples,), dtype=np.int64)
            for idx_arr in parts_rows_idx_samples:
                for idx in idx_arr:
                    rows_per_sample[idx] += 1

            # 构建 sample_row_ptr: 每个 sample 的行起始位置
            rows_sample_row_ptr = np.zeros((num_samples + 1,), dtype=np.int64)
            np.cumsum(rows_per_sample, out=rows_sample_row_ptr[1:])
            total_rows = int(rows_sample_row_ptr[-1])

            logger.info(f"      Total rows (slices): {total_rows:,}")
            logger.info(f"      Average rows per sample: {total_rows/num_samples:.1f}")

            # 为每一行统计唯一键数量
            rows_row_ptr = np.zeros((total_rows + 1,), dtype=np.int64)
            rows_slice_ids_buf = np.zeros((total_rows,), dtype=np.int32)
            rows_token_len_buf = np.zeros((total_rows,), dtype=np.int64)

            # 填充每个 sample 的行信息
            for i in range(len(parts_rows_idx_samples)):
                idx_arr = parts_rows_idx_samples[i]
                slice_arr = parts_rows_slice_ids[i]
                rp = parts_rows_row_ptr[i]
                tok_len = parts_rows_token_len[i]

                for j, sample_idx in enumerate(idx_arr):
                    # 当前行在全局 rows/ 中的索引
                    # 需要为每个 sample 分配连续的行位置
                    # 我们使用一个计数器来追踪每个 sample 已经分配了多少行
                    pass  # 将在下面的循环中处理

            # 重新组织：按 sample 顺序收集所有行
            sample_row_counts = np.zeros((num_samples,), dtype=np.int64)

            for i in range(len(parts_rows_idx_samples)):
                idx_arr = parts_rows_idx_samples[i]
                slice_arr = parts_rows_slice_ids[i]
                rp = parts_rows_row_ptr[i]
                tok_len = parts_rows_token_len[i]

                for j, sample_idx in enumerate(idx_arr):
                    # 当前 sample 的第几行
                    local_row_idx = sample_row_counts[sample_idx]
                    # 在全局 rows/ 中的行索引
                    global_row_idx = int(rows_sample_row_ptr[sample_idx]) + local_row_idx

                    # 该行的键数量
                    num_keys = int(rp[j+1] - rp[j])
                    rows_row_ptr[global_row_idx + 1] = num_keys
                    rows_slice_ids_buf[global_row_idx] = slice_arr[j]
                    rows_token_len_buf[global_row_idx] = tok_len[j]

                    sample_row_counts[sample_idx] += 1

            # 累加 rows_row_ptr
            np.cumsum(rows_row_ptr[1:], out=rows_row_ptr[1:])
            total_row_keys = int(rows_row_ptr[-1])

            logger.info(f"      Total unique keys in rows/: {total_row_keys:,}")

            # 分配 rows/ memmap
            logger.info("\n[3c/6] Row-CSR Bank - Allocating memory-mapped arrays...")
            os.makedirs(paths.rows_dir, exist_ok=True)

            rows_keys_mm = create_memmap(paths.rows_keys, np.uint32, shape=(total_row_keys,))
            rows_wmax_mm = create_memmap(paths.rows_w_max, np.float16, shape=(total_row_keys,))
            rows_wsum_mm = create_memmap(paths.rows_w_sum, np.float16, shape=(total_row_keys,))

            write_array_atomic(paths.rows_sample_row_ptr, rows_sample_row_ptr)
            write_array_atomic(paths.rows_row_ptr, rows_row_ptr)
            write_array_atomic(paths.rows_slice_ids, rows_slice_ids_buf)

            rows_token_row_ptr = np.empty((total_rows + 1,), dtype=np.int64)
            # Represent rows_token_row_ptr as a **global** cumsum over all rows.
            # This guarantees monotonic non-decreasing pointers and avoids the
            # shared-boundary ambiguity between the last row of sample s and the
            # first row of sample s+1.
            rows_token_row_ptr[0] = 0
            np.cumsum(rows_token_len_buf, out=rows_token_row_ptr[1:])

            # NOTE: Downstream readers must convert to per-sample *local* token
            # coordinates by subtracting the sample base value at row_start.
            write_array_atomic(paths.rows_token_row_ptr, rows_token_row_ptr)

            # 写入行级数据
            logger.info("\n[3d/6] Row-CSR Bank - Writing row-level data...")

            # 重置计数器
            sample_row_counts[:] = 0

            for i in range(len(parts_rows_idx_samples)):
                idx_arr = parts_rows_idx_samples[i]
                rp = parts_rows_row_ptr[i]
                keys_arr = parts_rows_keys[i]
                wmax_arr = parts_rows_wmax[i]
                wsum_arr = parts_rows_wsum[i]

                for j, sample_idx in enumerate(idx_arr):
                    local_row_idx = sample_row_counts[sample_idx]
                    global_row_idx = int(rows_sample_row_ptr[sample_idx]) + local_row_idx

                    # 该行的数据范围
                    start_key = int(rp[j])
                    end_key = int(rp[j + 1])

                    if end_key > start_key:
                        # 写入到 rows/ memmap
                        dst_start = int(rows_row_ptr[global_row_idx])
                        dst_end = int(rows_row_ptr[global_row_idx + 1])

                        rows_keys_mm[dst_start:dst_end] = keys_arr[start_key:end_key]
                        rows_wmax_mm[dst_start:dst_end] = wmax_arr[start_key:end_key].astype(np.float16)
                        rows_wsum_mm[dst_start:dst_end] = wsum_arr[start_key:end_key].astype(np.float16)

                    sample_row_counts[sample_idx] += 1

            logger.info(f"      Row-CSR Bank construction complete!")
            logger.info(f"      - Total rows: {total_rows:,}")
            logger.info(f"      - Total keys: {total_row_keys:,}")
            logger.info(f"      - Avg keys/row: {total_row_keys/total_rows:.1f}")

    # ---- 分配 base & index memmap
    logger.info("\n[4/6] Allocating memory-mapped arrays...")
    keys_mm = create_memmap(paths.keys, np.uint32, shape=(total_K,))
    wmax_mm = create_memmap(paths.w_max, np.float16, shape=(total_K,))
    wsum_mm = create_memmap(paths.w_sum, np.float16, shape=(total_K,))
    perm_max = create_memmap(paths.perm_max, np.int32, shape=(total_K,))
    perm_sum = create_memmap(paths.perm_sum, np.int32, shape=(total_K,))
    prefix_max = create_memmap(paths.prefix_max, np.float16, shape=(total_K,))
    prefix_sum = create_memmap(paths.prefix_sum, np.float16, shape=(total_K,))

    write_array_atomic(paths.row_ptr, row_ptr)

    # Allocate token arrays if data available
    if tok_lengths is not None and total_T > 0:
        logger.info(f"      Allocating token metadata arrays ({total_T:,} tokens)...")
        token_ids_mm = create_memmap(paths.token_ids, np.int32, shape=(total_T,))
        tok_conf_mm = create_memmap(paths.tok_conf, np.float32, shape=(total_T,))
        tok_neg_ent_mm = create_memmap(paths.tok_neg_entropy, np.float32, shape=(total_T,))
        tok_gini_mm = create_memmap(paths.tok_gini, np.float32, shape=(total_T,))
        tok_self_mm = create_memmap(paths.tok_selfcert, np.float32, shape=(total_T,))
        tok_logp_mm = create_memmap(paths.tok_logprob, np.float32, shape=(total_T,))
        write_array_atomic(paths.token_row_ptr, tok_row_ptr)

    # ---- Reduce‑B：真正归并写入 base，并生成 index
    logger.info("\n[5/6] Reduce stage - Phase B: Merging and writing cache...")

    for s in range(num_samples):
        if (s + 1) % 500 == 0 or (s + 1) == num_samples:
            logger.debug(f"      Merging: {s+1}/{num_samples} samples")

        start, end = int(row_ptr[s]), int(row_ptr[s + 1])
        if end == start:
            continue

        segs = []
        for i in range(len(partial_paths)):
            rp = parts_row_ptr[i]
            a, b = int(rp[s]), int(rp[s + 1])
            if a < b:
                segs.append((parts_keys[i][a:b], parts_wmax[i][a:b], parts_wsum[i][a:b]))

        uniq, wmx, wsm = _merge_union_values(segs)
        L = end - start
        assert L == uniq.size, f"length mismatch at sample {s}: expect {L}, got {uniq.size}"

        # 写入 base
        keys_mm[start:end] = uniq
        wmax_mm[start:end] = wmx.astype(np.float16)
        wsum_mm[start:end] = wsm.astype(np.float16)

        # 生成 index: max
        if L > 0:
            order_max = np.argsort(-wmx, kind="mergesort")
            perm_max[start:end] = (start + order_max).astype(np.int32)
            cdf = np.cumsum(wmx[order_max], dtype=np.float32)
            prefix_max[start:end] = (cdf / cdf[-1]).astype(np.float16) if cdf[-1] > 0 else 0

        # 生成 index: sum
        if L > 0:
            order_sum = np.argsort(-wsm, kind="mergesort")
            perm_sum[start:end] = (start + order_sum).astype(np.int32)
            cdf = np.cumsum(wsm[order_sum], dtype=np.float32)
            prefix_sum[start:end] = (cdf / cdf[-1]).astype(np.float16) if cdf[-1] > 0 else 0

        # Merge token data for this sample
        if tok_lengths is not None and tok_lengths[s] > 0:
            ta, tb = int(tok_row_ptr[s]), int(tok_row_ptr[s + 1])
            pos = ta

            # Concatenate token segments from all shards for this sample
            for i in range(len(tok_parts_row_ptr)):
                rp = tok_parts_row_ptr[i]
                aa, bb = int(rp[s]), int(rp[s + 1])
                if aa >= bb:
                    continue

                seg_len = bb - aa
                token_ids_mm[pos:pos + seg_len] = tok_parts_token_ids[i][aa:bb]

                # Only write if arrays exist and have data
                if tok_parts_conf[i].size > 0:
                    tok_conf_mm[pos:pos + seg_len] = tok_parts_conf[i][aa:bb]
                if tok_parts_neg_ent[i].size > 0:
                    tok_neg_ent_mm[pos:pos + seg_len] = tok_parts_neg_ent[i][aa:bb]
                if tok_parts_gini[i].size > 0:
                    tok_gini_mm[pos:pos + seg_len] = tok_parts_gini[i][aa:bb]
                if tok_parts_self[i].size > 0:
                    tok_self_mm[pos:pos + seg_len] = tok_parts_self[i][aa:bb]
                if tok_parts_logp[i].size > 0:
                    tok_logp_mm[pos:pos + seg_len] = tok_parts_logp[i][aa:bb]

                pos += seg_len

    logger.info(f"\n      Merge completed.")

    # ---- Flush & manifest
    logger.info("\n[6/6] Finalizing and creating manifest...")
    del keys_mm, wmax_mm, wsum_mm, perm_max, perm_sum, prefix_max, prefix_sum

    # Delete token memmaps if allocated
    if tok_lengths is not None and total_T > 0:
        del token_ids_mm, tok_conf_mm, tok_neg_ent_mm, tok_gini_mm, tok_self_mm, tok_logp_mm

    files = {
        "base/keys.uint32": paths.keys,
        "base/w_max.float16": paths.w_max,
        "base/w_sum.float16": paths.w_sum,
        "base/row_ptr.int64": paths.row_ptr,
        "index/perm_max.int32": paths.perm_max,
        "index/prefix_max.float16": paths.prefix_max,
        "index/perm_sum.int32": paths.perm_sum,
        "index/prefix_sum.float16": paths.prefix_sum,
    }

    # Add token files to manifest if available
    if tok_lengths is not None and total_T > 0:
        files.update({
            "token_data/token_row_ptr.int64": paths.token_row_ptr,
            "token_data/token_ids.int32": paths.token_ids,
            "token_data/tok_conf.float32": paths.tok_conf,
            "token_data/tok_neg_entropy.float32": paths.tok_neg_entropy,
            "token_data/tok_gini.float32": paths.tok_gini,
            "token_data/tok_selfcert.float32": paths.tok_selfcert,
            "token_data/tok_logprob.float32": paths.tok_logprob,
        })

    files_sha = {rel: sha256_file(path) for rel, path in files.items()}

    # Prepare token metadata if available
    token_metadata_config = None
    token_row_ptr_sum_val = None
    if tok_lengths is not None and total_T > 0:
        token_metadata_config = {
            "enabled": True,
            "fields": ["token_ids", "tok_conf", "tok_neg_entropy", "tok_gini", "tok_selfcert", "tok_logprob"],
            "dtypes": {
                "token_ids": "int32",
                "tok_conf": "float32",
                "tok_neg_entropy": "float32",
                "tok_gini": "float32",
                "tok_selfcert": "float32",
                "tok_logprob": "float32"
            },
            "description": "Token-level metadata: IDs and uncertainty metrics"
        }
        token_row_ptr_sum_val = total_T

    manifest = Manifest(
        version=MANIFEST_VERSION,
        dataset_id=dataset_id,
        model_id=model_id,
        num_runs=num_samples,
        aggregations=["max", "sum"],
        dtypes={
            "keys": "uint32",
            "w_max": "float16",
            "w_sum": "float16",
            "perm_*": "int32",
            "prefix_*": "float16"
        },
        row_ptr_sum=int(row_ptr[-1]),
        token_row_ptr_sum=token_row_ptr_sum_val,
        token_metadata=token_metadata_config,
        global_key_dict_hash="v4",  # 可替换为真实 hash
        files_sha256=files_sha,
        run_metadata={
            "position_size": int(pos_size),   # 统一口径：1 position = pos_size tokens（默认 32）
            "has_row_bank": bool(row_bank)    # 是否构建了Row-CSR Bank
        }
    )

    manifest.save(paths.manifest)

    # 清理临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Cache build completed successfully!")
    logger.info(f"{'='*80}")
    logger.info(f"Samples (runs): {num_samples:,}")
    logger.info(f"Total unique keys: {total_K:,}")
    logger.info(f"Cache location: {cache_root}")
    logger.info(f"Manifest: {paths.manifest}")
    logger.info(f"{'='*80}\n")
