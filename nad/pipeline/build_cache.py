
from __future__ import annotations
import os, json, math, time, logging
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import numpy as np

# Setup logger for this module
logger = logging.getLogger(__name__)

from ..core.storage.cache_paths import CachePaths
from ..core.storage.binary_io import create_memmap, write_array_atomic
from ..core.schema.manifest import Manifest, MANIFEST_VERSION, sha256_file
# NOTE: Avoid importing optional adapters at module import time to prevent
# ImportError when only basic cache-build is used. Advanced builders can
# perform their own local imports.

# NAD v4.0: Build cache from NPZ shards with extended schema (token metadata)

def _dedup_agg(keys: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Ensure uint32 keys
    keys = keys.astype(np.uint64, copy=False)
    scores = scores.astype(np.float32, copy=False)
    # sort by key
    order = np.argsort(keys, kind="mergesort")
    sk = keys[order]
    sw = scores[order]
    if sk.size == 0:
        return sk.astype(np.uint32), sw.astype(np.float32), sw.astype(np.float32)

    # boundaries
    diff = np.empty_like(sk, dtype=bool)
    diff[0] = True
    diff[1:] = sk[1:] != sk[:-1]
    idx = np.nonzero(diff)[0]
    # run lengths
    lens = np.diff(np.append(idx, sk.size))
    # reduceat for sum & max
    sumv = np.add.reduceat(sw, idx)
    maxv = np.maximum.reduceat(sw, idx)
    uniq_keys = sk[idx].astype(np.uint32, copy=False)
    return uniq_keys, maxv.astype(np.float32, copy=False), sumv.astype(np.float32, copy=False)

def build_cache(raw_dir: str, cache_root: str,
                dataset_id: str = "demo", model_id: str = "demo",
                use_sum: bool = True) -> None:
    paths = CachePaths(cache_root)
    os.makedirs(paths.base_dir, exist_ok=True)
    os.makedirs(paths.index_dir, exist_ok=True)

    # Discover runs
    # Build a flat list of (problem_id, run_path)
    run_paths: List[str] = []
    problem_ids: List[str] = []
    # problems.json expected at raw_dir
    problems_meta_path = os.path.join(raw_dir, "problems.json")
    if not os.path.exists(problems_meta_path):
        raise FileNotFoundError(f"Missing {problems_meta_path}")
    with open(problems_meta_path, "r", encoding="utf-8") as f:
        problems_meta = json.load(f)
    problems = problems_meta["problems"]
    # Map run_id -> (problem_id, run_idx)
    run_id_map: Dict[int, Tuple[str,int]] = {}
    rid = 0
    for prob, run_list in problems.items():
        for run_idx in run_list:
            run_path = os.path.join(raw_dir, prob, f"run_{run_idx}.npz")
            run_paths.append(run_path)
            problem_ids.append(prob)
            run_id_map[rid] = (prob, run_idx)
            rid += 1
    num_runs = len(run_paths)

    # ---- PASS 1: lengths ----
    lengths = np.zeros((num_runs,), dtype=np.int64)
    key_space_hash = 0
    for i, p in enumerate(run_paths):
        data = np.load(p)
        k = data["keys"]
        s = data["scores"]
        uniq_keys, wmax, wsum = _dedup_agg(k, s)
        lengths[i] = uniq_keys.size
        key_space_hash ^= int(uniq_keys.size + (uniq_keys[:min(3, uniq_keys.size)].sum() if uniq_keys.size>0 else 0))

    row_ptr = np.zeros(num_runs + 1, dtype=np.int64)
    np.cumsum(lengths, out=row_ptr[1:])
    total_K = int(row_ptr[-1])

    # ---- Allocate base and index ----
    keys_mm = create_memmap(paths.keys, np.uint32, shape=(total_K,))
    w_max_mm = create_memmap(paths.w_max, np.float16, shape=(total_K,))
    w_sum_mm = create_memmap(paths.w_sum, np.float16, shape=(total_K,)) if use_sum else None
    perm_max_mm = create_memmap(paths.perm_max, np.int32, shape=(total_K,))
    perm_sum_mm = create_memmap(paths.perm_sum, np.int32, shape=(total_K,)) if use_sum else None
    prefix_max_mm = create_memmap(paths.prefix_max, np.float16, shape=(total_K,))
    prefix_sum_mm = create_memmap(paths.prefix_sum, np.float16, shape=(total_K,)) if use_sum else None

    # Write row_ptr atomically
    write_array_atomic(paths.row_ptr, row_ptr)

    # ---- PASS 2: fill arrays ----
    for i, p in enumerate(run_paths):
        start, end = int(row_ptr[i]), int(row_ptr[i+1])
        data = np.load(p)
        uniq_keys, wmax, wsum = _dedup_agg(data["keys"], data["scores"])
        L = end - start
        assert L == uniq_keys.size, "length mismatch between pass1 and pass2"

        # base
        keys_mm[start:end] = uniq_keys
        w_max_mm[start:end] = wmax.astype(np.float16)
        if use_sum:
            w_sum_mm[start:end] = wsum.astype(np.float16)

        # index for max
        if L > 0:
            order_max = np.argsort(-wmax, kind="mergesort")  # descending by weight
            perm_max_mm[start:end] = (start + order_max).astype(np.int32)
            cdf = np.cumsum(wmax[order_max], dtype=np.float32)
            if cdf[-1] > 0:
                prefix_max_mm[start:end] = (cdf / cdf[-1]).astype(np.float16)
            else:
                prefix_max_mm[start:end] = 0
        # index for sum
        if use_sum and L > 0:
            order_sum = np.argsort(-wsum, kind="mergesort")
            perm_sum_mm[start:end] = (start + order_sum).astype(np.int32)
            cdf = np.cumsum(wsum[order_sum], dtype=np.float32)
            if cdf[-1] > 0:
                prefix_sum_mm[start:end] = (cdf / cdf[-1]).astype(np.float16)
            else:
                prefix_sum_mm[start:end] = 0

    # Flush memmaps
    del keys_mm, w_max_mm, w_sum_mm, perm_max_mm, perm_sum_mm, prefix_max_mm, prefix_sum_mm

    # Manifest
    files = {
        "base/keys.uint32": paths.keys,
        "base/w_max.float16": paths.w_max,
        "base/w_sum.float16": paths.w_sum if use_sum else None,
        "base/row_ptr.int64": paths.row_ptr,
        "index/perm_max.int32": paths.perm_max,
        "index/prefix_max.float16": paths.prefix_max,
        "index/perm_sum.int32": paths.perm_sum if use_sum else None,
        "index/prefix_sum.float16": paths.prefix_sum if use_sum else None,
    }
    files_sha = { rel: sha256_file(path) for rel, path in files.items() if path is not None }

    manifest = Manifest(
        version=MANIFEST_VERSION,
        dataset_id=dataset_id,
        model_id=model_id,
        num_runs=num_runs,
        aggregations=["max", "sum"] if use_sum else ["max"],
        dtypes={
            "keys": "uint32",
            "w_max": "float16",
            "w_sum": "float16" if use_sum else "none",
            "perm_*": "int32",
            "prefix_*": "float16",
        },
        row_ptr_sum=int(row_ptr[-1]),
        global_key_dict_hash=hex(key_space_hash),
        files_sha256=files_sha,
    )
    manifest.save(paths.manifest)
    logger.info(f"[build_cache] Done. Runs={num_runs}, total_K={total_K}. Cache at: {cache_root}")


def build_cache_v4(shard_dir: str, run_index_dir: str, cache_root: str,
                   dataset_id: str = "aime24", model_id: str = "deepseek-r1",
                   include_token_metadata: bool = True,
                   use_sum: bool = True) -> None:
    """
    Build cache v4.0 from NPZ shards with extended schema.

    Args:
        shard_dir: Directory containing 60 NPZ shard files
        run_index_dir: Directory containing run_index.json
        cache_root: Output cache directory
        dataset_id: Dataset identifier
        model_id: Model identifier
        include_token_metadata: Whether to cache token-level metadata
        use_sum: Whether to compute sum aggregation (in addition to max)
    """
    # Deferred import to avoid hard dependency at module import time
    try:
        from ..core.adapters.shard_reader import ShardAdapter  # type: ignore
        from tqdm import tqdm  # imported here to avoid hard dependency for basic build
    except Exception as e:
        raise ImportError(
            "build_cache_v4 depends on a ShardAdapter that is not available in this build. "
            "Please use 'build_cache_fast' (cache-build-fast) which supersedes this path."
        ) from e

    start_time = time.time()
    logger.info(f"\n{'='*80}")
    logger.info(f"Building NAD Cache v4.0")
    logger.info(f"{'='*80}")
    logger.info(f"Shard directory: {shard_dir}")
    logger.info(f"Run index: {run_index_dir}")
    logger.info(f"Cache output: {cache_root}")
    logger.info(f"Token metadata: {'ENABLED' if include_token_metadata else 'DISABLED'}")
    logger.info(f"{'='*80}\n")

    # Initialize paths
    paths = CachePaths(cache_root)
    os.makedirs(paths.base_dir, exist_ok=True)
    os.makedirs(paths.index_dir, exist_ok=True)
    if include_token_metadata:
        os.makedirs(paths.token_dir, exist_ok=True)
    os.makedirs(paths.metadata_dir, exist_ok=True)

    # Initialize ShardAdapter
    logger.info("[1/4] Initializing ShardAdapter...")
    adapter = ShardAdapter(shard_dir, run_index_dir, include_token_metadata=include_token_metadata)
    num_runs = adapter.num_runs
    logger.info(f"      Total runs to process: {num_runs:,}")

    # ---- PASS 1: Calculate lengths ----
    logger.info("\n[2/4] Pass 1: Calculating array sizes...")
    neuron_lengths = np.zeros((num_runs,), dtype=np.int64)
    token_lengths = np.zeros((num_runs,), dtype=np.int32) if include_token_metadata else None

    # Metadata arrays
    sample_ids = np.zeros((num_runs,), dtype=np.int32)
    slice_ids = np.zeros((num_runs,), dtype=np.int32)
    problem_ids = np.zeros((num_runs,), dtype=np.int16)
    num_tokens_arr = np.zeros((num_runs,), dtype=np.int32)

    key_space_hash = 0

    for run_data in tqdm(adapter.iter_runs(progress=False), total=num_runs, desc="      Scanning runs"):
        run_id = run_data.run_id

        # Dedup and aggregate neuron data
        if run_data.neuron_keys.size > 0:
            uniq_keys, wmax, wsum = _dedup_agg(run_data.neuron_keys, run_data.neuron_scores)
            neuron_lengths[run_id] = uniq_keys.size
            key_space_hash ^= int(uniq_keys.size + (uniq_keys[:min(3, uniq_keys.size)].sum() if uniq_keys.size > 0 else 0))
        else:
            neuron_lengths[run_id] = 0

        # Token metadata lengths
        if include_token_metadata and run_data.token_data:
            token_lengths[run_id] = run_data.num_tokens
        elif include_token_metadata:
            token_lengths[run_id] = 0

        # Store run metadata
        sample_ids[run_id] = run_data.sample_id
        slice_ids[run_id] = run_data.slice_id
        problem_ids[run_id] = run_data.problem_id
        num_tokens_arr[run_id] = run_data.num_tokens

    # Calculate CSR pointers
    neuron_row_ptr = np.zeros(num_runs + 1, dtype=np.int64)
    np.cumsum(neuron_lengths, out=neuron_row_ptr[1:])
    total_neuron_keys = int(neuron_row_ptr[-1])

    if include_token_metadata:
        token_row_ptr = np.zeros(num_runs + 1, dtype=np.int64)
        np.cumsum(token_lengths, out=token_row_ptr[1:])
        total_tokens = int(token_row_ptr[-1])
    else:
        token_row_ptr = None
        total_tokens = 0

    logger.info(f"      Total neuron keys (after dedup): {total_neuron_keys:,}")
    if include_token_metadata:
        logger.info(f"      Total tokens: {total_tokens:,}")
    logger.info(f"      Estimated cache size: ~{(total_neuron_keys * 8 + total_tokens * 20) / 1e9:.2f} GB")

    # ---- PASS 2: Allocate and fill arrays ----
    logger.info(f"\n[3/4] Pass 2: Allocating memory-mapped arrays...")

    # Neuron data (base)
    keys_mm = create_memmap(paths.keys, np.uint32, shape=(total_neuron_keys,))
    w_max_mm = create_memmap(paths.w_max, np.float16, shape=(total_neuron_keys,))
    w_sum_mm = create_memmap(paths.w_sum, np.float16, shape=(total_neuron_keys,)) if use_sum else None

    # Index
    perm_max_mm = create_memmap(paths.perm_max, np.int32, shape=(total_neuron_keys,))
    perm_sum_mm = create_memmap(paths.perm_sum, np.int32, shape=(total_neuron_keys,)) if use_sum else None
    prefix_max_mm = create_memmap(paths.prefix_max, np.float16, shape=(total_neuron_keys,))
    prefix_sum_mm = create_memmap(paths.prefix_sum, np.float16, shape=(total_neuron_keys,)) if use_sum else None

    # Token metadata
    if include_token_metadata and total_tokens > 0:
        tok_logprob_mm = create_memmap(paths.tok_logprob, np.float32, shape=(total_tokens,))
        tok_conf_mm = create_memmap(paths.tok_conf, np.float32, shape=(total_tokens,))
        tok_entropy_mm = create_memmap(paths.tok_entropy, np.float32, shape=(total_tokens,))
        tok_gini_mm = create_memmap(paths.tok_gini, np.float32, shape=(total_tokens,))
        tok_selfcert_mm = create_memmap(paths.tok_selfcert, np.float32, shape=(total_tokens,))
    else:
        tok_logprob_mm = tok_conf_mm = tok_entropy_mm = tok_gini_mm = tok_selfcert_mm = None

    # Write row pointers and metadata atomically
    write_array_atomic(paths.row_ptr, neuron_row_ptr)
    if include_token_metadata:
        write_array_atomic(paths.token_row_ptr, token_row_ptr)

    write_array_atomic(paths.sample_ids, sample_ids)
    write_array_atomic(paths.slice_ids, slice_ids)
    write_array_atomic(paths.problem_ids, problem_ids)
    write_array_atomic(paths.num_tokens, num_tokens_arr)

    logger.info(f"      Allocated {total_neuron_keys:,} neuron slots")
    if include_token_metadata:
        logger.info(f"      Allocated {total_tokens:,} token slots")

    # Fill arrays
    logger.info(f"      Filling arrays...")
    for run_data in tqdm(adapter.iter_runs(progress=False), total=num_runs, desc="      Processing runs"):
        run_id = run_data.run_id

        # Neuron data
        n_start, n_end = int(neuron_row_ptr[run_id]), int(neuron_row_ptr[run_id + 1])
        L = n_end - n_start

        if L > 0:
            # Dedup and aggregate
            uniq_keys, wmax, wsum = _dedup_agg(run_data.neuron_keys, run_data.neuron_scores)
            assert L == uniq_keys.size, f"Length mismatch for run {run_id}: expected {L}, got {uniq_keys.size}"

            # Fill base arrays
            keys_mm[n_start:n_end] = uniq_keys
            w_max_mm[n_start:n_end] = wmax.astype(np.float16)
            if use_sum:
                w_sum_mm[n_start:n_end] = wsum.astype(np.float16)

            # Fill index for max
            order_max = np.argsort(-wmax, kind="mergesort")  # descending
            perm_max_mm[n_start:n_end] = (n_start + order_max).astype(np.int32)
            cdf = np.cumsum(wmax[order_max], dtype=np.float32)
            if cdf[-1] > 0:
                prefix_max_mm[n_start:n_end] = (cdf / cdf[-1]).astype(np.float16)
            else:
                prefix_max_mm[n_start:n_end] = 0

            # Fill index for sum
            if use_sum:
                order_sum = np.argsort(-wsum, kind="mergesort")
                perm_sum_mm[n_start:n_end] = (n_start + order_sum).astype(np.int32)
                cdf = np.cumsum(wsum[order_sum], dtype=np.float32)
                if cdf[-1] > 0:
                    prefix_sum_mm[n_start:n_end] = (cdf / cdf[-1]).astype(np.float16)
                else:
                    prefix_sum_mm[n_start:n_end] = 0

        # Token metadata
        if include_token_metadata and run_data.token_data and run_data.num_tokens > 0:
            t_start, t_end = int(token_row_ptr[run_id]), int(token_row_ptr[run_id + 1])

            if 'logprob' in run_data.token_data:
                tok_logprob_mm[t_start:t_end] = run_data.token_data['logprob']
            if 'conf' in run_data.token_data:
                tok_conf_mm[t_start:t_end] = run_data.token_data['conf']
            if 'entropy' in run_data.token_data:
                tok_entropy_mm[t_start:t_end] = run_data.token_data['entropy']
            if 'gini' in run_data.token_data:
                tok_gini_mm[t_start:t_end] = run_data.token_data['gini']
            if 'selfcert' in run_data.token_data:
                tok_selfcert_mm[t_start:t_end] = run_data.token_data['selfcert']

    # Flush memmaps
    logger.info("      Flushing memory-mapped files...")
    del keys_mm, w_max_mm, w_sum_mm, perm_max_mm, perm_sum_mm, prefix_max_mm, prefix_sum_mm
    if include_token_metadata:
        del tok_logprob_mm, tok_conf_mm, tok_entropy_mm, tok_gini_mm, tok_selfcert_mm

    # ---- Create Manifest ----
    logger.info("\n[4/4] Creating manifest and computing checksums...")

    # Collect all files
    files = {
        "base/row_ptr.int64": paths.row_ptr,
        "base/keys.uint32": paths.keys,
        "base/w_max.float16": paths.w_max,
        "index/perm_max.int32": paths.perm_max,
        "index/prefix_max.float16": paths.prefix_max,
        "run_metadata/sample_ids.int32": paths.sample_ids,
        "run_metadata/slice_ids.int32": paths.slice_ids,
        "run_metadata/problem_ids.int16": paths.problem_ids,
        "run_metadata/num_tokens.int32": paths.num_tokens,
    }

    if use_sum:
        files["base/w_sum.float16"] = paths.w_sum
        files["index/perm_sum.int32"] = paths.perm_sum
        files["index/prefix_sum.float16"] = paths.prefix_sum

    if include_token_metadata:
        files["token_data/token_row_ptr.int64"] = paths.token_row_ptr
        files["token_data/tok_logprob.float32"] = paths.tok_logprob
        files["token_data/tok_conf.float32"] = paths.tok_conf
        files["token_data/tok_entropy.float32"] = paths.tok_entropy
        files["token_data/tok_gini.float32"] = paths.tok_gini
        files["token_data/tok_selfcert.float32"] = paths.tok_selfcert

    # Compute SHA256 checksums
    files_sha = {}
    for rel_path, abs_path in tqdm(files.items(), desc="      Computing SHA256"):
        if abs_path and os.path.exists(abs_path):
            files_sha[rel_path] = sha256_file(abs_path)

    # Create manifest
    manifest = Manifest(
        version=MANIFEST_VERSION,
        dataset_id=dataset_id,
        model_id=model_id,
        num_runs=num_runs,
        aggregations=["max", "sum"] if use_sum else ["max"],
        dtypes={
            "keys": "uint32",
            "w_max": "float16",
            "w_sum": "float16" if use_sum else "none",
            "perm_*": "int32",
            "prefix_*": "float16",
            "tok_*": "float32" if include_token_metadata else "none",
        },
        row_ptr_sum=total_neuron_keys,
        token_row_ptr_sum=total_tokens if include_token_metadata else None,
        global_key_dict_hash=hex(key_space_hash),
        files_sha256=files_sha,
        created_at=datetime.now().isoformat(),
        endianness="little"
    )

    manifest.save(paths.manifest)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"Cache build completed successfully!")
    logger.info(f"{'='*80}")
    logger.info(f"Runs processed: {num_runs:,}")
    logger.info(f"Neuron keys (after dedup): {total_neuron_keys:,}")
    if include_token_metadata:
        logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Cache location: {cache_root}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"{'='*80}\n")
