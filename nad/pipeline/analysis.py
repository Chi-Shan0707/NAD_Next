
from __future__ import annotations
import os, json, logging, math
from typing import Dict, List, Sequence, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from tqdm import tqdm

from ..core.views.reader import CacheReader, ViewSpec, CutSpec, Agg, CutType, Order, RunView
from ..core.distance.engine import DistanceEngine, DistanceSpec
from ..core.selectors.base import SelectorSpec, SelectorContext
from ..core.selectors.registry import build_selector, expand_selector_all
from .window_cache import ensure_window_cache
from .profiler import PerformanceMonitor
from ..core.storage.binary_io import mmap_from_file

# Setup logger for this module
logger = logging.getLogger(__name__)


def _get_max_tokens(cache_root: str) -> int:
    """
    Get the maximum token count across all samples from rows/ bank.

    Args:
        cache_root: Path to the cache directory

    Returns:
        Maximum token count across all samples

    Raises:
        FileNotFoundError: If rows/ bank is not available
    """
    rows_dir = Path(cache_root) / "rows"
    trp_path = rows_dir / "token_row_ptr.int64"
    srp_path = rows_dir / "sample_row_ptr.int64"

    if not trp_path.exists() or not srp_path.exists():
        raise FileNotFoundError(
            f"rows/ bank not found at {rows_dir}. "
            f"Please rebuild cache with --row-bank flag to enable pos_window='all' mode."
        )

    trp = mmap_from_file(str(trp_path), np.int64)
    srp = mmap_from_file(str(srp_path), np.int64)

    num_samples = len(srp) - 1
    max_tokens = 0

    for i in range(num_samples):
        row_start = int(srp[i])
        row_end = int(srp[i + 1])
        if row_end > row_start:
            base = int(trp[row_start])
            total = int(trp[row_end]) - base
            max_tokens = max(max_tokens, total)

    return max_tokens


def _map_indices_to_run_ids(indices, run_ids: List[int], selector_name: str, problem_id: str):
    """
    Map selector-returned indices (0 to n-1) to actual global run_ids.

    Args:
        indices: int or list[int] - group-internal indices from selector
        run_ids: List[int] - the global run_ids for this problem group
        selector_name: str - name of the selector (for error messages)
        problem_id: str - problem ID (for error messages)

    Returns:
        int or list[int] - the mapped global run_id(s)
    """
    n_runs = len(run_ids)

    # Handle single int case
    if isinstance(indices, (int, np.integer)):
        idx = int(indices)
        assert 0 <= idx < n_runs, \
            f"Selector '{selector_name}' for problem {problem_id} returned invalid index {idx} (valid range: 0 to {n_runs-1})"
        chosen_run_id = int(run_ids[idx])
        # Extra validation: ensure mapped run_id is in the candidate set
        assert chosen_run_id in run_ids, \
            f"Mapped run_id {chosen_run_id} not in candidates for problem {problem_id} (internal error)"
        return chosen_run_id

    # Handle list/array case
    result = []
    for idx in indices:
        idx = int(idx)
        assert 0 <= idx < n_runs, \
            f"Selector '{selector_name}' for problem {problem_id} returned invalid index {idx} (valid range: 0 to {n_runs-1})"
        chosen_run_id = int(run_ids[idx])
        assert chosen_run_id in run_ids, \
            f"Mapped run_id {chosen_run_id} not in candidates for problem {problem_id} (internal error)"
        result.append(chosen_run_id)
    return result


def _analyze_all_positions(
    cache_root: str,
    agg: str,
    cut: str,
    distance: str,
    normalize: bool,
    selectors: Sequence[Dict] | str | None,
    group_topk_policy: str,
    emit_index: bool,
    pos_size: int,
    out_json: str,
    enable_profiling: bool,
    distance_threads: int,
    assume_unique_keys: bool,
    pos_max: Optional[int] = None  # Optional: limit max position for testing
) -> Dict:
    """
    Analyze all position windows from 0-1 to 0-max_position.

    This function automatically detects the maximum token count across all samples
    and runs analysis for every position window: 0-1, 0-2, 0-3, ..., 0-max_position.

    Args:
        cache_root: Path to the cache directory
        (other args same as analyze())

    Returns:
        Dict with structure:
        {
            "mode": "all_positions",
            "pos_size": 32,
            "max_tokens": 1025,
            "max_position": 33,
            "windows": {
                "0-1": { "problems": {...}, "config": {...} },
                "0-2": { ... },
                ...
            }
        }
    """
    # Get maximum token count
    max_tokens = _get_max_tokens(cache_root)
    max_position = math.ceil(max_tokens / pos_size)

    # Apply pos_max limit if specified
    if pos_max is not None and pos_max < max_position:
        logger.info(f"Limiting max_position from {max_position} to {pos_max} (--pos-max)")
        max_position = pos_max

    logger.info(f"All positions mode: max_tokens={max_tokens}, max_position={max_position}, pos_size={pos_size}")
    logger.info(f"Will analyze {max_position} windows: 0-1, 0-2, ..., 0-{max_position}")

    # Prepare result structure
    results = {
        "mode": "all_positions",
        "pos_size": pos_size,
        "max_tokens": max_tokens,
        "max_position": max_position,
        "config": {
            "agg": agg,
            "cut": cut,
            "distance": distance,
            "normalize": normalize,
            "group_topk_policy": group_topk_policy,
            "distance_threads": distance_threads
        },
        "windows": {}
    }

    # Analyze each window
    for pos_hi in tqdm(range(1, max_position + 1), desc="Analyzing position windows", unit="window"):
        window_name = f"0-{pos_hi}"

        # Call analyze for this specific window (without out_json to avoid intermediate files)
        window_result = analyze(
            cache_root=cache_root,
            agg=agg,
            cut=cut,
            distance=distance,
            normalize=normalize,
            selectors=selectors,
            group_topk_policy=group_topk_policy,
            emit_index=emit_index,
            pos_window=window_name,
            pos_size=pos_size,
            out_json=None,  # Don't write intermediate files
            enable_profiling=False,  # Disable per-window profiling
            distance_threads=distance_threads,
            assume_unique_keys=assume_unique_keys
        )

        # Store result for this window
        results["windows"][window_name] = window_result

        logger.debug(f"Completed window {window_name}")

    # Save aggregated results
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved all-positions results to {out_json}")

    return results


def analyze(cache_root: str,
            agg: str = "max",
            cut: str = "mass:0.98",
            distance: str = "wj",
            normalize: bool = True,
            selectors: Sequence[Dict] | str | None = None,
            group_topk_policy: str = "none",   # "none" | "min" | "max" | "legacy-min" | "fixed:<K>"
            emit_index: bool = False,          # Output both index and run_id for debugging
            pos_window: Optional[str] = None,  # "lo-hi" in position units, or "all" for full sweep (NEW v4.1)
            pos_size: int = 32,                 # tokens per position (NEW v4.1)
            pos_max: Optional[int] = None,     # Optional: limit max position for "all" mode
            out_json: str = None,
            enable_profiling: bool = False,    # Enable performance profiling (default: disabled)
            distance_threads: int = 16,        # NEW: Number of threads for distance computation
            assume_unique_keys: bool = True):  # NEW: Assume keys are unique (faster intersect1d)

    # Handle pos_window="all" mode: sweep through all position windows
    if pos_window is not None and pos_window.lower() == "all":
        return _analyze_all_positions(
            cache_root=cache_root,
            agg=agg,
            cut=cut,
            distance=distance,
            normalize=normalize,
            selectors=selectors,
            group_topk_policy=group_topk_policy,
            emit_index=emit_index,
            pos_size=pos_size,
            out_json=out_json,
            enable_profiling=enable_profiling,
            distance_threads=distance_threads,
            assume_unique_keys=assume_unique_keys,
            pos_max=pos_max
        )

    # Initialize performance monitor
    profiler = PerformanceMonitor(enabled=enable_profiling)

    with profiler.stage("initialization"):
        # Load problems from cache/meta.json (auto-generate from samples)
        meta_json_path = Path(cache_root) / "meta.json"

        if not meta_json_path.exists():
            raise FileNotFoundError(
                f"❌ 错误: 在 cache 目录中未找到 meta.json\n"
                f"  路径: {meta_json_path}\n"
                f"  提示: 请确保 cache 是由 build_cache_with_ground_truth.sh 构建的"
            )

        logger.info(f"从 meta.json 自动生成 problems: {meta_json_path}")

        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # 从 samples 数组中提取 problems
        if "samples" not in meta_data:
            raise ValueError(
                f"❌ 错误: meta.json 缺少 'samples' 字段\n"
                f"  路径: {meta_json_path}\n"
                f"  提示: meta.json 格式可能不正确"
            )

        # 按 problem_id 分组，收集 sample_id (run_ids)
        groups: Dict[str, List[int]] = {}
        for sample_id, sample in enumerate(meta_data['samples']):
            pid = str(sample['problem_id'])
            if pid not in groups:
                groups[pid] = []
            groups[pid].append(sample_id)

        logger.info(f"从 meta.json 提取了 {len(groups)} 个问题，"
                    f"总计 {sum(len(runs) for runs in groups.values())} 个样本")

        # Normalize group_topk_policy for compatibility (min ≡ legacy-min)
        if group_topk_policy == "min":
            group_topk_policy = "legacy-min"
            logger.info("Normalized group_topk_policy 'min' -> 'legacy-min'")

        # Parse pos_window (NEW v4.1)
        pos_lo, pos_hi = None, None
        if pos_window:
            if "-" not in pos_window:
                raise ValueError(f"Invalid pos_window format: '{pos_window}'. Expected 'lo-hi' (e.g., '0-32')")
            parts = pos_window.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid pos_window format: '{pos_window}'. Expected 'lo-hi' (e.g., '0-32')")
            pos_lo = int(parts[0])
            pos_hi = int(parts[1])
            if pos_lo < 0 or pos_hi <= pos_lo:
                raise ValueError(f"Invalid pos_window range: '{pos_window}'. Must have 0 <= lo < hi")
            logger.info(f"Position window query: [{pos_lo}, {pos_hi}) (pos_size={pos_size})")

            # Ensure window cache exists (if using rows/ bank)
            if ensure_window_cache(cache_root, pos_lo, pos_hi, pos_size):
                logger.info(f"Window cache ready for pos [{pos_lo}, {pos_hi})")
            else:
                logger.warning("rows/ bank not available, falling back to full-query")

        reader = CacheReader(cache_root)

        # Parse cut
        if ":" in cut:
            ctype, val = cut.split(":")
            if ctype == "topk":
                cs = CutSpec(CutType.TOPK, float(int(val)))
            else:
                cs = CutSpec(CutType.MASS, float(val))
        else:
            cs = CutSpec(CutType.MASS, 0.98)

        base_agg = Agg(agg.lower())
        vspec_default = ViewSpec(agg=base_agg, cut=cs, order=Order.BY_KEY)
        dspec = DistanceSpec(name=distance.lower(),
                            normalize=normalize,
                            num_threads=distance_threads,
                            assume_unique=assume_unique_keys)

        # expand selectors (string "all" supported; also support "all,<custom>" mixed usage)
        if selectors is None:
            selectors = [{"name": "min-activation"},
                         {"name": "medoid"},
                         {"name": "knn-medoid", "params": {"k": 5}}]
        elif isinstance(selectors, str):
            # Support "all" token combined with custom names, e.g., "all,py:module:Class"
            tokens = [x.strip() for x in selectors.split(",") if x.strip()]
            if any(t.lower() == "all" for t in tokens):
                builtins = expand_selector_all(cache_root=cache_root)
                extras = [{"name": t} for t in tokens if t.lower() != "all"]
                selectors = builtins + extras
            else:
                selectors = [{"name": t} for t in tokens]

    results = {"problems": {}}

    with profiler.stage("problem_processing"):
        for pid, run_ids in tqdm(groups.items(), desc="Processing problems", unit="problem"):
            # IMPORTANT: run_ids is the list of global run IDs for this problem group
            # Selectors will work with indices 0 to len(run_ids)-1
            run_ids = list(run_ids)  # Ensure it's a list for indexing

            # Decide per-problem unified Top-K policy (for legacy compatibility)
            # raw lengths (Kmax) from CSR
            kmax_lengths = []
            rp = reader.row_ptr
            for rid in run_ids:
                kmax_lengths.append(int(rp[rid+1] - rp[rid]))
            kmax_lengths = np.asarray(kmax_lengths, dtype=np.int32)

            # derive common K
            policy = group_topk_policy.lower() if isinstance(group_topk_policy, str) else "none"
            common_K = None
            if policy == "min":
                common_K = int(kmax_lengths.min())
            elif policy in ("max", "legacy-min"):
                # legacy: old code used the largest K across runs (computationally heaviest)
                common_K = int(kmax_lengths.max())
            elif policy.startswith("fixed:"):
                common_K = int(policy.split(":",1)[1])
            # else: "none" keep vspec_default (mass/topk as provided)

            # Build views with possible per-run TopK override
            with profiler.stage(f"view_construction_p{pid}"):
                views = []
                lengths = []
                for rid, kmax in zip(run_ids, kmax_lengths):
                    if common_K is not None:
                        # clamp to the available length
                        K = min(common_K, int(kmax))
                        vsp = ViewSpec(agg=base_agg, cut=CutSpec(CutType.TOPK, float(K)), order=Order.BY_KEY)
                    else:
                        vsp = vspec_default

                    # Use window query if pos_window is specified
                    if pos_window is not None:
                        v = reader.get_window_view(rid, pos_lo, pos_hi, pos_size, vsp,
                                                  normalize_l1=(distance.lower()=="wj" and normalize))
                    else:
                        v = reader.get_run_view(rid, vsp,
                                               normalize_l1=(distance.lower()=="wj" and normalize))

                    views.append(v)
                    lengths.append(len(v.keys))

                    # Sample memory periodically (every 10 views)
                    if len(views) % 10 == 0:
                        profiler.sample_memory_in_stage()

                lengths = np.asarray(lengths, dtype=np.int32)

            # Compute distance matrix once
            with profiler.stage(f"distance_computation_p{pid}"):
                D = DistanceEngine(dspec).dense_matrix(views)
                profiler.sample_memory_in_stage()

            # Prepare stats for selectors
            run_stats = {"lengths": lengths, "views": views}

            # Run selectors and map indices to run_ids
            sel_out = {}
            sel_out_index = {} if emit_index else None

            with profiler.stage(f"selector_execution_p{pid}"):
                for s in selectors:
                    spec = SelectorSpec(name=s["name"], params=s.get("params"))
                    selector = build_selector(spec)

                    # Bind NAD context for user-defined selectors (optional)
                    try:
                        ctx = SelectorContext(
                            cache=reader,
                            problem_id=pid,
                            run_ids=run_ids,
                            views=views,
                            pos_window=(pos_lo, pos_hi) if pos_window else None,
                            pos_size=pos_size
                        )
                        if hasattr(selector, "bind"):
                            selector.bind(ctx)
                    except Exception as e:
                        logger.debug(f"Selector '{s['name']}' bind() skipped or failed: {e}")

                    # Selector returns group-internal indices (0 to n-1)
                    chosen_idx = selector.select(D, run_stats)

                    # Map indices to actual global run_ids
                    chosen_run_id = _map_indices_to_run_ids(chosen_idx, run_ids, s["name"], pid)

                    # Store the mapped run_id(s)
                    if isinstance(chosen_run_id, list):
                        sel_out[s["name"]] = chosen_run_id
                        if emit_index:
                            sel_out_index[s["name"]] = [int(x) for x in (chosen_idx if isinstance(chosen_idx, list) else [chosen_idx])]
                    else:
                        sel_out[s["name"]] = int(chosen_run_id)
                        if emit_index:
                            sel_out_index[s["name"]] = int(chosen_idx)

            results["problems"][pid] = {
                "num_runs": len(run_ids),
                "run_ids": [int(x) for x in run_ids],  # Include the actual run_ids for validation
                "lengths": lengths.tolist(),
                "kmax_lengths": kmax_lengths.tolist(),
                "group_topk_policy": group_topk_policy,
                "common_K": common_K,
                "selectors": sel_out
            }

            if emit_index:
                results["problems"][pid]["selectors_index"] = sel_out_index

            logger.debug(f"problem={pid} runs={len(run_ids)} run_ids={run_ids[:5]}{'...' if len(run_ids) > 5 else ''} selectors={sel_out}")

    # Add performance summary to results
    if enable_profiling:
        results["performance"] = profiler.get_summary()

    # Save results to JSON
    if out_json:
        with profiler.stage("save_results"):
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to {out_json}")

    # Print performance summary to terminal
    if enable_profiling:
        profiler.print_summary()

    return results
