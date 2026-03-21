import argparse, json, os, logging, sys
# Lazy imports to avoid circular dependency issues
# from ..pipeline.build_cache import build_cache
# from ..pipeline.analysis import analyze
# from ..pipeline.build_cache_fast import build_cache_fast


def setup_logging(log_level: str = "INFO"):
    """
    Configure logging for NAD CLI and all submodules.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set level for nad package
    logging.getLogger('nad').setLevel(numeric_level)


def main():
    p = argparse.ArgumentParser("nad CLI")

    # Global options (apply to all subcommands)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Set logging level (default: INFO)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # Original cache-build command
    b = sub.add_parser("cache-build", help="Build CSR and index cache from raw dir")
    b.add_argument("--raw-dir", required=True)
    b.add_argument("--cache-root", required=True)
    b.add_argument("--dataset-id", default="demo")
    b.add_argument("--model-id", default="demo")
    b.add_argument("--no-sum", action="store_true")

    # NEW: Fast parallel build with Map-Reduce
    bf = sub.add_parser("cache-build-fast",
                       help="Fast parallel build (Map-Reduce over shards, fixes 1920 samples issue)")
    bf.add_argument("--raw-dir", required=True,
                   help="Directory containing shard NPZ files")
    bf.add_argument("--meta-json", required=True,
                   help="Path to meta.json (contains samples/problems definition)")
    bf.add_argument("--cache-root", required=True,
                   help="Output cache directory")
    bf.add_argument("--dataset-id", default="v4",
                   help="Dataset identifier")
    bf.add_argument("--model-id", default="v4",
                   help="Model identifier")
    bf.add_argument("--workers", type=int, default=32,
                   help="Number of parallel workers (default: 32)")
    bf.add_argument("--row-bank", action="store_true",
                   help="Build Row-CSR Bank for window queries (v4.1+)")
    bf.add_argument("--pos-size", type=int, default=32,
                   help="Position size (tokens per position, default: 32)")

    # Analyze command
    a = sub.add_parser("analyze", help="Analyze problems with distance + selectors")
    a.add_argument("--cache-root", required=True,
                   help="Cache directory (must contain meta.json)")
    a.add_argument("--agg", default="max", choices=["max", "sum"])
    a.add_argument("--cut", default="mass:0.98", help="e.g., 'mass:0.98' or 'topk:8192'")
    a.add_argument("--distance", default="wj", choices=["wj", "ja"])
    a.add_argument("--no-normalize", action="store_true")
    a.add_argument("--selectors", default="all",
                   help='JSON list or "all" (default: all)')
    a.add_argument("--group-topk-policy", default="none",
                   help='Unified Top-K policy within a problem group: '
                        '"none" | "min" | "max" | "legacy-min" | "fixed:<K>"')
    a.add_argument("--emit-index", action="store_true",
                   help="Debug mode: emit both group-internal index and global run_id for each selector")
    a.add_argument("--pos-window", default=None,
                   help="Position window query (v4.1+): 'lo-hi' (e.g., '0-8') or 'all' to sweep all windows")
    a.add_argument("--pos-size", type=int, default=32,
                   help="Position size (tokens per position, default: 32)")
    a.add_argument("--pos-max", type=int, default=None,
                   help="Limit max position for 'all' mode (for testing, default: no limit)")
    a.add_argument("--enable-profiling", action="store_true",
                   help="Enable performance profiling (memory and timing)")
    a.add_argument("--distance-threads", type=int, default=16,
                   help="Number of threads for distance matrix computation (default: 16)")
    # Correct boolean flag pairing so users can disable the default
    a.add_argument("--assume-unique-keys", dest="assume_unique_keys", action="store_true",
                   help="Assume keys are unique within each run (enables faster np.intersect1d)")
    a.add_argument("--no-assume-unique-keys", dest="assume_unique_keys", action="store_false",
                   help="Do not assume keys are unique (safer but slower intersect1d)")
    a.set_defaults(assume_unique_keys=True)
    a.add_argument("--out", default=None)

    # NEW: accuracy subcommand (compute selector accuracy from selection JSON + cache ground truth)
    acc = sub.add_parser("accuracy", help="Compute selector accuracy from selection JSON + cache ground truth")
    acc.add_argument("--selection", required=True, help="Selection JSON file (output from analyze)")
    acc.add_argument("--cache-root", required=True, help="Cache root directory")
    acc.add_argument("--out", required=False, default=None, help="Output JSON file (optional)")

    args = p.parse_args()

    # Setup logging with user-specified level
    setup_logging(args.log_level)

    if args.cmd == "cache-build":
        from ..pipeline.build_cache import build_cache
        build_cache(args.raw_dir, args.cache_root,
                   dataset_id=args.dataset_id,
                   model_id=args.model_id,
                   use_sum=not args.no_sum)

    elif args.cmd == "cache-build-fast":
        from ..pipeline.build_cache_fast import build_cache_fast
        build_cache_fast(raw_dir=args.raw_dir,
                        cache_root=args.cache_root,
                        meta_json=args.meta_json,
                        dataset_id=args.dataset_id,
                        model_id=args.model_id,
                        workers=args.workers,
                        row_bank=args.row_bank,
                        pos_size=args.pos_size)

    elif args.cmd == "analyze":
        from ..pipeline.analysis import analyze
        selectors = json.loads(args.selectors) if args.selectors.strip().startswith("[") else args.selectors
        analyze(cache_root=args.cache_root,
               agg=args.agg,
               cut=args.cut,
               distance=args.distance,
               normalize=(not args.no_normalize),
               selectors=selectors,
               group_topk_policy=args.group_topk_policy,
               emit_index=args.emit_index,
               pos_window=args.pos_window,
               pos_size=args.pos_size,
               pos_max=args.pos_max,
               enable_profiling=args.enable_profiling,
               distance_threads=args.distance_threads,
               assume_unique_keys=args.assume_unique_keys,
               out_json=args.out)

    elif args.cmd == "accuracy":
        from ..ops.accuracy import compute_accuracy_report
        from pathlib import Path
        rep = compute_accuracy_report(args.selection, args.cache_root)
        print("Selector Accuracy Summary:")
        for k, ct in rep.selector_counts.items():
            c, t = ct
            acc = rep.selector_accuracy.get(k, 0.0)
            print(f"  {k:>28s}  {acc:6.2f}%  {c}/{t}")
        if args.out:
            out = {
                "selector_accuracy": rep.selector_accuracy,
                "selector_counts": {k: {"correct": c, "total": t} for k,(c,t) in rep.selector_counts.items()},
                "per_problem": rep.per_problem,
            }
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    else:
        p.print_help()


if __name__ == "__main__":
    main()
