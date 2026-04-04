#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.bestofn_extreme8 import (
    default_method_name,
    default_submission_filename,
    discover_cache_entries,
    validate_submission_payload,
    write_submission_payload,
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _chunked(items: list[str], n_chunks: int) -> list[list[str]]:
    n_chunks = max(1, min(int(n_chunks), len(items)))
    chunk_size = int(math.ceil(len(items) / n_chunks))
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in str(raw).split(',') if item.strip()]


def _build_balanced_3plus3_shards(cache_keys: list[str]) -> list[list[str]]:
    group_specs = [
        ['gpqa', 'aime24', 'aime25'],
        ['lcb_v5', 'brumo25', 'hmmt25'],
    ]
    by_model: dict[str, dict[str, str]] = {}
    for cache_key in cache_keys:
        model_name, dataset_name = cache_key.split('/', 1)
        by_model.setdefault(model_name, {})[dataset_name] = cache_key

    shards: list[list[str]] = []
    for model_name in sorted(by_model):
        datasets = by_model[model_name]
        used: set[str] = set()
        for group in group_specs:
            shard = [datasets[dataset_name] for dataset_name in group if dataset_name in datasets]
            if shard:
                shards.append(shard)
                used.update(group)
        leftovers = [datasets[name] for name in sorted(datasets) if name not in used]
        if leftovers:
            shards.append(leftovers)
    return shards


def _merge_payloads(paths: list[Path], expected_cache_keys: list[str]) -> dict:
    merged = None
    seen_cache_keys: set[str] = set()
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        if merged is None:
            merged = {
                'task': data['task'],
                'method_name': data['method_name'],
                'scores': {},
            }
        else:
            if data.get('task') != merged['task']:
                raise ValueError(f'task mismatch while merging {path}')
            if data.get('method_name') != merged['method_name']:
                raise ValueError(f'method_name mismatch while merging {path}')

        scores = data.get('scores', {})
        if not isinstance(scores, dict):
            raise ValueError(f'invalid scores payload in {path}')

        overlap = seen_cache_keys.intersection(scores.keys())
        if overlap:
            raise ValueError(f'duplicate cache keys while merging {path}: {sorted(overlap)}')

        merged['scores'].update(scores)
        seen_cache_keys.update(scores.keys())

    if merged is None:
        raise ValueError('no shard payloads found to merge')

    validate_submission_payload(merged, expected_cache_keys=expected_cache_keys)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description='Run Best-of-N export in safe cache-key shards and merge validated outputs')
    ap.add_argument('--cache-root', default='/home/jovyan/public-ro/MUI_HUB/cache_test', help='Root directory containing cache_test model/dataset subdirectories')
    ap.add_argument('--best-model', default='models/ml_selectors/extreme8_best.pkl', help='Best-model path')
    ap.add_argument('--worst-model', default='models/ml_selectors/extreme8_worst.pkl', help='Worst-model path')
    ap.add_argument('--out-dir', default='submission/BestofN', help='Final output directory')
    ap.add_argument('--work-dir', default='/tmp/bestofn_parallel_work', help='Temporary shard workspace')
    ap.add_argument('--parallel-jobs', type=int, default=4, help='Number of shard workers to run in parallel when grouping=chunked')
    ap.add_argument('--grouping', choices=('balanced_3plus3', 'chunked'), default='balanced_3plus3', help='Shard grouping policy')
    ap.add_argument('--cache-keys', default='', help='Optional comma-separated cache keys to include')
    ap.add_argument('--max-caches', type=int, default=None, help='Optional max caches for smoke tests')
    ap.add_argument('--max-problems', type=int, default=None, help='Optional max problems per cache for smoke tests')
    ap.add_argument('--tuple-size', type=int, default=8, help='Tuple size')
    ap.add_argument('--num-tuples', type=int, default=1024, help='Random tuples per problem')
    ap.add_argument('--reflection-threshold', type=float, default=0.30, help='Reflection threshold at inference time')
    ap.add_argument('--seed', type=int, default=42, help='Base random seed')
    ap.add_argument('--expected-samples-per-problem', type=int, default=64, help='Expected sample count per problem')
    ap.add_argument('--best-method-name', default='', help='Optional override for best-only method_name')
    ap.add_argument('--mix-method-name', default='', help='Optional override for mix method_name')
    ap.add_argument('--best-filename', default='', help='Optional override for best-only filename')
    ap.add_argument('--mix-filename', default='', help='Optional override for mix filename')
    ap.add_argument('--keep-work-dir', action='store_true', help='Keep shard workspace after success')
    args = ap.parse_args()

    os.chdir(REPO_ROOT)

    entries = discover_cache_entries(args.cache_root)
    requested_cache_keys = set(_parse_csv(args.cache_keys))
    if requested_cache_keys:
        entries = [entry for entry in entries if entry.cache_key in requested_cache_keys]
    if args.max_caches is not None:
        entries = entries[: max(0, int(args.max_caches))]
    if not entries:
        raise SystemExit('No cache_test entries were found.')

    cache_keys = [entry.cache_key for entry in entries]
    if args.grouping == 'balanced_3plus3':
        shards = _build_balanced_3plus3_shards(cache_keys)
    else:
        shards = _chunked(cache_keys, int(args.parallel_jobs))
    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f'Found {len(cache_keys)} cache keys; grouping={args.grouping}; running {len(shards)} shard workers')
    for shard_idx, shard_keys in enumerate(shards, start=1):
        print(f'  shard {shard_idx}: {shard_keys}')

    processes: list[tuple[int, subprocess.Popen, Path]] = []
    for shard_idx, shard_keys in enumerate(shards, start=1):
        shard_dir = work_dir / f'shard_{shard_idx:02d}'
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / 'run.log'
        cmd = [
            sys.executable,
            str(REPO_ROOT / 'scripts' / 'export_bestofn_submissions.py'),
            '--cache-root', str(args.cache_root),
            '--best-model', str(args.best_model),
            '--worst-model', str(args.worst_model),
            '--out-dir', str(shard_dir),
            '--tuple-size', str(int(args.tuple_size)),
            '--num-tuples', str(int(args.num_tuples)),
            '--reflection-threshold', str(float(args.reflection_threshold)),
            '--seed', str(int(args.seed) + (shard_idx - 1) * 1_000_000),
            '--expected-samples-per-problem', str(int(args.expected_samples_per_problem)),
            '--cache-keys', ','.join(shard_keys),
            '--best-method-name', args.best_method_name or default_method_name('best_only', float(args.reflection_threshold), int(args.num_tuples)),
            '--mix-method-name', args.mix_method_name or default_method_name('mix', float(args.reflection_threshold), int(args.num_tuples)),
            '--best-filename', args.best_filename or default_submission_filename('best_only', float(args.reflection_threshold), int(args.num_tuples)),
            '--mix-filename', args.mix_filename or default_submission_filename('mix', float(args.reflection_threshold), int(args.num_tuples)),
        ]
        if args.max_problems is not None:
            cmd.extend(['--max-problems', str(int(args.max_problems))])
        log_file = log_path.open('w', encoding='utf-8')
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        log_file.close()
        processes.append((shard_idx, proc, log_path))

    failed = []
    for shard_idx, proc, log_path in processes:
        code = proc.wait()
        if code != 0:
            failed.append((shard_idx, code, log_path))
        else:
            print(f'shard {shard_idx} finished successfully; log: {_display_path(log_path)}')

    if failed:
        lines = []
        for shard_idx, code, log_path in failed:
            lines.append(f'shard {shard_idx} failed with exit code {code}; log: {log_path}')
        raise SystemExit('\n'.join(lines))

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best_filename = args.best_filename or default_submission_filename('best_only', float(args.reflection_threshold), int(args.num_tuples))
    mix_filename = args.mix_filename or default_submission_filename('mix', float(args.reflection_threshold), int(args.num_tuples))

    best_paths = [work_dir / f'shard_{idx:02d}' / best_filename for idx in range(1, len(shards) + 1)]
    mix_paths = [work_dir / f'shard_{idx:02d}' / mix_filename for idx in range(1, len(shards) + 1)]

    merged_best = _merge_payloads(best_paths, expected_cache_keys=cache_keys)
    merged_mix = _merge_payloads(mix_paths, expected_cache_keys=cache_keys)

    best_out = write_submission_payload(merged_best, out_dir / best_filename)
    mix_out = write_submission_payload(merged_mix, out_dir / mix_filename)

    print(f'Saved merged best_only submission to {_display_path(best_out)}')
    print(f'Saved merged mix submission to {_display_path(mix_out)}')

    if not args.keep_work_dir:
        shutil.rmtree(work_dir)
        print(f'Removed temporary workspace {_display_path(work_dir)}')
    else:
        print(f'Kept temporary workspace {_display_path(work_dir)}')


if __name__ == '__main__':
    main()
