#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.views.reader import CacheReader
from nad.ops.bestofn_extreme8 import discover_cache_entries


DEFAULT_INPUTS = [
    "submission/BestofN/best_only_ref030_t1024_scale100_rank.json",
    "submission/BestofN/mix_ref030_t1024_scale100_rank.json",
]
DEFAULT_CACHE_ROOT = "/home/jovyan/public-ro/MUI_HUB/cache_test"
DEFAULT_INSERT_TOPK_VALUES = list(range(4, 14))
LOCK_PREFIX = 3


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _rank_scale(n_items: int) -> list[float]:
    if n_items <= 0:
        return []
    if n_items == 1:
        return [100.0]
    return [
        100.0 - 99.0 * idx / float(n_items - 1)
        for idx in range(n_items)
    ]


def _mean_confidence_score(reader: CacheReader, sample_id: int) -> float:
    token_view = reader.get_token_view(int(sample_id))
    tok_conf = token_view.tok_conf
    if tok_conf is None or tok_conf.size == 0:
        return float("-inf")
    return float(-np.mean(tok_conf, dtype=np.float64))


def _ordered_sample_ids_from_scores(sample_map: OrderedDict[str, float]) -> list[str]:
    original_order = {sample_id: idx for idx, sample_id in enumerate(sample_map.keys())}
    return sorted(
        sample_map.keys(),
        key=lambda sample_id: (-float(sample_map[sample_id]), original_order[sample_id]),
    )


def _confidence_ranked_sample_ids(
    reader: CacheReader,
    sample_ids: Iterable[str],
    current_order: list[str],
) -> list[str]:
    current_rank = {sample_id: idx for idx, sample_id in enumerate(current_order)}
    scores = {
        sample_id: _mean_confidence_score(reader, int(sample_id))
        for sample_id in sample_ids
    }
    return sorted(
        sample_ids,
        key=lambda sample_id: (-scores[sample_id], current_rank[sample_id]),
    )


def _splice_order(
    current_order: list[str],
    confidence_order: list[str],
    insert_topk: int,
    lock_prefix: int = LOCK_PREFIX,
) -> list[str]:
    locked = list(current_order[:lock_prefix])
    locked_set = set(locked)

    injected: list[str] = []
    for sample_id in confidence_order:
        if sample_id in locked_set:
            continue
        injected.append(sample_id)
        if len(injected) >= int(insert_topk):
            break

    injected_set = set(injected)
    remainder = [
        sample_id
        for sample_id in current_order[lock_prefix:]
        if sample_id not in injected_set
    ]
    return locked + injected + remainder


def _rebuild_sample_map(ordered_sample_ids: list[str]) -> OrderedDict[str, float]:
    scaled_scores = _rank_scale(len(ordered_sample_ids))
    return OrderedDict(
        (sample_id, float(score))
        for sample_id, score in zip(ordered_sample_ids, scaled_scores)
    )


def _output_path(input_path: Path, insert_topk: int) -> Path:
    return input_path.with_name(
        f"{input_path.stem}_confins{int(insert_topk):02d}_after3{input_path.suffix}"
    )


def _method_name(method_name: str, insert_topk: int) -> str:
    return f"{method_name}_confins{int(insert_topk):02d}_after3"


def _load_cache_readers(cache_root: str) -> dict[str, CacheReader]:
    readers: dict[str, CacheReader] = {}
    for entry in discover_cache_entries(cache_root):
        readers[entry.cache_key] = CacheReader(str(entry.cache_root))
    return readers


def _validate_structure(original_scores: dict, updated_scores: dict, insert_topk: int) -> None:
    if set(original_scores.keys()) != set(updated_scores.keys()):
        raise ValueError("cache_key set changed during rewrite")

    for cache_key, original_problem_map in original_scores.items():
        updated_problem_map = updated_scores[cache_key]
        if set(original_problem_map.keys()) != set(updated_problem_map.keys()):
            raise ValueError(f"{cache_key}: problem_id set changed during rewrite")

        for problem_id, original_sample_map in original_problem_map.items():
            updated_sample_map = updated_problem_map[problem_id]
            original_ids = list(original_sample_map.keys())
            updated_ids = list(updated_sample_map.keys())

            if set(original_ids) != set(updated_ids):
                raise ValueError(f"{cache_key}/{problem_id}: sample_id set changed during rewrite")
            if len(updated_ids) != len(set(updated_ids)):
                raise ValueError(f"{cache_key}/{problem_id}: duplicate sample_ids after rewrite")
            if LOCK_PREFIX > 0:
                original_order = _ordered_sample_ids_from_scores(original_sample_map)
                updated_order = _ordered_sample_ids_from_scores(updated_sample_map)
                if updated_order[:LOCK_PREFIX] != original_order[:LOCK_PREFIX]:
                    raise ValueError(
                        f"{cache_key}/{problem_id}: top-{LOCK_PREFIX} changed after rewrite"
                    )
                expected_insert_len = min(
                    int(insert_topk),
                    max(0, len(updated_order) - LOCK_PREFIX),
                )
                if len(updated_order[LOCK_PREFIX:LOCK_PREFIX + expected_insert_len]) != expected_insert_len:
                    raise ValueError(
                        f"{cache_key}/{problem_id}: inserted segment length mismatch"
                    )

            values = list(updated_sample_map.values())
            if any(not np.isfinite(float(v)) for v in values):
                raise ValueError(f"{cache_key}/{problem_id}: non-finite score detected")
            if values:
                if max(values) > 100.0 + 1e-9 or min(values) < 1.0 - 1e-9:
                    raise ValueError(f"{cache_key}/{problem_id}: scores out of 1..100 range")
                if any(values[idx] < values[idx + 1] - 1e-9 for idx in range(len(values) - 1)):
                    raise ValueError(f"{cache_key}/{problem_id}: scores are not monotonic")


def rewrite_submission(
    input_path: Path,
    readers: dict[str, CacheReader],
    insert_topk_values: list[int],
) -> list[Path]:
    data = json.loads(input_path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    scores = data.get("scores")
    if not isinstance(scores, dict):
        raise ValueError(f"{input_path}: missing or invalid scores mapping")

    outputs: list[Path] = []
    for insert_topk in insert_topk_values:
        updated_scores: OrderedDict[str, OrderedDict[str, OrderedDict[str, float]]] = OrderedDict()

        for cache_key, problem_map in scores.items():
            if cache_key not in readers:
                raise KeyError(f"{input_path}: cache_key {cache_key!r} not found under cache root")
            reader = readers[cache_key]
            updated_problem_map: OrderedDict[str, OrderedDict[str, float]] = OrderedDict()

            for problem_id, sample_map in problem_map.items():
                current_order = _ordered_sample_ids_from_scores(sample_map)
                confidence_order = _confidence_ranked_sample_ids(
                    reader=reader,
                    sample_ids=current_order,
                    current_order=current_order,
                )
                updated_order = _splice_order(
                    current_order=current_order,
                    confidence_order=confidence_order,
                    insert_topk=int(insert_topk),
                    lock_prefix=LOCK_PREFIX,
                )
                updated_problem_map[str(problem_id)] = _rebuild_sample_map(updated_order)

            updated_scores[str(cache_key)] = updated_problem_map

        _validate_structure(scores, updated_scores, insert_topk=int(insert_topk))

        updated_data = OrderedDict(data)
        updated_data["method_name"] = _method_name(str(data["method_name"]), int(insert_topk))
        updated_data["scores"] = updated_scores
        score_postprocess = OrderedDict(updated_data.get("score_postprocess", {}))
        score_postprocess["reorder_rule"] = "lock_top3_insert_conf_topK"
        score_postprocess["insert_topk"] = int(insert_topk)
        score_postprocess["lock_prefix"] = LOCK_PREFIX
        score_postprocess["confidence_metric"] = "-mean(tok_conf)"
        score_postprocess["rest_order"] = "preserved"
        updated_data["score_postprocess"] = score_postprocess

        out_path = _output_path(input_path, int(insert_topk))
        out_path.write_text(
            json.dumps(updated_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs.append(out_path)

    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rewrite rank submissions by locking top-3 and injecting mean-confidence top-K after them."
    )
    ap.add_argument(
        "--inputs",
        default=",".join(DEFAULT_INPUTS),
        help="Comma-separated rank submission JSON paths",
    )
    ap.add_argument(
        "--cache-root",
        default=DEFAULT_CACHE_ROOT,
        help="Cache root used to compute mean-confidence ranks",
    )
    ap.add_argument(
        "--insert-topk-values",
        default=",".join(str(v) for v in DEFAULT_INSERT_TOPK_VALUES),
        help="Comma-separated K values to inject after top-3",
    )
    args = ap.parse_args()

    os.chdir(REPO_ROOT)

    input_paths = [
        (REPO_ROOT / raw.strip())
        for raw in str(args.inputs).split(",")
        if raw.strip()
    ]
    insert_topk_values = _parse_int_list(args.insert_topk_values)
    readers = _load_cache_readers(args.cache_root)

    for input_path in input_paths:
        outputs = rewrite_submission(
            input_path=input_path,
            readers=readers,
            insert_topk_values=insert_topk_values,
        )
        print(f"Input: {_display_path(input_path)}")
        for out_path in outputs:
            print(f"  -> {_display_path(out_path)}")


if __name__ == "__main__":
    main()
