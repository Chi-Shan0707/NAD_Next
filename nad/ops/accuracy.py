#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selector accuracy computation as a reusable operator.

Inputs:
- selection JSON produced by "nad.cli analyze" or earlier tools (problems -> selectors -> selected run index)
- cache root containing meta.json and evaluation_report.json / evaluation_report_compact.json

Output:
- Dict with selector_accuracy, per_problem details, totals
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import re
import ast
from collections import Counter

def load_correctness_map(cache_root: str):
    """Public helper: map sample_id -> is_correct from cache root."""
    from pathlib import Path
    return _load_ground_truth(Path(cache_root))


@dataclass
class AccuracyReport:
    selector_accuracy: Dict[str, float]
    selector_counts: Dict[str, Tuple[int,int]]  # correct, total
    per_problem: Dict[str, Dict]

def _load_ground_truth(cache_root: Path) -> Dict[int, bool]:
    """
    Build sample_id -> correctness map from evaluation_report.* + meta.json
    """
    cache_root = Path(cache_root)
    meta_json = cache_root / "meta.json"
    if not meta_json.exists():
        raise FileNotFoundError(f"缺少 meta.json: {meta_json}")
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    # Build (problem_id, run_index) -> sample_id
    sample_index = {}
    for sample_id, sample in enumerate(meta.get("samples", [])):
        # Always use problem_id as string for consistency
        pid = str(sample["problem_id"])
        sample_index[(pid, int(sample.get("run_index", 0)))] = int(sample_id)

    # Prefer compact report
    gt_files = [
        cache_root / "evaluation_report_compact.json",
        cache_root / "evaluation_report.json"
    ]
    gt = None
    for f in gt_files:
        if f.exists():
            gt = json.loads(f.read_text(encoding="utf-8"))
            break
    if gt is None:
        raise FileNotFoundError("未找到 evaluation_report_compact.json 或 evaluation_report.json")

    correctness_map: Dict[int, bool] = {}

    # Two schema variants
    if "results" in gt:
        for result in gt["results"]:
            # Always use problem_id as string for consistency
            pid = str(result["problem_id"])
            for run in result.get("runs", []):
                ri = int(run.get("run_index", run.get("index", 0)))
                sid = sample_index.get((pid, ri))
                if sid is not None:
                    correctness_map[sid] = bool(run.get("is_correct", False))
    elif "sample_breakdown" in gt:
        bd = gt["sample_breakdown"]
        for sid in bd.get("correct_samples", []):
            correctness_map[int(sid)] = True
        for sid in bd.get("incorrect_samples", []):
            correctness_map[int(sid)] = False
    else:
        raise ValueError("无法识别 ground truth JSON 结构")

    return correctness_map

def _load_evaluation_report_full(cache_root: Path):
    """
    Load full evaluation report with all runs' answers and correctness.
    Returns: {problem_id: {"runs": [...], "ground_truth": ...}}
    """
    cache_root = Path(cache_root)
    gt_files = [
        cache_root / "evaluation_report_compact.json",
        cache_root / "evaluation_report.json"
    ]

    for f in gt_files:
        if f.exists():
            report = json.loads(f.read_text(encoding="utf-8"))
            if "results" in report:
                # Convert to easier format
                problem_data = {}
                for result in report["results"]:
                    pid = str(result["problem_id"])
                    runs = []
                    for run in result.get("runs", []):
                        # Parse extracted_answer: handles both formats
                        # - AIME format: "[204, '204']" -> "204"
                        # - GPQA format: "C" -> "C"
                        answer_str = run.get("extracted_answer", "")

                        if not answer_str:
                            answer = ""
                        else:
                            try:
                                # Try to parse as Python literal (for AIME format)
                                parsed = ast.literal_eval(answer_str)
                                if isinstance(parsed, list) and len(parsed) > 0:
                                    answer = str(parsed[0])
                                else:
                                    answer = str(parsed) if parsed is not None else ""
                            except (ValueError, SyntaxError):
                                # If parsing fails, use the original string (for GPQA format)
                                answer = str(answer_str)

                        runs.append({
                            "run_index": int(run.get("run_index", run.get("index", 0))),
                            "answer": answer,
                            "is_correct": bool(run.get("is_correct", False))
                        })

                    problem_data[pid] = {
                        "runs": runs,
                        "ground_truth": result.get("ground_truth", "")
                    }

                return problem_data

    return {}

def _compute_baseline_accuracy(selector_name: str, problems_dict: Dict, eval_report: Dict) -> Tuple[int, int]:
    """
    Compute accuracy for baseline selectors (avgN@, conN@).

    Args:
        selector_name: Name of selector (e.g., "avg64@", "con64@")
        problems_dict: Problem list from selection JSON
        eval_report: Full evaluation report from _load_evaluation_report_full()

    Returns:
        (correct, total) tuple
        - For avgN@: (sum of avg accuracies * 100, total problems * 100) to preserve precision
        - For conN@: (num problems with correct majority, total problems)
    """
    if selector_name.startswith("avg"):
        # avgN@: Average accuracy across all runs per problem, then average across problems
        total_avg_accuracy = 0.0
        num_problems = 0

        for pid in problems_dict.keys():
            if pid not in eval_report:
                continue

            runs = eval_report[pid]["runs"]
            if not runs:
                continue

            # Calculate average accuracy for this problem
            num_correct = sum(1 for r in runs if r["is_correct"])
            problem_avg_accuracy = num_correct / len(runs)
            total_avg_accuracy += problem_avg_accuracy
            num_problems += 1

        if num_problems == 0:
            return (0, 0)

        # Return as scaled integers to fit existing format
        # Overall average accuracy = total_avg_accuracy / num_problems
        # We return it as (accuracy_sum * 100, num_problems * 100)
        # so that correct/total gives the right percentage
        overall_avg = total_avg_accuracy / num_problems
        return (int(overall_avg * num_problems * 100), num_problems * 100)

    elif selector_name.startswith("con"):
        # conN@: Majority voting - check if most common answer is correct
        correct = 0
        total = 0

        for pid in problems_dict.keys():
            if pid not in eval_report:
                continue

            runs = eval_report[pid]["runs"]
            if not runs:
                continue

            # Find majority answer
            answers = [r["answer"] for r in runs if r["answer"]]
            if not answers:
                continue

            majority_answer = Counter(answers).most_common(1)[0][0]

            # Check if majority answer is correct
            # Find a run with this answer and check its correctness
            is_majority_correct = False
            for r in runs:
                if r["answer"] == majority_answer:
                    is_majority_correct = r["is_correct"]
                    break

            if is_majority_correct:
                correct += 1
            total += 1

        return (correct, total)

    return (0, 0)

def compute_accuracy_report(selection_json: Path, cache_root: Path) -> AccuracyReport:
    sel = json.loads(Path(selection_json).read_text(encoding="utf-8"))
    problems = sel.get("problems") or {}
    if not problems:
        raise ValueError("Selection JSON 缺少 'problems' 字段")

    correctness_map = _load_ground_truth(Path(cache_root))

    # Determine selector names from the first problem
    first = next(iter(problems.values()))
    selector_names = list((first.get("selectors") or {}).keys())
    if not selector_names:
        raise ValueError("Selection JSON 缺少 selectors 数据")

    # Identify baseline selectors (avgN@, conN@)
    baseline_selectors = [name for name in selector_names if re.match(r'^(avg|con)\d+@$', name)]
    standard_selectors = [name for name in selector_names if name not in baseline_selectors]

    # Load evaluation report for baseline selectors
    eval_report = {}
    if baseline_selectors:
        eval_report = _load_evaluation_report_full(Path(cache_root))

    # Build sample_id -> run_index mapping using meta.json
    # NOTE: The selector output from analyze() is already sample_id (global 0-1919),
    # not run_index (local per problem). We just need to get run_index for display.
    meta = json.loads((Path(cache_root) / "meta.json").read_text(encoding="utf-8"))
    sid_to_rindex = {}
    for sid, s in enumerate(meta.get("samples", [])):
        sid_to_rindex[int(sid)] = int(s.get("run_index", 0))

    # Tally
    correct = {name: 0 for name in selector_names}
    total = {name: 0 for name in selector_names}
    per_problem: Dict[str, Dict] = {}

    # Handle baseline selectors first
    for name in baseline_selectors:
        c, t = _compute_baseline_accuracy(name, problems, eval_report)
        correct[name] = c
        total[name] = t

    for pid_str, pdata in problems.items():
        # Always use problem_id as string for consistency
        pid = str(pid_str)
        sel_dict = pdata.get("selectors") or {}
        entry = {"problem_id": pid, "selectors": {}}
        for name in standard_selectors:  # Only process standard selectors here
            sd = sel_dict.get(name)
            if sd is None:
                # Skip if not present
                continue

            # Extract sample_id: selector output is already a sample_id (not run_index!)
            if isinstance(sd, int):
                sid = sd  # This is a sample_id
            elif isinstance(sd, dict):
                # Accept multiple possible keys from dict
                sid = sd.get("sample_id")
                if sid is None:
                    sid = sd.get("selected")
                if sid is None and isinstance(sd.get("selected_run_index"), int):
                    # Fallback: if dict has run_index but not sample_id (shouldn't happen)
                    sid = int(sd.get("selected_run_index"))
                if sid is None:
                    # Skip if not present
                    continue
            else:
                # Unknown type, skip
                continue

            # Get run_index for display (optional)
            ri = sid_to_rindex.get(int(sid), -1)

            # Direct lookup using sample_id
            is_corr = bool(correctness_map.get(int(sid), False))
            correct[name] += 1 if is_corr else 0
            total[name] += 1
            entry["selectors"][name] = {"run_index": int(ri), "sample_id": int(sid), "is_correct": is_corr}
        per_problem[str(pid)] = entry

    selector_accuracy = {k: (100.0 * correct[k] / total[k] if total[k] else 0.0) for k in selector_names}
    selector_counts = {k: (int(correct[k]), int(total[k])) for k in selector_names}

    return AccuracyReport(selector_accuracy=selector_accuracy, selector_counts=selector_counts, per_problem=per_problem)

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Compute selector accuracy report")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out", required=False)
    args = parser.parse_args()
    rep = compute_accuracy_report(args.selection, args.cache_root)
    # Print summary
    print("Selector Accuracy Summary:")
    for k, (c,t) in rep.selector_counts.items():
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
