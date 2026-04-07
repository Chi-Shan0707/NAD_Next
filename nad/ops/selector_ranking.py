"""
NAD Selector Multi-Task Ranking System

This module implements a comprehensive ranking algorithm for selectors across
multiple model×dataset combinations using task-level normalization and
cross-task aggregation.

Based on the algorithm provided, it computes:
- RNS (Rank-Normalized Score)
- Relative Regret
- Copeland scores
- Cross-task aggregation with configurable weights
- Bootstrap confidence intervals
"""

from __future__ import annotations
import json
import math
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# -----------------------------
# Task Category Definitions
# -----------------------------

TASK_CATEGORIES = {
    'programming': ['mbpp', 'humaneval', 'livecodebench', 'livecodebench_v5'],
    'math': ['aime24', 'aime25', 'brumo25', 'hmmt25'],
    'science': ['gpqa']
}

def filter_tasks_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Filter tasks DataFrame by category (programming, math, science, or all).

    Args:
        df: DataFrame with columns including 'dataset' or 'task_id'
        category: Category name ('all', 'programming', 'math', 'science')

    Returns:
        Filtered DataFrame containing only tasks from the specified category
    """
    if category == 'all' or category is None:
        return df

    if category not in TASK_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Valid options: all, programming, math, science")

    datasets_in_category = TASK_CATEGORIES[category]

    # Check if we have a 'dataset' column
    if 'dataset' in df.columns:
        return df[df['dataset'].isin(datasets_in_category)]
    # Otherwise try to extract from task_id
    elif 'task_id' in df.columns:
        # task_id format is "model_name/dataset_name"
        df_copy = df.copy()
        df_copy['_dataset'] = df_copy['task_id'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
        result = df_copy[df_copy['_dataset'].isin(datasets_in_category)].drop(columns=['_dataset'])
        return result
    else:
        raise ValueError("DataFrame must have either 'dataset' or 'task_id' column for filtering")

def get_category_stats(df: pd.DataFrame, category: str) -> Dict[str, Any]:
    """
    Get statistics for a specific category.

    Args:
        df: DataFrame with task data
        category: Category name

    Returns:
        Dictionary with category statistics
    """
    filtered = filter_tasks_by_category(df, category)

    if filtered.empty:
        return {
            'name': category,
            'tasks': 0,
            'total_problems': 0,
            'datasets': []
        }

    # Get unique tasks
    if 'task_id' in filtered.columns:
        unique_tasks = filtered['task_id'].nunique()
        # Extract datasets from task_ids
        datasets = filtered['task_id'].apply(lambda x: x.split('/')[-1] if '/' in x else x).unique().tolist()
    else:
        unique_tasks = len(filtered)
        datasets = filtered.get('dataset', pd.Series()).unique().tolist()

    # Sum up problems if 'n' column exists
    total_problems = filtered['n'].sum() if 'n' in filtered.columns else 0

    return {
        'name': category,
        'tasks': unique_tasks,
        'total_problems': int(total_problems),
        'datasets': sorted(datasets)
    }

# -----------------------------
# 1) Data Loading and Validation
# -----------------------------

def load_task_data(cache_path: Path, results_path: Path = None) -> Dict[str, Any]:
    """
    Load accuracy data and problem count from a cache directory.

    Args:
        cache_path: Path to cache directory containing meta.json and accuracy files
        results_path: Optional path to results directory where accuracy_full_sequence.json might be stored

    Returns:
        Dictionary with task data including problem count and accuracies
    """
    cache_path = Path(cache_path)

    # Load meta.json to get actual problem count (NOT hardcoded!)
    meta_path = cache_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {cache_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    # Extract unique problem IDs to get actual count
    problem_ids = set()
    for sample in meta.get("samples", []):
        problem_ids.add(str(sample.get("problem_id")))

    n = len(problem_ids)
    if n == 0:
        logger.warning(f"No problems found in {cache_path}")

    # Load accuracy file with per-problem data
    # Try multiple locations for the accuracy file
    per_problem = {}
    selector_counts = {}
    selector_accuracy = {}
    accuracy_found = False

    # First try cache directory
    accuracy_path = cache_path / "accuracy_full_sequence.json"

    # If not found in cache and results_path provided, try results directory
    if not accuracy_path.exists() and results_path:
        # Extract model and dataset names from cache path
        cache_parts = cache_path.parts
        if len(cache_parts) >= 3:
            # Assume structure: .../<model>/<dataset>/<cache_name>
            model_name = cache_parts[-3]
            dataset_name = cache_parts[-2]
            cache_name = cache_parts[-1]
            results_accuracy_path = results_path / model_name / dataset_name / cache_name / "accuracy_full_sequence.json"
            if results_accuracy_path.exists():
                accuracy_path = results_accuracy_path
                accuracy_found = True

    if accuracy_path.exists() or accuracy_found:
        with open(accuracy_path) as f:
            accuracy_data = json.load(f)
        selector_accuracy = accuracy_data.get("selector_accuracy", {})
        selector_counts = accuracy_data.get("selector_counts", {})

        # Extract per-problem correctness
        for pid, prob_data in accuracy_data.get("per_problem", {}).items():
            per_problem[str(pid)] = {}
            for sel_name, sel_data in prob_data.get("selectors", {}).items():
                per_problem[str(pid)][sel_name] = sel_data.get("is_correct", False)
    else:
        selector_accuracy = {}
        # Only warn if we actually tried to find the file
        logger.debug(f"No accuracy file found for {cache_path}")

    return {
        "n": n,  # True problem count for weights
        "accuracies": selector_accuracy,  # Selector -> accuracy percentage
        "per_problem": per_problem,  # Problem -> selector -> is_correct
        "selector_counts": selector_counts,  # Selector -> {correct, total}
        "cache_path": str(cache_path)
    }

def _ensure_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has proper accuracy values.
    Handles both direct accuracy and correct/total formats.
    """
    df = df.copy()
    must_have = {"task_id", "selector"}
    if not must_have.issubset(df.columns):
        raise ValueError("Must have columns: task_id, selector")

    if "accuracy" not in df.columns:
        if {"correct", "total"}.issubset(df.columns):
            # Handle avg64@ special case where total might be inflated
            df["accuracy"] = df["correct"] / df["total"]
        else:
            raise ValueError("Missing accuracy; also no correct/total provided")

    if "n" not in df.columns:
        raise ValueError("Missing column n (task problem count from meta.json)")

    # Convert percentage to fraction if needed
    if df["accuracy"].max() > 1.5:  # Likely percentage
        df["accuracy"] = df["accuracy"] / 100.0

    # Constrain to valid range
    df["accuracy"] = df["accuracy"].clip(lower=0.0, upper=1.0)

    return df[["task_id", "selector", "n", "accuracy"]].copy()

def _drop_uninformative_tasks(
    df: pd.DataFrame,
    min_spread: float = 0.005,
    tie_tol: float = 0.0
) -> pd.DataFrame:
    """
    Remove tasks with insufficient variation.

    Args:
        df: Input DataFrame
        min_spread: Minimum max-min spread required
        tie_tol: Tolerance for considering values equal

    Returns:
        Filtered DataFrame excluding uninformative tasks
    """
    keep_task_ids = []
    excluded_tasks = []

    for tid, g in df.groupby("task_id"):
        p = g["accuracy"].to_numpy()
        pmax, pmin = p.max(), p.min()
        spread = pmax - pmin

        if spread <= max(tie_tol, min_spread):
            # Low variation task - exclude
            excluded_tasks.append((tid, spread))
            logger.info(f"Excluded task '{tid}': spread={spread:.4f} ≤ {min_spread}")
        else:
            keep_task_ids.append(tid)

    if excluded_tasks:
        logger.info(f"Excluded {len(excluded_tasks)} low-variation tasks")

    return df[df["task_id"].isin(keep_task_ids)].copy()

# -----------------------------
# 2) Task-Level Metrics
# -----------------------------

def _avg_rank_with_ties(series: pd.Series, descending: bool = True) -> pd.Series:
    """
    Compute ranks with average handling for ties.
    """
    return series.rank(ascending=not descending, method="average")

def _task_rns_regret(df_task: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RNS and regret for a single task.

    RNS = 1 - (rank-1)/(K-1) where K = number of selectors
    regret = (p*_t - p_s,t)/max(p*_t, ε)
    """
    df_task = df_task.copy()

    # Ranks (1 = best)
    ranks = _avg_rank_with_ties(df_task["accuracy"])
    K = len(df_task)

    if K <= 1:
        df_task["rank"] = 1.0
        df_task["RNS"] = 1.0
    else:
        df_task["rank"] = ranks
        df_task["RNS"] = 1.0 - (ranks - 1.0) / (K - 1.0)

    # Regret
    p_star = df_task["accuracy"].max()
    eps = 1e-12
    df_task["regret"] = (p_star - df_task["accuracy"]) / max(p_star, eps)

    return df_task[["selector", "accuracy", "rank", "RNS", "regret"]].copy()

def _copeland_per_task(df_task: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Copeland scores (pairwise wins-losses) for a single task.
    Normalized to [-1, 1].
    """
    acc = df_task.set_index("selector")["accuracy"].to_dict()
    sels = list(acc.keys())
    raw = dict.fromkeys(sels, 0.0)

    # Pairwise comparisons
    for i in range(len(sels)):
        for j in range(i + 1, len(sels)):
            si, sj = sels[i], sels[j]
            pi, pj = acc[si], acc[sj]

            if abs(pi - pj) < 1e-12:
                # Tie - no change
                continue
            elif pi > pj:
                raw[si] += 1.0
                raw[sj] -= 1.0
            else:
                raw[sj] += 1.0
                raw[si] -= 1.0

    # Normalize to [-1, 1]
    K = len(sels)
    if K > 1:
        for s in raw:
            raw[s] = raw[s] / (K - 1.0)

    return raw

# -----------------------------
# 3) Cross-Task Aggregation
# -----------------------------

def _task_weight(n: int, mode: str = "sqrt") -> float:
    """
    Compute task weight based on problem count.

    Modes:
      - 'sqrt': sqrt(n) - balanced between micro and macro
      - 'micro': n - weighted by problem count
      - 'macro': 1 - all tasks equal
    """
    if mode == "sqrt":
        return math.sqrt(max(0, n))
    elif mode == "micro":
        return float(n)
    elif mode == "macro":
        return 1.0
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

def compute_global_scores(
    df_in: pd.DataFrame,
    *,
    min_spread: float = 0.005,
    weight_mode: str = "sqrt",
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    min_coverage: float = 0.0,
    min_tasks: int = 1,
    coverage_penalty_lambda: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to compute global selector rankings.

    Args:
        df_in: Input DataFrame with columns (task_id, selector, n, accuracy)
        min_spread: Minimum spread to keep task
        weight_mode: How to weight tasks ('sqrt', 'micro', 'macro')
        alpha, beta, gamma: Weights for RNS, (1-regret), Copeland in final score
        min_coverage: Minimum coverage for MAIN tier
        min_tasks: Minimum tasks for MAIN tier
        coverage_penalty_lambda: Penalty exponent for low coverage

    Returns:
        df_scores: Global rankings with all metrics
        df_task_details: Per-task detailed metrics
    """
    # Prepare data
    df = _ensure_accuracy(df_in)

    # Drop uninformative tasks
    df_orig = df.copy()
    df = _drop_uninformative_tasks(df, min_spread=min_spread)

    # Track excluded tasks
    all_tasks_orig = set(df_orig["task_id"].unique())
    all_tasks = set(df["task_id"].unique())
    excluded_tasks = all_tasks_orig - all_tasks

    if excluded_tasks:
        logger.info(f"Excluded tasks: {excluded_tasks}")

    # Compute task-level metrics
    task_details = []
    copeland_accum = {}
    weight_accum = {}
    top_share_weighted = {}

    for tid, g in df.groupby("task_id"):
        n_t = int(g["n"].iloc[0])
        w_t = _task_weight(n_t, weight_mode)

        # RNS and regret
        td = _task_rns_regret(g[["selector", "accuracy"]])
        td.insert(0, "task_id", tid)
        td.insert(1, "n", n_t)
        task_details.append(td)

        # Copeland
        c_t = _copeland_per_task(g[["selector", "accuracy"]])
        for s, v in c_t.items():
            copeland_accum[s] = copeland_accum.get(s, 0.0) + w_t * v
            weight_accum[s] = weight_accum.get(s, 0.0) + w_t

        # Top share (fraction of being best)
        best = td["accuracy"].max()
        winners = td.loc[
            td["accuracy"].apply(lambda x: abs(x - best) < 1e-12),
            "selector"
        ].tolist()
        share = 1.0 / len(winners) if winners else 0.0

        for s in td["selector"]:
            add = share if s in winners else 0.0
            top_share_weighted[s] = top_share_weighted.get(s, 0.0) + w_t * add

    df_task_details = pd.concat(task_details, ignore_index=True)

    # Cross-task aggregation
    agg_rows = []

    for s, df_s in df_task_details.groupby("selector"):
        # Task weights
        w = df_s["n"].apply(lambda x: _task_weight(int(x), weight_mode)).to_numpy()

        # Weighted metrics
        rns_w = np.average(df_s["RNS"].to_numpy(), weights=w)
        regret_w = np.average(df_s["regret"].to_numpy(), weights=w)

        # Percentiles for regret
        regret_p90 = np.percentile(df_s["regret"].to_numpy(), 90)

        # Micro-average accuracy
        acc_micro = np.average(
            df_s["accuracy"].to_numpy(),
            weights=df_s["n"].to_numpy()
        )

        # Copeland normalized
        copeland_norm = copeland_accum.get(s, 0.0) / max(weight_accum.get(s, 1e-12), 1e-12)

        # Top share
        top_share = top_share_weighted.get(s, 0.0) / max(weight_accum.get(s, 1e-12), 1e-12)

        # Coverage
        tasks_s = set(df_s["task_id"].unique())
        coverage = len(tasks_s) / max(len(all_tasks), 1)
        evidence_w = sum(w)

        # Tier classification
        tier = "MAIN" if (coverage >= min_coverage and len(tasks_s) >= min_tasks) else "PROVISIONAL"

        # Final score
        score = (alpha * rns_w) + (beta * (1.0 - regret_w)) + (gamma * copeland_norm)

        # Coverage penalty
        if coverage_penalty_lambda > 0 and coverage < 1.0:
            score *= coverage ** coverage_penalty_lambda

        agg_rows.append({
            "selector": s,
            "micro_accuracy": acc_micro,
            "RNS_w": rns_w,
            "regret_w": regret_w,
            "regret_p90": regret_p90,
            "copeland_norm": copeland_norm,
            "top_share": top_share,
            "coverage": coverage,
            "tasks_count": len(tasks_s),
            "evidence_w": evidence_w,
            "tier": tier,
            "Score": score
        })

    df_scores = pd.DataFrame(agg_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    df_scores.insert(0, "rank", np.arange(1, len(df_scores) + 1))

    return df_scores, df_task_details

# -----------------------------
# 4) Bootstrap Confidence Intervals
# -----------------------------

def bootstrap_score_ci(
    df_in: pd.DataFrame,
    *,
    B: int = 5000,
    seed: int = 42,
    **compute_kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute bootstrap confidence intervals by resampling tasks.

    Args:
        df_in: Input DataFrame
        B: Number of bootstrap samples
        seed: Random seed for reproducibility
        compute_kwargs: Arguments for compute_global_scores

    Returns:
        df_ci: Confidence intervals for each selector
        P: Pairwise comparison matrix with differences and probabilities
    """
    rng = np.random.default_rng(seed)

    # Prepare base data
    base_df = _ensure_accuracy(df_in)
    base_df = _drop_uninformative_tasks(
        base_df,
        min_spread=compute_kwargs.get("min_spread", 0.005)
    )

    selectors = sorted(base_df["selector"].unique().tolist())
    selector_index = {s: idx for idx, s in enumerate(selectors)}

    weight_mode = compute_kwargs.get("weight_mode", "sqrt")
    alpha = compute_kwargs.get("alpha", 0.5)
    beta = compute_kwargs.get("beta", 0.3)
    gamma = compute_kwargs.get("gamma", 0.2)
    coverage_penalty_lambda = compute_kwargs.get("coverage_penalty_lambda", 0.0)

    tasks = []
    task_records = []

    for task_idx, (tid, g) in enumerate(base_df.groupby("task_id", sort=False)):
        tasks.append(tid)
        n_t = int(g["n"].iloc[0])
        w_t = float(_task_weight(n_t, weight_mode))

        td = _task_rns_regret(g[["selector", "accuracy"]].copy())
        c_t = _copeland_per_task(g[["selector", "accuracy"]])

        selector_indices = []
        rns_vals = []
        regret_vals = []
        copeland_vals = []

        for row in td.itertuples():
            selector_indices.append(selector_index[row.selector])
            rns_vals.append(float(row.RNS))
            regret_vals.append(float(row.regret))
            copeland_vals.append(float(c_t[row.selector]))

        task_records.append({
            "bit": 1 << task_idx,
            "weight": w_t,
            "selector_indices": np.array(selector_indices, dtype=np.int64),
            "RNS": np.array(rns_vals, dtype=np.float64),
            "regret": np.array(regret_vals, dtype=np.float64),
            "copeland": np.array(copeland_vals, dtype=np.float64)
        })

    n_tasks = len(tasks)
    n_selectors = len(selectors)

    if n_tasks == 0 or n_selectors == 0:
        raise ValueError("No tasks/selectors available for bootstrap sampling")

    # Collect bootstrap samples
    scores_stack = {s: [] for s in selectors}

    # Preallocate accumulators for reuse
    weight_sum = np.zeros(n_selectors, dtype=np.float64)
    rns_sum = np.zeros(n_selectors, dtype=np.float64)
    regret_sum = np.zeros(n_selectors, dtype=np.float64)
    copeland_sum = np.zeros(n_selectors, dtype=np.float64)

    # Add progress bar for bootstrap iterations
    use_tqdm = False
    print_interval = max(1, B // 20)  # Print progress every 5%

    try:
        from tqdm import tqdm
        # Force tqdm to output even in non-TTY environments
        iterator = tqdm(range(B), desc="Bootstrap sampling", unit="samples",
                       disable=False, dynamic_ncols=True, file=sys.stderr)
        use_tqdm = True
    except ImportError:
        iterator = range(B)
        logger.info(f"Starting {B} bootstrap samples...")

    for b in iterator:
        # Print manual progress if no tqdm
        if not use_tqdm and b > 0 and b % print_interval == 0:
            progress = (b / B) * 100
            logger.info(f"Bootstrap progress: {progress:.1f}% ({b}/{B} samples)")

        weight_sum.fill(0.0)
        rns_sum.fill(0.0)
        regret_sum.fill(0.0)
        copeland_sum.fill(0.0)
        coverage_mask = [0] * n_selectors
        total_mask = 0

        sampled_indices = rng.integers(0, n_tasks, size=n_tasks)

        for idx in sampled_indices:
            record = task_records[idx]
            bit = record["bit"]
            total_mask |= bit

            sel_idx = record["selector_indices"]
            if sel_idx.size == 0:
                continue

            w = record["weight"]

            weight_sum[sel_idx] += w
            rns_sum[sel_idx] += w * record["RNS"]
            regret_sum[sel_idx] += w * record["regret"]
            copeland_sum[sel_idx] += w * record["copeland"]

            for s_idx in sel_idx:
                coverage_mask[s_idx] |= bit

        total_tasks = max(total_mask.bit_count(), 1)

        for sel_idx, sel_name in enumerate(selectors):
            w = weight_sum[sel_idx]
            if w <= 0:
                scores_stack[sel_name].append(np.nan)
                continue

            rns_w = rns_sum[sel_idx] / w
            regret_w = regret_sum[sel_idx] / w
            copeland_norm = copeland_sum[sel_idx] / max(w, 1e-12)

            score = (alpha * rns_w) + (beta * (1.0 - regret_w)) + (gamma * copeland_norm)

            if coverage_penalty_lambda > 0:
                coverage = coverage_mask[sel_idx].bit_count() / total_tasks
                if coverage < 1.0:
                    score *= coverage ** coverage_penalty_lambda

            scores_stack[sel_name].append(score)

    # Compute CIs
    ci_rows = []
    for s in selectors:
        arr = np.array([v for v in scores_stack[s] if not np.isnan(v)], dtype=float)

        if arr.size == 0:
            ci_rows.append({
                "selector": s,
                "Score_mean": np.nan,
                "Score_ci_lower": np.nan,
                "Score_ci_upper": np.nan
            })
            continue

        ci_rows.append({
            "selector": s,
            "Score_mean": float(np.mean(arr)),
            "Score_ci_lower": float(np.percentile(arr, 2.5)),
            "Score_ci_upper": float(np.percentile(arr, 97.5))
        })

    df_ci = pd.DataFrame(ci_rows).sort_values("Score_mean", ascending=False, na_position="last").reset_index(drop=True)

    # Compute pairwise differences and probabilities
    P_data = []

    for i, si in enumerate(selectors):
        for j, sj in enumerate(selectors):
            if i >= j:
                continue

            ai = np.array([v for v in scores_stack[si] if not np.isnan(v)], dtype=float)
            aj = np.array([v for v in scores_stack[sj] if not np.isnan(v)], dtype=float)

            if len(ai) == 0 or len(aj) == 0:
                continue

            # Align lengths
            L = min(len(ai), len(aj))
            ai, aj = ai[:L], aj[:L]

            # Compute difference statistics
            diff = ai - aj

            P_data.append({
                "comparison": f"{si} - {sj}",
                "mean_diff": float(np.mean(diff)),
                "diff_ci_lower": float(np.percentile(diff, 2.5)),
                "diff_ci_upper": float(np.percentile(diff, 97.5)),
                "prob_greater": float(np.mean(diff > 0))
            })

    P = pd.DataFrame(P_data)

    return df_ci, P

# -----------------------------
# 5) Convenience Functions
# -----------------------------

# -----------------------------
# 5) Enhanced Metrics Calculation
# -----------------------------

def calculate_problem_difficulty(per_problem_data: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, float]:
    """
    Calculate difficulty score for each problem based on overall success rate.

    Args:
        per_problem_data: Dict of task_id -> problem_id -> selectors data

    Returns:
        Dict of problem_id -> difficulty score (0=easy, 1=hard)
    """
    problem_success = {}
    problem_attempts = {}

    for task_id, task_problems in per_problem_data.items():
        for problem_id, problem_data in task_problems.items():
            # Skip metadata entries
            if problem_id in ["problem_id", "selectors"]:
                continue

            # Create unique problem ID
            unique_pid = f"{task_id}::{problem_id}"
            if unique_pid not in problem_success:
                problem_success[unique_pid] = 0
                problem_attempts[unique_pid] = 0

            # Get selectors data
            selectors_data = problem_data.get("selectors", problem_data)

            for selector_name, selector_info in selectors_data.items():
                # Handle different data formats
                if isinstance(selector_info, dict):
                    is_correct = selector_info.get("is_correct", False)
                elif isinstance(selector_info, bool):
                    is_correct = selector_info
                else:
                    continue

                problem_attempts[unique_pid] += 1
                if is_correct:
                    problem_success[unique_pid] += 1

    # Calculate difficulty as 1 - success_rate
    problem_difficulty = {}
    for problem_id in problem_success:
        if problem_attempts[problem_id] > 0:
            success_rate = problem_success[problem_id] / problem_attempts[problem_id]
            problem_difficulty[problem_id] = 1.0 - success_rate
        else:
            problem_difficulty[problem_id] = 0.5  # Neutral if no data

    return problem_difficulty


def calculate_selector_consistency(df_task_details: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consistency metrics for each selector across tasks.

    Args:
        df_task_details: DataFrame with per-task selector performance

    Returns:
        DataFrame with consistency metrics
    """
    consistency_rows = []

    for selector, group in df_task_details.groupby('selector'):
        accuracies = group['accuracy'].values

        # Calculate consistency metrics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        cv = std_acc / mean_acc if mean_acc > 0 else 0  # Coefficient of variation

        # Calculate IQR-based consistency
        q25 = np.percentile(accuracies, 25)
        q75 = np.percentile(accuracies, 75)
        iqr = q75 - q25
        iqr_consistency = 1.0 - (iqr / mean_acc) if mean_acc > 0 else 0

        consistency_rows.append({
            'selector': selector,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'cv': cv,
            'consistency_score': 1.0 - cv,  # Higher is more consistent
            'iqr_consistency': iqr_consistency,
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'range': np.max(accuracies) - np.min(accuracies)
        })

    return pd.DataFrame(consistency_rows)


def calculate_complementarity_matrix(per_problem_data: Dict[str, Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """
    Calculate complementarity matrix showing how often selector pairs solve different problems.

    Args:
        per_problem_data: Dict with structure:
            task_id -> problem_id -> {"selectors": {selector_name: {"is_correct": bool, ...}}}

    Returns:
        DataFrame with pairwise complementarity scores
    """
    # Aggregate all problems across tasks
    all_problems = {}
    for task_id, task_problems in per_problem_data.items():
        for problem_id, problem_data in task_problems.items():
            # Skip metadata entries
            if problem_id in ["problem_id", "selectors"]:
                continue

            full_pid = f"{task_id}::{problem_id}"

            # Extract selector results - handle nested structure
            if isinstance(problem_data, dict) and "selectors" in problem_data:
                selectors_data = problem_data["selectors"]
            else:
                selectors_data = problem_data if isinstance(problem_data, dict) else {}

            # Extract is_correct values for each selector
            selector_results = {}
            for sel_name, sel_info in selectors_data.items():
                if isinstance(sel_info, dict):
                    selector_results[sel_name] = sel_info.get("is_correct", False)
                elif isinstance(sel_info, bool):
                    selector_results[sel_name] = sel_info

            if selector_results:
                all_problems[full_pid] = selector_results

    # Get all selector names
    all_selectors = set()
    for selectors in all_problems.values():
        all_selectors.update(selectors.keys())
    all_selectors = sorted(all_selectors)

    # Build complementarity matrix
    n_selectors = len(all_selectors)
    comp_matrix = np.zeros((n_selectors, n_selectors))

    for i, sel1 in enumerate(all_selectors):
        for j, sel2 in enumerate(all_selectors):
            if i == j:
                comp_matrix[i, j] = 0.0
                continue

            # Count problems where selectors differ
            unique_solves = 0
            shared_solves = 0
            total_problems = 0

            for problem_id, selectors in all_problems.items():
                if sel1 in selectors and sel2 in selectors:
                    total_problems += 1
                    s1_correct = selectors[sel1]
                    s2_correct = selectors[sel2]

                    if s1_correct and not s2_correct:
                        unique_solves += 1
                    elif not s1_correct and s2_correct:
                        unique_solves += 1
                    elif s1_correct and s2_correct:
                        shared_solves += 1

            # Complementarity score: fraction of problems with different outcomes
            if total_problems > 0:
                comp_matrix[i, j] = unique_solves / total_problems
            else:
                comp_matrix[i, j] = 0.0

    # Convert to DataFrame
    df_comp = pd.DataFrame(comp_matrix, index=all_selectors, columns=all_selectors)
    return df_comp


def calculate_dataset_specialization(df_task_details: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate how specialized each selector is for different dataset types.

    Args:
        df_task_details: DataFrame with per-task selector performance

    Returns:
        DataFrame with dataset specialization scores
    """
    spec_rows = []

    # Extract dataset type from task_id
    df_task_details = df_task_details.copy()
    df_task_details['dataset'] = df_task_details['task_id'].apply(
        lambda x: x.split('/')[-1] if '/' in x else x
    )

    for selector, group in df_task_details.groupby('selector'):
        dataset_accs = {}
        for dataset, dgroup in group.groupby('dataset'):
            dataset_accs[dataset] = dgroup['accuracy'].mean()

        if len(dataset_accs) > 1:
            # Calculate specialization as std dev of dataset accuracies
            accs = list(dataset_accs.values())
            specialization = np.std(accs)

            # Find best and worst datasets
            best_dataset = max(dataset_accs, key=dataset_accs.get)
            worst_dataset = min(dataset_accs, key=dataset_accs.get)

            spec_rows.append({
                'selector': selector,
                'specialization_score': specialization,
                'best_dataset': best_dataset,
                'best_dataset_acc': dataset_accs[best_dataset],
                'worst_dataset': worst_dataset,
                'worst_dataset_acc': dataset_accs[worst_dataset],
                'dataset_range': dataset_accs[best_dataset] - dataset_accs[worst_dataset]
            })

    return pd.DataFrame(spec_rows) if spec_rows else pd.DataFrame()


def load_summary_data(summary_path: Path) -> pd.DataFrame:
    """
    Load data from results_all_models/summary.json format.
    """
    with open(summary_path) as f:
        summary = json.load(f)

    rows = []

    for model_name, model_data in summary.get("models", {}).items():
        for dataset_name, dataset_data in model_data.items():
            if dataset_data.get("status") != "success":
                continue

            # Get cache path
            cache_name = dataset_data.get("cache", "")

            # Get selectors
            full_seq = dataset_data.get("full_sequence", {})

            # Create task_id
            task_id = f"{model_name}/{dataset_name}"

            # We need to load the actual n from meta.json
            # For now, use placeholder - this should be loaded from actual cache
            n = 100  # This should be loaded from meta.json!

            for selector, accuracy in full_seq.items():
                rows.append({
                    "task_id": task_id,
                    "selector": selector,
                    "accuracy": accuracy / 100.0 if accuracy > 1.5 else accuracy,
                    "n": n,  # Should be loaded from meta.json
                    "model": model_name,
                    "dataset": dataset_name,
                    "cache": cache_name
                })

    return pd.DataFrame(rows)

def format_ranking_table(df_scores: pd.DataFrame, show_ci: bool = False) -> str:
    """
    Format ranking table for terminal output with proper alignment.
    """
    lines = []

    # Header with fixed widths
    if show_ci:
        header = (f"{'Rank':<6} {'Tier':<6} {'Selector':<18} {'Score':<9} "
                 f"{'95% CI':<22} {'Tasks':<7} {'micro_acc':<11} "
                 f"{'RNS_w':<9} {'regret_w':<11} {'copeland':<10}")
    else:
        header = (f"{'Rank':<6} {'Tier':<6} {'Selector':<18} {'Score':<9} "
                 f"{'Tasks':<7} {'micro_acc':<11} "
                 f"{'RNS_w':<9} {'regret_w':<11} {'copeland':<10}")

    lines.append(header)
    lines.append("─" * len(header))

    # Data rows with consistent formatting
    for _, row in df_scores.iterrows():
        line_parts = [
            f"{row['rank']:<6d}",
            f"{row.get('tier', 'MAIN'):<6}",
            f"{row['selector']:<18}",
            f"{row['Score']:<9.3f}"
        ]

        if show_ci and 'Score_ci_lower' in row:
            ci_str = f"[{row['Score_ci_lower']:.3f}, {row['Score_ci_upper']:.3f}]"
            line_parts.append(f"{ci_str:<22}")

        line_parts.extend([
            f"{row.get('tasks_count', 0):<7d}",
            f"{row['micro_accuracy']:<11.3f}",
            f"{row['RNS_w']:<9.3f}",
            f"{row['regret_w']:<11.3f}",
            f"{row['copeland_norm']:<10.3f}"
        ])

        lines.append(" ".join(line_parts))

    return "\n".join(lines)
