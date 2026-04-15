from __future__ import annotations

import ast
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

FULL_REPORT_NAME = "evaluation_report.json"
COMPACT_REPORT_NAME = "evaluation_report_compact.json"

EXECUTION_FIELD_NAMES = (
    "passed_tests",
    "passed",
    "stderr",
    "stdout",
    "status",
    "error",
    "compile_error",
    "runtime_error",
    "test_output",
)

STATIC_FEATURE_NAMES = [
    "finish_reason_length",
    "output_tokens",
    "input_tokens",
    "code_chars",
    "code_lines",
    "generated_chars",
    "reasoning_code_ratio",
    "syntax_ok",
    "ast_node_count",
    "ast_depth",
    "n_functions",
    "n_classes",
    "n_loops",
    "n_conditionals",
    "n_imports",
    "n_returns",
    "n_try",
    "has_main_guard",
    "has_main_def",
    "uses_stdin",
    "uses_input",
    "uses_sys",
    "has_pass_stmt",
    "has_todo_marker",
    "recovered_code_nonempty",
    "code_source_extracted",
    "code_source_fence",
    "code_source_post_think",
    "code_source_generated_tail",
]

_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_UNCLOSED_FENCE_RE = re.compile(r"```(?:python)?\s*(.*)$", flags=re.IGNORECASE | re.DOTALL)
_CODE_LINE_RE = re.compile(
    r"^\s*(def |class |from |import |if __name__|for |while |print\(|return |[A-Za-z_][A-Za-z0-9_]*\s*=)"
)


def _load_meta(cache_root: str | Path) -> dict[str, Any]:
    cache_path = Path(cache_root)
    meta_path = cache_path / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found in {cache_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def choose_report_path(cache_root: str | Path, *, prefer_full: bool = True) -> tuple[Path, str]:
    cache_path = Path(cache_root)
    full_path = cache_path / FULL_REPORT_NAME
    compact_path = cache_path / COMPACT_REPORT_NAME
    if prefer_full and full_path.is_file():
        return full_path, "full"
    if compact_path.is_file():
        return compact_path, "compact"
    if full_path.is_file():
        return full_path, "full"
    raise FileNotFoundError(
        f"{FULL_REPORT_NAME} / {COMPACT_REPORT_NAME} not found in {cache_path}"
    )


def recover_code_text(extracted_answer: str | None, generated_text: str | None) -> tuple[str, str]:
    extracted = (extracted_answer or "").strip()
    generated = generated_text or ""
    if extracted:
        return extracted, "extracted"

    fenced = _FENCE_RE.findall(generated)
    if fenced:
        block = max((block.strip() for block in fenced), key=len, default="")
        if block:
            return block, "fence"

    unclosed = _UNCLOSED_FENCE_RE.search(generated)
    if unclosed:
        block = (unclosed.group(1) or "").strip()
        if block:
            return block, "fence"

    lower = generated.lower()
    if "</think>" in lower:
        tail = generated[lower.rfind("</think>") + len("</think>"):].strip()
        if tail:
            return tail, "post_think"

    lines = generated.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if _CODE_LINE_RE.match(line.strip()):
            start_idx = idx
            break
    if start_idx is not None:
        tail = "\n".join(lines[start_idx:]).strip()
        if tail:
            return tail, "generated_tail"

    return "", "empty"


def load_coding_report_records(
    cache_root: str | Path,
    *,
    prefer_full: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = _load_meta(cache_root)
    report_path, report_kind = choose_report_path(cache_root, prefer_full=prefer_full)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    sample_records: list[dict[str, Any]] = []
    for sample_id, sample in enumerate(meta.get("samples", [])):
        sample_records.append(
            {
                "sample_id": int(sample_id),
                "problem_id": str(sample.get("problem_id")),
                "run_index": int(sample.get("run_index", 0)),
                "prompt": "",
                "actual_prompt": "",
                "generated_text": "",
                "extracted_answer": "",
                "is_correct": None,
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "stop_reason": None,
                "finish_reason": None,
                "dataset": None,
                "difficulty": None,
                "language": None,
            }
        )

    sample_index = {
        (str(sample["problem_id"]), int(sample.get("run_index", 0))): int(sample_id)
        for sample_id, sample in enumerate(meta.get("samples", []))
    }

    for result in report.get("results", []):
        problem_id = str(result.get("problem_id"))
        prompt = result.get("prompt", "") or ""
        dataset = result.get("dataset")
        difficulty = result.get("difficulty")
        language = result.get("language") or result.get("lang")
        for run in result.get("runs", []):
            key = (problem_id, int(run.get("run_index", run.get("index", 0))))
            sample_id = sample_index.get(key)
            if sample_id is None:
                continue
            sample_records[sample_id].update(
                {
                    "prompt": prompt,
                    "actual_prompt": run.get("actual_prompt", "") or "",
                    "generated_text": run.get("generated_text", "") or "",
                    "extracted_answer": run.get("extracted_answer", "") or "",
                    "is_correct": bool(run.get("is_correct", False)),
                    "input_tokens": float(run.get("input_tokens", 0.0) or 0.0),
                    "output_tokens": float(run.get("output_tokens", 0.0) or 0.0),
                    "stop_reason": run.get("stop_reason"),
                    "finish_reason": run.get("finish_reason"),
                    "dataset": dataset,
                    "difficulty": difficulty,
                    "language": language,
                }
            )
            for field_name in EXECUTION_FIELD_NAMES:
                if field_name in run:
                    sample_records[sample_id][field_name] = run.get(field_name)

    info = {
        "report_path": str(report_path),
        "report_kind": report_kind,
        "n_samples": int(len(sample_records)),
    }
    return sample_records, info


def audit_coding_inputs(cache_root: str | Path, *, prefer_full: bool = True) -> dict[str, Any]:
    cache_path = Path(cache_root)
    records, info = load_coding_report_records(cache_path, prefer_full=prefer_full)

    prompt_count = sum(1 for row in records if str(row.get("prompt", "")).strip())
    actual_prompt_count = sum(1 for row in records if str(row.get("actual_prompt", "")).strip())
    generated_text_count = sum(1 for row in records if str(row.get("generated_text", "")).strip())
    extracted_answer_count = sum(1 for row in records if str(row.get("extracted_answer", "")).strip())
    recovered_rows = [recover_code_text(row.get("extracted_answer"), row.get("generated_text")) for row in records]
    recovered_code_count = sum(1 for code, _ in recovered_rows if code.strip())
    execution_counts = {
        field_name: sum(1 for row in records if row.get(field_name) not in (None, "", [], {}))
        for field_name in EXECUTION_FIELD_NAMES
    }
    has_execution_fields = any(count > 0 for count in execution_counts.values())

    if recovered_code_count > 0 and prompt_count > 0 and has_execution_fields:
        sufficiency_tier = "Tier A"
        sufficiency_reason = "Code text, prompt text, and execution-style fields are locally recoverable."
    elif recovered_code_count > 0 and prompt_count > 0:
        sufficiency_tier = "Tier B"
        sufficiency_reason = "Code text and prompt text are locally recoverable; execution-style fields are absent."
    else:
        sufficiency_tier = "Tier C"
        sufficiency_reason = "Only cache-native numeric traces are locally recoverable."

    return {
        "cache_root": str(cache_path),
        "has_local_full_report": bool((cache_path / FULL_REPORT_NAME).is_file()),
        "has_local_compact_report": bool((cache_path / COMPACT_REPORT_NAME).is_file()),
        "selected_report_path": info["report_path"],
        "selected_report_kind": info["report_kind"],
        "n_samples": int(len(records)),
        "prompt_nonempty_count": int(prompt_count),
        "actual_prompt_nonempty_count": int(actual_prompt_count),
        "generated_text_nonempty_count": int(generated_text_count),
        "extracted_answer_nonempty_count": int(extracted_answer_count),
        "recovered_code_nonempty_count": int(recovered_code_count),
        "execution_field_counts": execution_counts,
        "sufficiency_tier": sufficiency_tier,
        "sufficiency_reason": sufficiency_reason,
    }


def _ast_summary(code_text: str) -> dict[str, float]:
    code = str(code_text or "")
    if not code.strip():
        return {
            "syntax_ok": 0.0,
            "ast_node_count": 0.0,
            "ast_depth": 0.0,
            "n_functions": 0.0,
            "n_classes": 0.0,
            "n_loops": 0.0,
            "n_conditionals": 0.0,
            "n_imports": 0.0,
            "n_returns": 0.0,
            "n_try": 0.0,
            "has_main_guard": 0.0,
            "has_main_def": 0.0,
            "uses_stdin": float("stdin" in code),
            "uses_input": float("input(" in code),
            "uses_sys": float("sys" in code),
            "has_pass_stmt": float("pass" in code),
            "has_todo_marker": float("todo" in code.lower()),
        }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "syntax_ok": 0.0,
            "ast_node_count": 0.0,
            "ast_depth": 0.0,
            "n_functions": 0.0,
            "n_classes": 0.0,
            "n_loops": 0.0,
            "n_conditionals": 0.0,
            "n_imports": 0.0,
            "n_returns": 0.0,
            "n_try": 0.0,
            "has_main_guard": 0.0,
            "has_main_def": 0.0,
            "uses_stdin": float("stdin" in code),
            "uses_input": float("input(" in code),
            "uses_sys": float("sys" in code),
            "has_pass_stmt": float("pass" in code),
            "has_todo_marker": float("todo" in code.lower()),
        }

    depth = 0
    stack = [(tree, 1)]
    while stack:
        node, d = stack.pop()
        depth = max(depth, d)
        for child in ast.iter_child_nodes(node):
            stack.append((child, d + 1))

    n_functions = 0
    n_classes = 0
    n_loops = 0
    n_conditionals = 0
    n_imports = 0
    n_returns = 0
    n_try = 0
    has_main_guard = 0
    has_main_def = 0
    node_count = 0

    for node in ast.walk(tree):
        node_count += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_functions += 1
            if getattr(node, "name", None) == "main":
                has_main_def = 1
        elif isinstance(node, ast.ClassDef):
            n_classes += 1
        elif isinstance(node, (ast.For, ast.While)):
            n_loops += 1
        elif isinstance(node, ast.If):
            n_conditionals += 1
            try:
                test_str = ast.unparse(node.test)
            except Exception:
                test_str = ""
            if "__name__" in test_str and "__main__" in test_str:
                has_main_guard = 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            n_imports += 1
        elif isinstance(node, ast.Return):
            n_returns += 1
        elif isinstance(node, ast.Try):
            n_try += 1

    return {
        "syntax_ok": 1.0,
        "ast_node_count": float(node_count),
        "ast_depth": float(depth),
        "n_functions": float(n_functions),
        "n_classes": float(n_classes),
        "n_loops": float(n_loops),
        "n_conditionals": float(n_conditionals),
        "n_imports": float(n_imports),
        "n_returns": float(n_returns),
        "n_try": float(n_try),
        "has_main_guard": float(has_main_guard),
        "has_main_def": float(has_main_def),
        "uses_stdin": float("stdin" in code),
        "uses_input": float("input(" in code),
        "uses_sys": float("sys" in code),
        "has_pass_stmt": float("pass" in code),
        "has_todo_marker": float("todo" in code.lower()),
    }


def extract_static_code_features(record: Mapping[str, Any]) -> dict[str, float]:
    code_text, source = recover_code_text(record.get("extracted_answer"), record.get("generated_text"))
    code_lines = [line for line in code_text.splitlines() if line.strip()]
    generated_text = str(record.get("generated_text", "") or "")
    ast_info = _ast_summary(code_text)

    out = {
        "finish_reason_length": float(record.get("finish_reason") == "length"),
        "output_tokens": float(record.get("output_tokens", 0.0) or 0.0),
        "input_tokens": float(record.get("input_tokens", 0.0) or 0.0),
        "code_chars": float(len(code_text)),
        "code_lines": float(len(code_lines)),
        "generated_chars": float(len(generated_text)),
        "reasoning_code_ratio": float(len(generated_text) / max(len(code_text), 1)),
        "recovered_code_nonempty": float(bool(code_text.strip())),
        "code_source_extracted": float(source == "extracted"),
        "code_source_fence": float(source == "fence"),
        "code_source_post_think": float(source == "post_think"),
        "code_source_generated_tail": float(source == "generated_tail"),
    }
    out.update(ast_info)
    return {name: float(out.get(name, 0.0)) for name in STATIC_FEATURE_NAMES}


def build_static_feature_cache(
    cache_root: str | Path,
    *,
    cache_path: str | Path | None = None,
    prefer_full: bool = True,
    refresh: bool = False,
) -> dict[str, Any]:
    cp = None if cache_path is None else Path(cache_path)
    if cp is not None and cp.exists() and not refresh:
        with cp.open("rb") as fh:
            payload = pickle.load(fh)
        return payload

    records, info = load_coding_report_records(cache_root, prefer_full=prefer_full)
    X = np.zeros((len(records), len(STATIC_FEATURE_NAMES)), dtype=np.float64)
    code_sources: list[str] = []
    for row_idx, record in enumerate(records):
        feat = extract_static_code_features(record)
        for col_idx, name in enumerate(STATIC_FEATURE_NAMES):
            X[row_idx, col_idx] = float(feat[name])
        _, source = recover_code_text(record.get("extracted_answer"), record.get("generated_text"))
        code_sources.append(source)

    payload = {
        "sample_ids": np.arange(len(records), dtype=np.int64),
        "problem_ids": np.asarray([str(row["problem_id"]) for row in records], dtype=object),
        "labels": np.asarray([int(bool(row.get("is_correct", False))) for row in records], dtype=np.int32),
        "feature_names": list(STATIC_FEATURE_NAMES),
        "X": np.asarray(X, dtype=np.float64),
        "code_sources": list(code_sources),
        "records": records,
        "report_info": info,
    }
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        with cp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=4)
    return payload
