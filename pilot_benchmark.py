#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


FAMILIES: Tuple[str, ...] = (
    "sequential_baseline",
    "single_branch",
    "simple_loop",
    "loop_plus_branch",
    "phase_switch_loop",
    "nested_loop",
    "function_scope",
)

DIFFICULTIES: Tuple[str, ...] = ("low", "medium", "high")


@dataclass
class TaskBlueprint:
    family: str
    difficulty: str
    code: str
    entry_fn: str
    input_args: Dict[str, Any]
    question: str
    boundary_points: List[str]
    attributes: Dict[str, Any]
    expected_failure_modes: List[str]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    return repr(value)


def code_block(text: str) -> str:
    return textwrap.dedent(text).strip("\n") + "\n"


def make_call_repr(entry_fn: str, input_args: Dict[str, Any]) -> str:
    joined = ", ".join(f"{key}={repr(value)}" for key, value in input_args.items())
    return f"{entry_fn}({joined})"


def difficulty_level(difficulty: str) -> int:
    return DIFFICULTIES.index(difficulty)


def annotate_source(code: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    tree = ast.parse(code)
    branch_entries: Dict[int, Dict[str, Any]] = {}
    loop_entries: Dict[int, Dict[str, Any]] = {}
    function_lines: Dict[int, Dict[str, Any]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_lines[node.lineno] = {"function_name": node.name}

        if isinstance(node, ast.If):
            if node.body:
                branch_entries[node.body[0].lineno] = {
                    "if_line": node.lineno,
                    "branch_taken": "then",
                }
            if node.orelse:
                branch_entries[node.orelse[0].lineno] = {
                    "if_line": node.lineno,
                    "branch_taken": "else",
                }

        if isinstance(node, (ast.For, ast.While)) and node.body:
            loop_entries[node.body[0].lineno] = {
                "loop_line": node.lineno,
                "loop_type": type(node).__name__.lower(),
            }

    return {
        "branch_entries": branch_entries,
        "loop_entries": loop_entries,
        "function_lines": function_lines,
    }


class TraceCollector:
    def __init__(self, filename: str, code: str, family: str) -> None:
        self.filename = filename
        self.source_lines = code.splitlines()
        self.family = family
        self.annotations = annotate_source(code)
        self.events: List[Dict[str, Any]] = []
        self.frame_state: Dict[int, Dict[str, Any]] = {}
        self.call_stack: List[int] = []
        self.next_call_id = 1
        self.loop_counters: Dict[Tuple[int, int], int] = defaultdict(int)

    def snapshot(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        return {key: json_safe(mapping[key]) for key in sorted(mapping)}

    def line_text(self, line_no: int) -> str:
        if 1 <= line_no <= len(self.source_lines):
            return self.source_lines[line_no - 1].rstrip()
        return ""

    def append_event(self, event: Dict[str, Any]) -> None:
        event["step"] = len(self.events)
        self.events.append(event)

    def finalize_open_line(
        self,
        state: Dict[str, Any],
        next_locals: Dict[str, Any],
    ) -> None:
        open_event = state.get("open_line_event")
        if open_event is None:
            return
        open_event["locals_after"] = next_locals
        self.append_event(open_event)
        state["open_line_event"] = None

    def tracer(self, frame: Any, event: str, arg: Any) -> Optional[Callable[..., Any]]:
        if frame.f_code.co_filename != self.filename:
            return None

        frame_id = id(frame)

        if event == "call":
            scope_depth = len(self.call_stack) + 1
            call_id = self.next_call_id
            self.next_call_id += 1
            self.call_stack.append(frame_id)
            state = {
                "call_id": call_id,
                "scope_name": frame.f_code.co_name,
                "scope_depth": scope_depth,
                "open_line_event": None,
            }
            self.frame_state[frame_id] = state
            locals_snapshot = self.snapshot(frame.f_locals)
            self.append_event(
                {
                    "event_type": "call",
                    "line": frame.f_code.co_firstlineno,
                    "line_text": self.line_text(frame.f_code.co_firstlineno),
                    "scope_name": state["scope_name"],
                    "scope_depth": scope_depth,
                    "call_id": call_id,
                    "locals_before": locals_snapshot,
                    "locals_after": locals_snapshot,
                    "branch_taken": None,
                    "branch_line": None,
                    "loop_iter": None,
                    "loop_line": None,
                    "phase": None,
                    "boundary_tags": ["scope_enter"],
                }
            )
            return self.tracer

        state = self.frame_state.get(frame_id)
        if state is None:
            return self.tracer

        if event == "line":
            current_locals = self.snapshot(frame.f_locals)
            self.finalize_open_line(state, current_locals)

            line_no = frame.f_lineno
            branch_meta = self.annotations["branch_entries"].get(line_no)
            loop_meta = self.annotations["loop_entries"].get(line_no)

            loop_iter = None
            loop_line = None
            if loop_meta is not None:
                loop_line = int(loop_meta["loop_line"])
                counter_key = (state["call_id"], loop_line)
                self.loop_counters[counter_key] += 1
                loop_iter = self.loop_counters[counter_key]

            open_line_event = {
                "event_type": "line",
                "line": line_no,
                "line_text": self.line_text(line_no),
                "scope_name": state["scope_name"],
                "scope_depth": state["scope_depth"],
                "call_id": state["call_id"],
                "locals_before": current_locals,
                "locals_after": None,
                "branch_taken": None if branch_meta is None else branch_meta["branch_taken"],
                "branch_line": None if branch_meta is None else int(branch_meta["if_line"]),
                "loop_iter": loop_iter,
                "loop_line": loop_line,
                "phase": None,
                "boundary_tags": [],
            }

            if branch_meta is not None:
                open_line_event["boundary_tags"].append("branch_entry")

            state["open_line_event"] = open_line_event
            return self.tracer

        if event == "return":
            current_locals = self.snapshot(frame.f_locals)
            self.finalize_open_line(state, current_locals)

            self.append_event(
                {
                    "event_type": "return",
                    "line": frame.f_lineno,
                    "line_text": self.line_text(frame.f_lineno),
                    "scope_name": state["scope_name"],
                    "scope_depth": state["scope_depth"],
                    "call_id": state["call_id"],
                    "locals_before": current_locals,
                    "locals_after": current_locals,
                    "return_value": json_safe(arg),
                    "branch_taken": None,
                    "branch_line": None,
                    "loop_iter": None,
                    "loop_line": None,
                    "phase": None,
                    "boundary_tags": ["scope_exit"],
                }
            )

            if self.call_stack and self.call_stack[-1] == frame_id:
                self.call_stack.pop()
            self.frame_state.pop(frame_id, None)
            return self.tracer

        return self.tracer

    def postprocess(self) -> None:
        loop_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        phase_events: List[Dict[str, Any]] = []

        for event in self.events:
            if event["event_type"] == "line" and event["loop_iter"] is not None and event["loop_line"] is not None:
                loop_groups[(event["call_id"], event["loop_line"])].append(event)

            if self.family == "phase_switch_loop" and event.get("branch_taken") in {"then", "else"}:
                event["phase"] = "phase_a" if event["branch_taken"] == "then" else "phase_b"
                phase_events.append(event)

        for grouped in loop_groups.values():
            grouped[0]["boundary_tags"].append("first_iteration")
            grouped[-1]["boundary_tags"].append("last_iteration")

        if phase_events:
            previous_phase: Optional[str] = None
            for index, event in enumerate(phase_events):
                current_phase = event["phase"]
                if previous_phase is not None and current_phase != previous_phase:
                    event["boundary_tags"].append("phase_switch")
                    event["boundary_tags"].append("first_post_switch_iteration")
                    phase_events[index - 1]["boundary_tags"].append("iteration_at_pivot")
                previous_phase = current_phase

        if self.family == "function_scope":
            for index, event in enumerate(self.events):
                if event["event_type"] == "call" and event["scope_name"] != "solve":
                    event["boundary_tags"].append("helper_call_entry")
                if event["event_type"] == "return" and event["scope_name"] != "solve":
                    event["boundary_tags"].append("helper_return")
                    if index + 1 < len(self.events):
                        next_event = self.events[index + 1]
                        if next_event["scope_name"] == "solve" and next_event["event_type"] == "line":
                            next_event["boundary_tags"].append("caller_resume")


def execute_with_trace(task_id: str, blueprint: TaskBlueprint) -> Tuple[Any, List[Dict[str, Any]]]:
    filename = f"<pilot::{task_id}>"
    namespace: Dict[str, Any] = {}
    compiled = compile(blueprint.code, filename, "exec")
    exec(compiled, namespace, namespace)
    entry_fn = namespace[blueprint.entry_fn]

    collector = TraceCollector(filename=filename, code=blueprint.code, family=blueprint.family)
    previous = sys.gettrace()
    sys.settrace(collector.tracer)
    try:
        output = entry_fn(**blueprint.input_args)
    finally:
        sys.settrace(previous)
    collector.postprocess()
    return json_safe(output), collector.events


def build_task_record(
    task_id: str,
    blueprint: TaskBlueprint,
    gold_output: Any,
    gold_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    max_live_vars = 0
    max_scope_depth = 0
    trace_summary = {
        "line_events": 0,
        "branch_events": 0,
        "loop_entry_events": 0,
        "call_events": 0,
        "return_events": 0,
    }

    for event in gold_trace:
        max_scope_depth = max(max_scope_depth, int(event["scope_depth"]))
        if event["event_type"] == "line":
            trace_summary["line_events"] += 1
            max_live_vars = max(max_live_vars, len(event["locals_after"] or {}))
            if event["branch_taken"] is not None:
                trace_summary["branch_events"] += 1
            if event["loop_iter"] is not None:
                trace_summary["loop_entry_events"] += 1
        elif event["event_type"] == "call":
            trace_summary["call_events"] += 1
        elif event["event_type"] == "return":
            trace_summary["return_events"] += 1

    attributes = dict(blueprint.attributes)
    attributes["live_var_count"] = max(attributes.get("live_var_count", 0), max_live_vars)
    attributes["scope_depth"] = max(attributes.get("scope_depth", 0), max_scope_depth)
    attributes["boundary_case_count"] = len(blueprint.boundary_points)

    call_repr = make_call_repr(blueprint.entry_fn, blueprint.input_args)

    return {
        "task_id": task_id,
        "family": blueprint.family,
        "difficulty": blueprint.difficulty,
        "language": "python",
        "entry_fn": blueprint.entry_fn,
        "input": json_safe(blueprint.input_args),
        "entry_call": call_repr,
        "question": blueprint.question,
        "code": blueprint.code,
        "gold_output": gold_output,
        "gold_trace": gold_trace,
        "attributes": attributes,
        "boundary_points": blueprint.boundary_points,
        "expected_failure_modes": blueprint.expected_failure_modes,
        "trace_summary": trace_summary,
    }


def generate_sequential(rng: random.Random, difficulty: str) -> TaskBlueprint:
    level = difficulty_level(difficulty)
    if level == 0:
        x = rng.randint(3, 9)
        add = rng.randint(2, 5)
        scale = rng.randint(2, 4)
        sub = rng.randint(1, 4)
        code = code_block(
            f"""
            def solve(x):
                base = x + {add}
                stretched = base * {scale}
                final = stretched - {sub}
                return final
            """
        )
        input_args = {"x": x}
    elif level == 1:
        x = rng.randint(4, 9)
        y = rng.randint(1, 5)
        add = rng.randint(2, 6)
        scale = rng.randint(2, 4)
        tail = rng.randint(1, 5)
        code = code_block(
            f"""
            def solve(x, y):
                base = x + {add}
                merged = base - y
                scaled = merged * {scale}
                final = scaled + {tail}
                return final
            """
        )
        input_args = {"x": x, "y": y}
    else:
        x = rng.randint(5, 10)
        y = rng.randint(2, 6)
        add = rng.randint(2, 6)
        drift = rng.randint(1, 4)
        scale = rng.randint(2, 5)
        tail = rng.randint(2, 6)
        code = code_block(
            f"""
            def solve(x, y):
                base = x + {add}
                guard = y - {drift}
                merged = base * {scale}
                folded = merged - guard
                final = folded + {tail}
                return final
            """
        )
        input_args = {"x": x, "y": y}

    return TaskBlueprint(
        family="sequential_baseline",
        difficulty=difficulty,
        code=code,
        entry_fn="solve",
        input_args=input_args,
        question=f"Predict the exact return value of `{make_call_repr('solve', input_args)}`.",
        boundary_points=[],
        attributes={
            "branch_depth": 0,
            "loop_nesting": 0,
            "live_var_count": 0,
            "phase_switch_count": 0,
            "boundary_case_count": 0,
            "scope_depth": 1,
            "invariant_needed": False,
        },
        expected_failure_modes=["wrong variable update", "hallucinated state"],
    )


def generate_single_branch(rng: random.Random, difficulty: str) -> TaskBlueprint:
    level = difficulty_level(difficulty)
    x = rng.randint(5, 12)
    y = rng.randint(1, 6)
    bonus = rng.randint(2, 5)
    penalty = rng.randint(1, 4)
    pivot = 1 if level == 0 else 2
    if level < 2:
        code = code_block(
            f"""
            def solve(x, y):
                total = x - y
                if total % 3 == {pivot}:
                    result = total + {bonus}
                else:
                    result = total - {penalty}
                return result
            """
        )
    else:
        shift = rng.randint(1, 3)
        code = code_block(
            f"""
            def solve(x, y):
                total = x - y
                shifted = total + {shift}
                if shifted % 4 == {pivot}:
                    result = shifted + {bonus}
                else:
                    result = shifted - {penalty}
                return result
            """
        )

    input_args = {"x": x, "y": y}
    return TaskBlueprint(
        family="single_branch",
        difficulty=difficulty,
        code=code,
        entry_fn="solve",
        input_args=input_args,
        question=f"Predict the exact return value of `{make_call_repr('solve', input_args)}`.",
        boundary_points=["branch_condition"],
        attributes={
            "branch_depth": 1,
            "loop_nesting": 0,
            "live_var_count": 0,
            "phase_switch_count": 0,
            "boundary_case_count": 1,
            "scope_depth": 1,
            "invariant_needed": False,
        },
        expected_failure_modes=["wrong branch selected", "wrong variable update"],
    )


def generate_simple_loop(rng: random.Random, difficulty: str) -> TaskBlueprint:
    level = difficulty_level(difficulty)
    n = rng.randint(3 + level, 5 + level)
    start = rng.randint(0, 3)
    scale = rng.randint(1, 3 + level)
    if level < 2:
        code = code_block(
            f"""
            def solve(n):
                acc = {start}
                for i in range(1, n + 1):
                    acc = acc + i * {scale}
                return acc
            """
        )
    else:
        tail = rng.randint(1, 4)
        code = code_block(
            f"""
            def solve(n):
                acc = {start}
                for i in range(1, n + 1):
                    term = i * {scale}
                    acc = acc + term
                return acc + {tail}
            """
        )

    input_args = {"n": n}
    return TaskBlueprint(
        family="simple_loop",
        difficulty=difficulty,
        code=code,
        entry_fn="solve",
        input_args=input_args,
        question=f"Predict the exact return value of `{make_call_repr('solve', input_args)}`.",
        boundary_points=["first_iteration", "last_iteration"],
        attributes={
            "branch_depth": 0,
            "loop_nesting": 1,
            "live_var_count": 0,
            "phase_switch_count": 0,
            "boundary_case_count": 2,
            "scope_depth": 1,
            "invariant_needed": True,
        },
        expected_failure_modes=["loop off-by-one", "wrong variable update", "invariant drift"],
    )


def generate_loop_plus_branch(rng: random.Random, difficulty: str) -> TaskBlueprint:
    level = difficulty_level(difficulty)
    n = rng.randint(4 + level, 6 + level)
    even_bonus = rng.randint(1, 4)
    odd_penalty = rng.randint(1, 3)
    if level < 2:
        code = code_block(
            f"""
            def solve(n):
                total = 0
                for i in range(1, n + 1):
                    if i % 2 == 0:
                        total = total + i + {even_bonus}
                    else:
                        total = total - {odd_penalty}
                return total
            """
        )
    else:
        tail = rng.randint(1, 3)
        code = code_block(
            f"""
            def solve(n):
                total = 0
                for i in range(1, n + 1):
                    if i % 2 == 0:
                        total = total + i + {even_bonus}
                    else:
                        total = total - {odd_penalty}
                return total + {tail}
            """
        )

    input_args = {"n": n}
    return TaskBlueprint(
        family="loop_plus_branch",
        difficulty=difficulty,
        code=code,
        entry_fn="solve",
        input_args=input_args,
        question=f"Predict the exact return value of `{make_call_repr('solve', input_args)}`.",
        boundary_points=["first_iteration", "last_iteration", "branch_alternation"],
        attributes={
            "branch_depth": 1,
            "loop_nesting": 1,
            "live_var_count": 0,
            "phase_switch_count": 0,
            "boundary_case_count": 3,
            "scope_depth": 1,
            "invariant_needed": True,
        },
        expected_failure_modes=["wrong branch selected", "loop off-by-one", "invariant drift"],
    )


def generate_phase_switch_loop(rng: random.Random, difficulty: str) -> TaskBlueprint:
    level = difficulty_level(difficulty)
    n = rng.randint(5 + level, 7 + level)
    pivot = max(2, n // 2)
    add_scale = rng.randint(1, 3)
    suffix_bias = rng.randint(2, 5)
    code = code_block(
        f"""
        def solve(n):
            pivot = {pivot}
            total = 0
            for i in range(1, n + 1):
                if i <= pivot:
                    total = total + i * {add_scale}
                else:
                    total = total + ({suffix_bias} - i)
            return total
        """
    )
    input_args = {"n": n}
    return TaskBlueprint(
        family="phase_switch_loop",
        difficulty=difficulty,
        code=code,
        entry_fn="solve",
        input_args=input_args,
        question=f"Predict the exact return value of `{make_call_repr('solve', input_args)}`.",
        boundary_points=[
            "first_iteration",
            "iteration_at_pivot",
            "first_post_switch_iteration",
            "last_iteration",
        ],
        attributes={
            "branch_depth": 1,
            "loop_nesting": 1,
            "live_var_count": 0,
            "phase_switch_count": 1,
            "boundary_case_count": 4,
            "scope_depth": 1,
            "invariant_needed": True,
        },
        expected_failure_modes=[
            "boundary transition failure",
            "wrong branch selected",
            "loop off-by-one",
            "wrong variable update",
        ],
    )


def generate_nested_loop(rng: random.Random, difficulty: str) -> TaskBlueprint:
    level = difficulty_level(difficulty)
    rows = rng.randint(2 + level, 3 + level)
    cols = rng.randint(2, 4 + level)
    inner_bias = rng.randint(0, 2)
    code = code_block(
        f"""
        def solve(rows, cols):
            total = 0
            for i in range(1, rows + 1):
                inner = {inner_bias}
                for j in range(1, cols + 1):
                    inner = inner + i + j
                total = total + inner - i
            return total
        """
    )
    input_args = {"rows": rows, "cols": cols}
    return TaskBlueprint(
        family="nested_loop",
        difficulty=difficulty,
        code=code,
        entry_fn="solve",
        input_args=input_args,
        question=f"Predict the exact return value of `{make_call_repr('solve', input_args)}`.",
        boundary_points=[
            "outer_first_iteration",
            "outer_last_iteration",
            "inner_first_iteration",
            "inner_last_iteration",
        ],
        attributes={
            "branch_depth": 0,
            "loop_nesting": 2,
            "live_var_count": 0,
            "phase_switch_count": 0,
            "boundary_case_count": 4,
            "scope_depth": 1,
            "invariant_needed": True,
        },
        expected_failure_modes=["loop off-by-one", "wrong variable update", "invariant drift"],
    )


def generate_function_scope(rng: random.Random, difficulty: str) -> TaskBlueprint:
    level = difficulty_level(difficulty)
    x = rng.randint(3, 8)
    y = rng.randint(1, 5)
    scale = rng.randint(2, 4)
    shift = rng.randint(1, 4)
    if level < 2:
        code = code_block(
            f"""
            def helper(seed, delta):
                local = seed * {scale}
                local = local - delta
                return local + {shift}

            def solve(x, y):
                outer = x + y
                inner = helper(outer, y)
                result = inner - x
                return result
            """
        )
    else:
        tail = rng.randint(1, 3)
        code = code_block(
            f"""
            def helper(seed, delta):
                local = seed * {scale}
                local = local - delta
                return local + {shift}

            def solve(x, y):
                outer = x + y
                inner = helper(outer, y)
                result = inner - x
                return result + {tail}
            """
        )

    input_args = {"x": x, "y": y}
    return TaskBlueprint(
        family="function_scope",
        difficulty=difficulty,
        code=code,
        entry_fn="solve",
        input_args=input_args,
        question=f"Predict the exact return value of `{make_call_repr('solve', input_args)}`.",
        boundary_points=["helper_call_entry", "helper_return", "caller_resume"],
        attributes={
            "branch_depth": 0,
            "loop_nesting": 0,
            "live_var_count": 0,
            "phase_switch_count": 0,
            "boundary_case_count": 3,
            "scope_depth": 2,
            "invariant_needed": False,
        },
        expected_failure_modes=["scope confusion", "wrong variable update", "hallucinated state"],
    )


GENERATOR_MAP: Dict[str, Callable[[random.Random, str], TaskBlueprint]] = {
    "sequential_baseline": generate_sequential,
    "single_branch": generate_single_branch,
    "simple_loop": generate_simple_loop,
    "loop_plus_branch": generate_loop_plus_branch,
    "phase_switch_loop": generate_phase_switch_loop,
    "nested_loop": generate_nested_loop,
    "function_scope": generate_function_scope,
}


def generate_tasks(
    families: Iterable[str],
    num_per_family: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []

    for family in families:
        generator = GENERATOR_MAP[family]
        for index in range(num_per_family):
            difficulty = DIFFICULTIES[index % len(DIFFICULTIES)]
            task_seed = rng.randint(0, 10**9)
            local_rng = random.Random(task_seed)
            blueprint = generator(local_rng, difficulty)
            task_id = f"{family}-{difficulty}-{index:03d}"
            gold_output, gold_trace = execute_with_trace(task_id, blueprint)
            record = build_task_record(task_id, blueprint, gold_output, gold_trace)
            records.append(record)
    return records


def write_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def pretty_print_samples(records: List[Dict[str, Any]], limit: int) -> None:
    for record in records[:limit]:
        preview = {
            "task_id": record["task_id"],
            "family": record["family"],
            "difficulty": record["difficulty"],
            "entry_call": record["entry_call"],
            "gold_output": record["gold_output"],
            "trace_events": len(record["gold_trace"]),
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))


def parse_families(raw: str) -> List[str]:
    if not raw.strip():
        return list(FAMILIES)
    families = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in families if item not in GENERATOR_MAP]
    if invalid:
        raise ValueError(f"Unknown families: {', '.join(invalid)}")
    return families


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic code-reasoning pilot benchmark.")
    parser.add_argument("--num-per-family", type=int, default=3, help="Number of tasks to generate per family.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for task generation.")
    parser.add_argument("--out", type=Path, default=Path("sample_tasks.jsonl"), help="Output JSONL path.")
    parser.add_argument(
        "--families",
        type=str,
        default=",".join(FAMILIES),
        help="Comma-separated list of families to generate.",
    )
    parser.add_argument(
        "--pretty-sample",
        type=int,
        default=0,
        help="Print the first K generated tasks as a compact preview.",
    )
    args = parser.parse_args()

    families = parse_families(args.families)
    records = generate_tasks(families=families, num_per_family=args.num_per_family, seed=args.seed)
    write_jsonl(records, args.out)

    if args.pretty_sample > 0:
        pretty_print_samples(records, args.pretty_sample)

    print(f"Wrote {len(records)} tasks to {args.out}")


if __name__ == "__main__":
    main()
