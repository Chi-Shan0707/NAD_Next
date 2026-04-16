#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import validate_earlystop_payload, write_earlystop_payload  # noqa: E402


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _copy_override_scores(
    *,
    base_payload: dict[str, Any],
    override_payload: dict[str, Any],
    cache_keys: tuple[str, ...],
) -> int:
    base_scores = base_payload.get("scores")
    override_scores = override_payload.get("scores")
    if not isinstance(base_scores, dict) or not isinstance(override_scores, dict):
        raise ValueError("Both payloads must contain a dict-valued 'scores' field")

    replaced = 0
    for cache_key in cache_keys:
        if cache_key not in base_scores:
            raise KeyError(f"Base payload missing cache key: {cache_key}")
        if cache_key not in override_scores:
            raise KeyError(f"Override payload missing cache key: {cache_key}")
        base_scores[cache_key] = override_scores[cache_key]
        replaced += 1
    return replaced


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch one EarlyStop submission with cache-level scores from another")
    ap.add_argument("--base-json", required=True, help="Base EarlyStop JSON to patch")
    ap.add_argument("--override-json", required=True, help="Override EarlyStop JSON providing replacement cache keys")
    ap.add_argument("--override-cache-keys", required=True, help="Comma-separated cache keys to replace")
    ap.add_argument("--method-name", default="", help="Optional method_name for patched output")
    ap.add_argument("--out", required=True, help="Output EarlyStop JSON path")
    ap.add_argument("--manifest-out", default="", help="Optional manifest JSON path")
    args = ap.parse_args()

    base_path = Path(args.base_json)
    override_path = Path(args.override_json)
    out_path = Path(args.out)
    cache_keys = _parse_csv(args.override_cache_keys)
    if not cache_keys:
        raise ValueError("--override-cache-keys must contain at least one cache key")

    base_payload = _load_json(base_path)
    override_payload = _load_json(override_path)
    validate_earlystop_payload(base_payload)
    validate_earlystop_payload(override_payload)

    replaced = _copy_override_scores(
        base_payload=base_payload,
        override_payload=override_payload,
        cache_keys=cache_keys,
    )
    if str(args.method_name).strip():
        base_payload["method_name"] = str(args.method_name).strip()

    summary = validate_earlystop_payload(base_payload)
    write_earlystop_payload(base_payload, out_path)

    if str(args.manifest_out).strip():
        manifest_path = Path(args.manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated_at_utc": _now_utc(),
            "method_name": str(base_payload.get("method_name", "")).strip(),
            "submission_path": str(out_path.resolve()),
            "base_submission": str(base_path.resolve()),
            "override_submission": str(override_path.resolve()),
            "override_cache_keys": list(cache_keys),
            "replaced_caches": int(replaced),
            "validation": dict(summary),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"manifest        : {manifest_path}")

    print(f"base_json       : {base_path}")
    print(f"override_json   : {override_path}")
    print(f"override_keys   : {list(cache_keys)}")
    print(f"replaced_caches : {replaced}")
    print(f"method_name     : {base_payload.get('method_name')}")
    print(f"written         : {out_path}")
    print(f"validation      : {summary}")


if __name__ == "__main__":
    main()
