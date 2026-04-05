# NAD_Next runtime notes (2026-04-03)

## Environment constraints
- This environment often blocks even read-only sandboxed shell commands with:
  `bwrap: No permissions to create a new namespace, likely because the kernel does not allow non-privileged user namespaces.`
- When a command matters, prefer running it directly with escalated permissions instead of first retrying inside the sandbox.
- `apply_patch` may also fail for the same namespace reason here; writing files via an escalated shell/Python command is a reliable fallback.

## CPU / parallelism
- Machine capacity seen on 2026-04-03: `240` logical CPUs (`120` physical cores, `2` threads/core).
- NAD_Next repo guidance says to keep `THREADS × PARALLEL_JOBS <= 16` on this machine.
- For cache_test Best-of-N export, a conservative and safe default is shard-level parallelism with `4` workers, each handling disjoint cache keys and writing separate shard outputs before merge validation.

## NAD_Next cache layout
- Historical training/eval used `MUI_HUB/cache/...`.
- New directories visible in this environment:
  - `/home/jovyan/public-ro/MUI_HUB/cache_test`
  - `/home/jovyan/public-ro/MUI_HUB/cache_train`
- For Best-of-N submissions, `sample_id` should be the global index from `meta.json` (`enumerate(meta["samples"])`), not `run_index`.

## Safer Best-of-N export workflow
- Use `scripts/export_bestofn_submissions.py` for single-process export.
- Use `scripts/export_bestofn_submissions_parallel.py` for safer multi-core execution:
  - split by disjoint `cache_key`
  - write shard-local outputs + logs
  - merge only after validating no missing/duplicate cache keys
- Current test-side decision on 2026-04-03: use `reflection_threshold=0.30` because the existing Extreme8 models were trained with that threshold.
