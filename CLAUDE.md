# pivotiq (futsal moment-intelligence)

Same codebase lineage as `IAFS` (package distribution `iafs`): legacy
`src/futsal_analytics/` plus `src/pivotiq/`. Remote is `isaacboque/pivotiq`; the
working branch is `training` (not `main`) — check `git branch --show-current`.
Python 3.10+; CI (`.github/workflows/tests.yml`) runs ruff + pytest on ubuntu/windows, py3.10–3.12.

## Commands

- `pip install -e ".[dev,viewer]"`
- `pytest tests/ -v --tb=short` (`pytest` puts this checkout's `src/` first via `pythonpath`,
  so it never tests another editable install such as IAFS's)
- `ruff check src tests web training` (line-length 110; `ruff format .` before committing)
- `pivotiq analyze <video.mp4>` — Milestone 1 skeleton: parses flags and prints the run
  config, **does not process video yet** (detect/track/register/events/report are stubs)

## Notes

- Read IAFS/CLAUDE.md for the torch/CUDA, `supervision<0.30` and model-weights gotchas — they apply here too.
- `out/`, `out_smoke/`, `runs/` are outputs; don't commit them or large weights.
