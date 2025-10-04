# Repository Guidelines

## Project Structure & Module Organization
- `tabularaml/generate/`: Core feature generation (`features.py`, `ops.py`).
- `tabularaml/preprocessing/`: Pipelines, encoders, imputers.
- `tabularaml/eval/`: Cross‑validation and scorers.
- `tabularaml/inspect/`: Feature importance and metadata utilities.
- `tabularaml/configs/`: Presets and search spaces.
- `tabularaml/ui/`: Flask + SocketIO UI (`feature_generator_server.py`).
- `tabularaml/utils/`: Datasets, GPU helpers, ID maps.
- Root: example scripts/notebooks and sample CSVs. Use `cache/` for outputs.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate` (Windows: `Scripts\\activate`).
- Install deps (project + UI): `pip install -e .` if packaged; otherwise `pip install flask flask-socketio xgboost pandas numpy tqdm cloudpickle`.
- Run UI locally: `python tabularaml/ui/feature_generator_server.py` (serves at `http://localhost:5000`).
- Lint/format (if configured): `ruff check .` and `black .`.

## Coding Style & Naming Conventions
- Follow PEP 8, 4‑space indentation, type hints where practical.
- Modules: `snake_case.py`; classes: `CapWords`; functions/vars: `snake_case`.
- Keep pure logic in `tabularaml/*`; UI concerns stay in `tabularaml/ui/*`.
- Prefer small, testable functions; avoid stateful globals.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/` mirroring package paths (e.g., `tests/generate/test_features.py`).
- Names: `test_*.py`, functions `test_*`.
- Run: `pytest -q` (add `-k <pattern>` to filter). Aim to cover key paths in `generate/`, `preprocessing/`, and `eval/`.

## Commit & Pull Request Guidelines
- Commits: concise imperative subject (e.g., `fix: handle NaNs in ops`), body explains rationale and edge cases.
- PRs: include description, linked issues, repro steps, and before/after metrics or screenshots when UI‑related.
- Touch only relevant modules; keep changes scoped and documented.

## Security & Configuration Tips
- Model persistence uses `cloudpickle`. When saving from the UI, the server serializes a plain `FeatureGenerator` snapshot to avoid thread‑lock objects; keep UI objects out of core state.
- Use `cache/` for artifacts; do not commit large binaries. Add secrets via environment variables, not code.
