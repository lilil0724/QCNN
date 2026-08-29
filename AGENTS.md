# Repository Guidelines

## Project Structure & Module Organization

`weight2ternary/` contains the reusable analysis package: `data_utils/` prepares and
samples paired checkpoint tensors, `model_utils/` defines weight-map models and
losses, and `eval_utils/` assembles and evaluates outputs. Command-line workflows
live in `tools/`; keep tool-specific loading and argument handling there while
placing reusable logic in the package. `tests/` holds lightweight synthetic checks.
`scripts/` contains shell workflows, and `docs/` records experiment context and
findings. Generated output belongs in ignored `results/`.

Read `docs/HANDOFF.md` and the checkpoint-family warning in
`docs/BONSAI_QWEN3_1.7B_FINDINGS.md` before starting a new model comparison.

## Setup, Test, and Development Commands

Install base dependencies with `pip install -r requirements.txt`; install the
additional package dependencies with `pip install -r requirements-w2t.txt` when
working on `weight2ternary` workflows.

- `bash scripts/run_tests.sh` runs every no-download synthetic test and tool check.
- `python tools/compare_qat_weights.py --synthetic` validates comparison statistics
  without a GPU or model download.
- `python -m py_compile path/to/file.py` syntax-checks a changed Python file.

The PoC scripts are GPU workflows, not routine local tests. Run them only with their
paths and storage assumptions reviewed.

## Coding Style & Naming Conventions

Use Python with four-space indentation, `snake_case` for modules, functions, and
variables, and `PascalCase` for classes. Keep command-line flags descriptive and
consistent with existing tools (for example, `--pair_dir` and `--base_model_id`).
Prefer small, explicit functions and assertions in synthetic checks. No formatter or
linter is configured; follow nearby code and avoid unrelated reformatting.

## Testing Guidelines

Name tests `tests/test_<feature>.py` and make them runnable directly with
`python tests/test_<feature>.py`; tests currently use plain assertions rather than a
test framework. Add a synthetic case for changed tensor logic, model behavior, or
CLI validation. Do not treat truncated logs as a successful run: inspect the full
output for errors and tracebacks.

## Commit & Pull Request Guidelines

Existing history uses short, imperative-style summaries such as `first commit,
feasibility study` and `updated with first and second round of experiments`. Use a
concise summary that names the substantive change. Pull requests should explain the
experiment or code change, list commands run and their results, link related issues
when available, and include relevant plots or output excerpts for behavior changes.

## Data, Models, and Safety

Use `HF_HOME=/home/pcs5060ti/Desktop/hf` and
`HF_HUB_CACHE=/home/pcs5060ti/Desktop/hf/hub` explicitly before Hub operations.
Keep derived pair shards outside the repository, document exclusions or retries, and
verify GPU-backed conclusions with actual runs rather than code inspection alone.
