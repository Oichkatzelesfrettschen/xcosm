# Installation Requirements (xcosm)

`xcosm` is a science application submodule. The policy is **library-first**:
only the declared “typed library surface” is gated; research scripts/docs are
excluded until promoted.

## Prerequisites

- Python (see `xcosm/pyproject.toml`)

## Install (dev)

```bash
cd xcosm
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quality gates (warnings-as-errors)

- Ruff: `cd xcosm && ruff check .`
- Mypy (scoped): `cd xcosm && mypy --config-file mypy.ini`

The meta-repo strict contract runner enforces only the scoped surfaces; run from
repo root: `scripts/audit/run_tiers.sh`.

