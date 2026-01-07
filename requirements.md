# Installation Requirements (cern-analysis-common)

`cern-analysis-common` is a Python library submodule (`src/cern_analysis_common`)
for shared HEP analysis utilities (vector helpers, cuts, IO, etc.).

## Prerequisites

- Python `>=3.9`

## Install (dev)

```bash
cd cern-analysis-common
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

## Tests and gates

- Tests: `pytest`
- Strict gates (warnings-as-errors on the contract surface): `scripts/audit/run_tiers.sh`

