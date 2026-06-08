# C2O Final Delivery Pipeline

This folder is the final code-only delivery for the Machine Learning and
Finance C2O coursework. It does not include the LaTeX report. The pipeline runs
all eight implemented steps:

1. Daily panel construction
2. Capacity-aware universe
3. Borrow-cost model
4. Alpha model
5. Back-test and cost decomposition
6. Audit diagnostics
7. Report figure generation
8. QuantStats tear-sheet generation

## Clean-Machine Reproduction

A new developer should be able to reproduce a historical run with only these
steps:

1. Clone the repo.
2. Install the Python version pinned in `.python-version`: Python 3.13.5.
3. Run the one-line environment setup:

```bash
python scripts/setup_env.py
```

4. Copy `.env.example` to `.env`.

```bash
copy .env.example .env
```

On macOS/Linux use:

```bash
cp .env.example .env
```

5. Point the config at the same input data. The default config expects the
   parquet inputs in `data/inputs/`; alternatively edit only
   `config/default.yaml` or set `MLF_C2O_CONFIG` in `.env`.
6. Run the same config file and, if reproducing a named historical run, the
   same run id:

```bash
python main.py --config config/default.yaml --run-id <historical_run_id>
```

For a new run, omit `--run-id`:

```bash
python main.py
```

No Python file should be edited to change paths, run ids or inputs.

## Project Layout

```text
MLF_C2O_Final_Deliverable/
|-- config/                  # checked-in config files
|-- src/mlf_c2o/             # installable package and single entry point
|-- data/inputs/             # read-only parquet inputs
|-- data/intermediary/       # disposable local workspace
|-- data/outputs/<run_id>/   # per-run outputs
|-- logs/                    # one log file per run
|-- scripts/                 # setup and verification helpers
`-- tests/                   # smoke checks
```

The only public function needed to run the project is:

```python
from mlf_c2o import run_pipeline

run_pipeline()
```

## Setup

The pinned dependency set is in `requirements-lock.txt`. The setup helper
installs those exact versions and then installs this package in editable mode:

```bash
python scripts/setup_env.py
```

Equivalent manual command:

```bash
python -m pip install -r requirements-lock.txt -e .
```

## Run

Run the complete pipeline directly from this folder:

```bash
python main.py
```

If the package has been installed, the module form also works:

```bash
python -m mlf_c2o.main
```

Run a dry check without executing the long model/back-test steps:

```bash
python scripts/verify.py
```

Run selected steps:

```bash
python main.py --start-at step5_backtest
python main.py --run-id <existing_run_id> --only step7_report_figures,step8_quantstats_tearsheet
```

Selected late steps require the same `run_id` as a prior run because they read
the earlier step outputs from that run's compatibility workspace.

Each full run writes to a fresh directory:

```text
data/outputs/<run_id>/
|-- manifest.json
|-- artifacts/legacy_workspace/Output/
|-- figures/
|-- reports/
`-- tables/
```

The `tables/` folder contains the CSV/TXT diagnostics copied from the run,
including the Step 5 headline table, cost decomposition, console output and
Step 8 QuantStats summary. The `figures/` folder contains the nine report
figures. The `reports/` folder contains `tearsheet_250m.html`.

## Inputs

Input files are declared in `config/default.yaml`. The pipeline treats
`data/inputs/` as read-only. If data must be replaced, replace the files there;
do not edit paths inside Python code.

## Notes

The preserved step scripts are in `src/mlf_c2o/legacy_steps/`. The delivery
wrapper provides the architecture boundary required by the guideline:
config-driven paths, one public entry point, per-run outputs, logs outside
outputs, pinned dependencies, `.env` support and a run manifest.
