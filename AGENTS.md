# AGENTS

## Verification

Fast smoke check:

```bash
python scripts/verify.py
```

Pinned environment setup:

```bash
python scripts/setup_env.py
```

Full run:

```bash
python main.py
```

## Architecture Rules

- The public entry point is `mlf_c2o.run_pipeline`.
- All input paths, output paths and pipeline steps are declared in
  `config/default.yaml`.
- Python is pinned in `.python-version`; dependencies are pinned in
  `requirements-lock.txt`.
- CLI runs load `.env` and use `MLF_C2O_CONFIG` when `--config` is omitted.
- The project never writes to `data/inputs/`.
- Every full run writes to `data/outputs/<run_id>/` and logs to `logs/<run_id>.log`.
- The preserved step scripts live in `src/mlf_c2o/legacy_steps/`; do not create
  parallel orchestrators.
- The configured pipeline has eight steps. Step 7 generates the nine report
  figures and Step 8 generates the 250M QuantStats tear-sheet.

## Gotchas

- `config/default.yaml` is JSON-compatible YAML so no YAML dependency is needed.
- The original step scripts are intentionally copied into a per-run compatibility
  workspace because they expect sibling `Data`, `Src` and `Output` folders.
- Public run artifacts are copied from the compatibility workspace into
  `tables/`, `figures/` and `reports/`.
- The full alpha-model run is slow.  Use `python scripts/verify.py` for quick
  checks before running the complete pipeline.
