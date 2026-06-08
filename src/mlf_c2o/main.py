"""Single entry point for the final C2O pipeline delivery.

Public API:
- run_pipeline: run all configured steps from a typed config object.

The orchestrator only prepares the run workspace, executes named steps, checks
declared outputs and writes a manifest.  Domain calculations remain in the
step modules copied into the per-run compatibility workspace.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mlf_c2o.config import ProjectConfig
from mlf_c2o.config import StepConfig
from mlf_c2o.config import load_config
from mlf_c2o.env import load_env_file
from mlf_c2o.io import RunPaths
from mlf_c2o.io import copy_public_outputs
from mlf_c2o.io import prepare_run_workspace
from mlf_c2o.io import validate_inputs
from mlf_c2o.io import write_manifest


@dataclass(frozen=True)
class RunResult:
    """Summary returned after a pipeline run."""

    run_id: str
    output_dir: Path
    log_file: Path
    completed_steps: tuple[str, ...]


def run_pipeline(
    config_path: str | Path = "config/default.yaml",
    dry_run: bool = False,
    run_id: str | None = None,
    start_at: str | None = None,
    only: str | None = None,
) -> RunResult:
    """Run all configured pipeline steps and return the run output locations."""
    config = load_config(config_path)
    validate_inputs(config)
    steps = _select_steps(config, start_at=start_at, only=only)
    resolved_run_id = _resolve_run_id(config, run_id, dry_run)

    if dry_run:
        _print_plan(config, steps, resolved_run_id)
        return RunResult(resolved_run_id, config.paths.output_dir / resolved_run_id, config.paths.log_dir / f"{resolved_run_id}.log", tuple(step.name for step in steps))

    paths = prepare_run_workspace(config, resolved_run_id)
    write_manifest(config, paths, steps)
    _run_steps(config, paths, steps)
    copy_public_outputs(paths)
    return RunResult(paths.run_id, paths.run_dir, paths.log_file, tuple(step.name for step in steps))


def _select_steps(config: ProjectConfig, start_at: str | None, only: str | None) -> tuple[StepConfig, ...]:
    step_names = [step.name for step in config.steps]
    if only:
        requested = [name.strip() for name in only.split(",") if name.strip()]
        _validate_step_names(requested, step_names)
        return tuple(step for step in config.steps if step.name in requested)
    if start_at:
        _validate_step_names([start_at], step_names)
        return config.steps[step_names.index(start_at):]
    return config.steps


def _run_steps(config: ProjectConfig, paths: RunPaths, steps: tuple[StepConfig, ...]) -> None:
    python = config.run.python_executable or sys.executable
    with paths.log_file.open("w", encoding="utf-8") as log:
        for step in steps:
            _log(log, f"START {step.name}")
            _run_one_step(python, config, paths, step, log)
            _check_expected_outputs(paths, step)
            _log(log, f"DONE  {step.name}")
        _log(log, "PIPELINE COMPLETED")


def _run_one_step(python: str, config: ProjectConfig, paths: RunPaths, step: StepConfig, log) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(
        [python, step.script],
        cwd=paths.legacy_src_dir,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=config.run.timeout_seconds,
        check=True,
    )


def _check_expected_outputs(paths: RunPaths, step: StepConfig) -> None:
    missing = [paths.legacy_root / output for output in step.expected_outputs if not (paths.legacy_root / output).exists()]
    if missing:
        raise FileNotFoundError(f"{step.name} finished but outputs are missing: {missing}")


def _resolve_run_id(config: ProjectConfig, run_id: str | None, dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    if run_id:
        return run_id
    if config.run.run_id:
        return config.run.run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _validate_step_names(requested: list[str], available: list[str]) -> None:
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(f"Unknown step name(s): {', '.join(unknown)}")


def _print_plan(config: ProjectConfig, steps: tuple[StepConfig, ...], run_id: str) -> None:
    print("C2O final pipeline dry run")
    print(f"Project root: {config.paths.project_root}")
    print(f"Run id: {run_id}")
    print("Steps: " + " -> ".join(step.name for step in steps))


def _log(log, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line)
    log.write(line + "\n")
    log.flush()


def _load_cli_env() -> None:
    project_root = Path(__file__).resolve().parents[2]
    root_env = project_root / ".env"
    cwd_env = Path.cwd() / ".env"
    load_env_file(root_env)
    if cwd_env.resolve() != root_env.resolve():
        load_env_file(cwd_env)


def main() -> None:
    """CLI wrapper for run_pipeline."""
    _load_cli_env()
    parser = argparse.ArgumentParser(description="Run the final C2O pipeline.")
    parser.add_argument("--config", default=os.environ.get("MLF_C2O_CONFIG", "config/default.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--start-at")
    parser.add_argument("--only")
    args = parser.parse_args()
    result = run_pipeline(args.config, args.dry_run, args.run_id, args.start_at, args.only)
    print(f"Output directory: {result.output_dir}")
    print(f"Log file: {result.log_file}")


if __name__ == "__main__":
    main()
