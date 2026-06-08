"""Own all filesystem boundaries for the final delivery pipeline.

Public API:
- validate_inputs: check declared input files and legacy scripts.
- prepare_run_workspace: build the per-run legacy workspace.
- write_manifest: save run metadata before execution.
- copy_public_outputs: copy tables, figures and reports into public folders.

The legacy step scripts are executed in a compatibility workspace because they
expect sibling Data, Src and Output folders.  The workspace lives under the
per-run output directory, so no root-level Output folder is created.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from mlf_c2o.config import ProjectConfig
from mlf_c2o.config import StepConfig


@dataclass(frozen=True)
class RunPaths:
    """Resolved directories for one run."""

    run_id: str
    run_dir: Path
    legacy_root: Path
    legacy_data_dir: Path
    legacy_src_dir: Path
    legacy_output_dir: Path
    tables_dir: Path
    figures_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    log_file: Path
    manifest_file: Path


def validate_inputs(config: ProjectConfig) -> None:
    """Raise if a declared input file or legacy step script is missing."""
    missing_inputs = [config.paths.input_dir / name for name in config.required_files if not (config.paths.input_dir / name).exists()]
    missing_scripts = [config.paths.legacy_step_dir / step.script for step in config.steps if not (config.paths.legacy_step_dir / step.script).exists()]
    if missing_inputs or missing_scripts:
        missing = [str(path) for path in missing_inputs + missing_scripts]
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))


def prepare_run_workspace(config: ProjectConfig, run_id: str) -> RunPaths:
    """Create a per-run workspace and expose inputs/scripts to legacy steps."""
    paths = _build_run_paths(config, run_id)
    _make_run_directories(paths)
    _expose_inputs(config, paths)
    _copy_legacy_scripts(config, paths)
    return paths


def write_manifest(config: ProjectConfig, paths: RunPaths, steps: Iterable[StepConfig]) -> None:
    """Write a self-describing manifest for the run."""
    payload = {
        "run_id": paths.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "selected_steps": [step.name for step in steps],
        "config": _stringify_paths(asdict(config)),
    }
    paths.manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_public_outputs(paths: RunPaths) -> None:
    """Copy report-facing outputs out of the compatibility workspace."""
    _copy_diagnostic_tables(paths)
    _copy_figures(paths)
    _copy_reports(paths)


def _copy_diagnostic_tables(paths: RunPaths) -> None:
    """Copy CSV and text diagnostics into the run tables directory."""
    diagnostics_dir = paths.legacy_output_dir / "diagnostics"
    if not diagnostics_dir.exists():
        return
    for path in diagnostics_dir.iterdir():
        if path.suffix.lower() in {".csv", ".txt"}:
            shutil.copy2(path, paths.tables_dir / path.name)


def _copy_figures(paths: RunPaths) -> None:
    figures_dir = paths.legacy_output_dir / "figures"
    if not figures_dir.exists():
        return
    for path in figures_dir.iterdir():
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
            shutil.copy2(path, paths.figures_dir / path.name)


def _copy_reports(paths: RunPaths) -> None:
    if not paths.legacy_output_dir.exists():
        return
    for path in paths.legacy_output_dir.iterdir():
        if path.suffix.lower() in {".html"}:
            shutil.copy2(path, paths.reports_dir / path.name)


def _build_run_paths(config: ProjectConfig, run_id: str) -> RunPaths:
    run_dir = config.paths.output_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    legacy_root = artifacts_dir / "legacy_workspace"
    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        legacy_root=legacy_root,
        legacy_data_dir=legacy_root / "Data",
        legacy_src_dir=legacy_root / "Src",
        legacy_output_dir=legacy_root / "Output",
        tables_dir=run_dir / "tables",
        figures_dir=run_dir / "figures",
        reports_dir=run_dir / "reports",
        artifacts_dir=artifacts_dir,
        log_file=config.paths.log_dir / f"{run_id}.log",
        manifest_file=run_dir / "manifest.json",
    )


def _make_run_directories(paths: RunPaths) -> None:
    for path in [
        paths.legacy_data_dir,
        paths.legacy_src_dir,
        paths.legacy_output_dir / "diagnostics",
        paths.tables_dir,
        paths.figures_dir,
        paths.reports_dir,
        paths.log_file.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _expose_inputs(config: ProjectConfig, paths: RunPaths) -> None:
    for filename in config.required_files:
        source = config.paths.input_dir / filename
        target = paths.legacy_data_dir / filename
        if not target.exists():
            _link_or_copy(source, target, config.run.copy_mode)


def _copy_legacy_scripts(config: ProjectConfig, paths: RunPaths) -> None:
    for step in config.steps:
        source = config.paths.legacy_step_dir / step.script
        shutil.copy2(source, paths.legacy_src_dir / step.script)


def _link_or_copy(source: Path, target: Path, copy_mode: str) -> None:
    if copy_mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _stringify_paths(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value
