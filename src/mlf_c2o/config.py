"""Load and validate the typed project configuration.

Public API:
- load_config: read config/default.yaml and return a ProjectConfig.

The YAML file is written as JSON-compatible YAML so the project avoids a
runtime dependency on PyYAML.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    """Resolved project paths."""

    project_root: Path
    input_dir: Path
    intermediary_dir: Path
    output_dir: Path
    log_dir: Path
    legacy_step_dir: Path


@dataclass(frozen=True)
class RunConfig:
    """Runtime settings for one pipeline invocation."""

    copy_mode: str
    python_executable: str
    run_id: str
    timeout_seconds: int


@dataclass(frozen=True)
class StepConfig:
    """One legacy step script and the outputs it must produce."""

    name: str
    script: str
    expected_outputs: tuple[str, ...]


@dataclass(frozen=True)
class ProjectConfig:
    """Validated project configuration."""

    paths: PathsConfig
    run: RunConfig
    required_files: tuple[str, ...]
    steps: tuple[StepConfig, ...]
    raw: dict[str, Any]


def load_config(config_path: str | Path, project_root: str | Path | None = None) -> ProjectConfig:
    """Return a validated ProjectConfig from a JSON-compatible YAML file."""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    path = _resolve(root, config_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _build_config(raw, root)


def _build_config(raw: dict[str, Any], project_root: Path) -> ProjectConfig:
    paths = raw["paths"]
    run = raw["run"]
    return ProjectConfig(
        paths=PathsConfig(
            project_root=project_root,
            input_dir=_resolve(project_root, paths["input_dir"]),
            intermediary_dir=_resolve(project_root, paths["intermediary_dir"]),
            output_dir=_resolve(project_root, paths["output_dir"]),
            log_dir=_resolve(project_root, paths["log_dir"]),
            legacy_step_dir=_resolve(project_root, paths["legacy_step_dir"]),
        ),
        run=RunConfig(
            copy_mode=str(run["copy_mode"]),
            python_executable=str(run["python_executable"]),
            run_id=str(run["run_id"]),
            timeout_seconds=int(run["timeout_seconds"]),
        ),
        required_files=tuple(str(name) for name in raw["inputs"]["required_files"]),
        steps=tuple(_build_step(step) for step in raw["steps"]),
        raw=raw,
    )


def _build_step(raw_step: dict[str, Any]) -> StepConfig:
    return StepConfig(
        name=str(raw_step["name"]),
        script=str(raw_step["script"]),
        expected_outputs=tuple(str(path) for path in raw_step["expected_outputs"]),
    )


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path

