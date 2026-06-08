"""Smoke tests for the final delivery package."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlf_c2o import run_pipeline


def test_dry_run_loads_config_and_steps() -> None:
    """The dry run validates inputs and returns the configured step plan."""
    result = run_pipeline(dry_run=True)
    assert result.run_id == "dry-run"
    assert result.completed_steps[-1] == "step8_quantstats_tearsheet"
