"""Fast smoke verification for the final delivery package."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlf_c2o import run_pipeline


def main() -> None:
    """Validate config, inputs and step declarations without running models."""
    run_pipeline(dry_run=True)


if __name__ == "__main__":
    main()

