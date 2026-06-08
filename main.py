"""Root-level runner for the final C2O delivery package.

This wrapper lets users run the project with:

    python main.py

without first installing the package in editable mode.  The actual public API
remains mlf_c2o.run_pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlf_c2o.main import main


if __name__ == "__main__":
    main()

