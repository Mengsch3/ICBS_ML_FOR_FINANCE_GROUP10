"""Install the pinned runtime dependencies for a clean-machine run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = PROJECT_ROOT / "requirements-lock.txt"


def main() -> None:
    """Install the locked dependencies and editable package."""
    if not LOCK_FILE.exists():
        raise FileNotFoundError(f"Missing dependency lock file: {LOCK_FILE}")

    expected = (3, 13, 5)
    if sys.version_info[:3] != expected:
        found = ".".join(str(part) for part in sys.version_info[:3])
        required = ".".join(str(part) for part in expected)
        raise RuntimeError(f"Python {required} is required; found Python {found}.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(LOCK_FILE),
            "-e",
            str(PROJECT_ROOT),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
