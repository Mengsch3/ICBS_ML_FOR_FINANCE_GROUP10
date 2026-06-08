"""Small .env loader used by the CLI.

The project does not require secrets, but the clean-machine workflow still uses
`.env` for local config selection.  This parser intentionally supports only the
simple KEY=VALUE form needed here, avoiding an extra runtime dependency.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path) -> None:
    """Load KEY=VALUE pairs from path without overriding existing variables."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_value(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
