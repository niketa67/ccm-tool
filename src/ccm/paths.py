"""Filesystem layout for the CCM tool.

All paths are derived from the location of this package so the project runs from
any working directory and never depends on a hard-coded absolute path.

The repository root can be overridden with the ``CCM_HOME`` environment
variable, which is what the test-suite and the Docker image use to redirect
generated artefacts.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "REPORT_DIR",
    "DB_PATH",
    "CONTROLS_CONFIG_PATH",
    "SOD_RULES_PATH",
    "ensure_directories",
]

# ``src/ccm/paths.py`` -> parents[2] == repository root (the folder holding
# ``config/``, ``src/`` and ``app/``).
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT: Path = Path(os.environ.get("CCM_HOME", _DEFAULT_ROOT)).resolve()

CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
REPORT_DIR: Path = PROJECT_ROOT / "reports"

CONTROLS_CONFIG_PATH: Path = CONFIG_DIR / "controls.yaml"
SOD_RULES_PATH: Path = CONFIG_DIR / "sod_rules.yaml"

#: SQLite database holding runs, control results and exceptions.
DB_PATH: Path = PROJECT_ROOT / "ccm.db"


def ensure_directories() -> None:
    """Create the directories the pipeline writes to, if they do not exist."""
    for directory in (CONFIG_DIR, RAW_DATA_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
