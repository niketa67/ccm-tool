"""Print a compact summary of the runs stored in ccm.db."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccm.storage import Storage  # noqa: E402


def main() -> None:
    """List every stored run."""
    runs = Storage().list_runs()
    if runs.empty:
        print("ccm.db is empty.")
        return
    for _, row in runs.iterrows():
        print(
            f"run #{row['run_id']:>3}  {row['executed_at'][:19]}  "
            f"controls={row['total_controls']}  exceptions={row['total_exceptions']}"
        )


if __name__ == "__main__":
    main()
