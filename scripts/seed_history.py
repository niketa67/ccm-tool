"""Seed historical monitoring runs so the dashboard's Trends view has data.

Each seed run advances the data's as-of date, simulating monitoring over the
same extract across several weeks. Because the source data is unchanged, every
run reports identical findings - which is itself the point: repeated runs
against the same inputs must be reproducible, and any drift would indicate the
engine (or the data pipeline) is unstable.

The Trends chart draws its real value when the underlying data changes between
periods; regenerate the extract with a different :class:`InjectionPlan` (see
``src/ccm/data_generator.py``) to watch findings fall over time.

Run from the project root:

    python scripts/seed_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ccm.config import load_controls_config  # noqa: E402
from ccm.engine import run_controls  # noqa: E402
from ccm.loader import load_data_bundle  # noqa: E402
from ccm.paths import DB_PATH, RAW_DATA_DIR  # noqa: E402
from ccm.storage import Storage  # noqa: E402


def main() -> None:
    """Generate historical runs at fortnightly as-of dates."""
    historical = [
        "2026-09-10",
        "2026-09-24",
        "2026-10-08",
        "2026-10-18",
    ]

    print("Seeding historical runs (one per monitoring period):")
    with Storage(DB_PATH) as storage:
        for as_of in historical:
            data = load_data_bundle(RAW_DATA_DIR, as_of=as_of)
            summary = run_controls(data, load_controls_config())
            storage.save_run(summary, executed_at=f"{as_of}T09:00:00+00:00")
            print(
                f"  run #{summary.run_id}: as-of {as_of} -> {summary.total_exceptions} exception(s) "
                f"(data hash {summary.data_hash[:12]}...)"
            )


if __name__ == "__main__":
    main()
