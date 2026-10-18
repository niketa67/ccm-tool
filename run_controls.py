"""Command-line entry point for the CCM tool.

Examples
--------
Generate the synthetic data extract::

    python run_controls.py --generate-data

Run every enabled control and print a summary::

    python run_controls.py --run

Run and write Markdown/HTML/CSV evidence reports::

    python run_controls.py --run --report

Run a subset of controls against an explicit data folder::

    python run_controls.py --run --controls CC-01 CC-03 --data-dir data/raw
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

# Make ``src`` importable when the script is run from a fresh clone.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd  # noqa: E402

from ccm import __version__  # noqa: E402
from ccm.config import ConfigError, load_controls_config  # noqa: E402
from ccm.data_generator import generate_dataset  # noqa: E402
from ccm.engine import RunSummary, run_controls  # noqa: E402
from ccm.loader import DataValidationError, load_data_bundle  # noqa: E402
from ccm.paths import PROJECT_ROOT, RAW_DATA_DIR, REPORT_DIR, ensure_directories  # noqa: E402
from ccm.reporting import generate_reports  # noqa: E402
from ccm.storage import Storage  # noqa: E402

logger = logging.getLogger("ccm.cli")

#: Column widths for the terminal summary table.
_SUMMARY_COLUMNS = {
    "control_id": ("Control", 8),
    "name": ("Control name", 40),
    "severity": ("Severity", 9),
    "status": ("Status", 7),
    "total_tested": ("Tested", 8),
    "exception_count": ("Exceptions", 11),
    "exception_rate": ("Rate", 8),
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_controls.py",
        description="Continuous Controls Monitoring (CCM) tool - run automated ITGC control tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples", 1)[-1] if __doc__ else None,
    )
    parser.add_argument("--generate-data", action="store_true", help="generate the synthetic data extract")
    parser.add_argument("--run", action="store_true", help="execute all enabled controls")
    parser.add_argument("--report", action="store_true", help="write Markdown/HTML/CSV evidence reports")
    parser.add_argument("--controls", nargs="+", metavar="ID", help="run only these control ids (e.g. CC-01 CC-03)")
    parser.add_argument("--data-dir", type=Path, default=None, help="directory holding the source CSVs")
    parser.add_argument("--output-dir", type=Path, default=None, help="directory for reports (default: reports/)")
    parser.add_argument("--db", type=Path, default=None, help="SQLite database path (default: ccm.db)")
    parser.add_argument("--as-of", type=str, default=None, help="data as-of date for age tests (YYYY-MM-DD)")
    parser.add_argument("--include-disabled", action="store_true", help="run controls even if config disables them")
    parser.add_argument("--no-store", action="store_true", help="do not persist results to SQLite")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable debug logging")
    parser.add_argument("--version", action="version", version=f"ccm-tool {__version__}")
    return parser


def configure_logging(verbose: bool) -> None:
    """Set up console logging.

    Args:
        verbose: When ``True`` emit ``DEBUG``; otherwise ``INFO``.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def action_generate_data(rows: int, data_dir: Path | None) -> Path:
    """Generate the synthetic data extract and print an injection summary."""
    return generate_dataset(rows=rows, output_dir=data_dir)


def action_run(
    *,
    controls: Sequence[str] | None,
    data_dir: Path | None,
    db_path: Path | None,
    report: bool,
    as_of: str | None,
    include_disabled: bool,
    store: bool,
    output_dir: Path | None,
) -> int:
    """Execute controls, persist results and optionally write reports.

    Returns:
        A process exit code (0 on success, 1 on a load/config failure).
    """
    config = load_controls_config()

    if as_of:
        try:
            pd.Timestamp(as_of)
        except ValueError:
            print(f"ERROR: --as-of must be a valid date (got {as_of!r})", file=sys.stderr)
            return 1

    try:
        bundle = load_data_bundle(data_dir or RAW_DATA_DIR, as_of=as_of)
    except DataValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    selected = list(controls) if controls else None
    if controls:
        try:
            # Fail fast on unknown ids rather than reporting an empty run.
            from ccm.controls.registry import registered_control_ids

            unknown = sorted(set(controls) - set(registered_control_ids()))
            if unknown:
                raise KeyError(f"Unknown control id(s): {', '.join(unknown)}")
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    summary = run_controls(
        bundle,
        config=config,
        only=selected,
        include_disabled=include_disabled,
    )

    if store:
        with Storage(db_path or PROJECT_ROOT / "ccm.db") as storage:
            run_id = storage.save_run(summary)
            print(f"Run {run_id} stored in SQLite ({(db_path or PROJECT_ROOT / 'ccm.db')})")
    else:
        run_id = None

    if report:
        paths = generate_reports(summary, config=config, output_dir=output_dir or REPORT_DIR)
        for label, path in paths.items():
            print(f"  {label:16} -> {path}")

    _print_summary(summary)
    return 0


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _print_summary(summary: RunSummary) -> None:
    """Print a clean, fixed-width summary table to the terminal."""
    rows = summary.as_table().to_dict("records")

    header = "  ".join(f"{title:<{width}}" for _, (title, width) in _SUMMARY_COLUMNS.items())
    print()
    print("=" * len(header))
    print("CONTINUOUS CONTROLS MONITORING - RUN SUMMARY")
    print("=" * len(header))
    print(f"Run ID           : {summary.run_id if summary.run_id else '(not stored)'}")
    print(f"Executed at      : {summary.executed_at}")
    print(f"Data as-of       : {summary.as_of[:10]}")
    print(f"Data hash        : {summary.data_hash}")
    print(f"Controls executed: {summary.total_controls}  "
          f"(PASS {summary.passed} / FAIL {summary.failed} / ERROR {summary.error_count})")
    print(f"Total exceptions : {summary.total_exceptions}")
    print(f"Effectiveness    : {summary.effectiveness}% of controls operating effectively")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for row in rows:
        line = []
        for column, (_, width) in _SUMMARY_COLUMNS.items():
            value = row.get(column, "")
            if column == "exception_rate":
                text = f"{float(value) * 100:.2f}%" if value is not None else "-"
            elif column in ("total_tested", "exception_count"):
                text = f"{int(value):,}" if value is not None else "-"
            else:
                text = str(value)
            line.append(f"{text:<{width}}")
        print("  ".join(line))

    print("-" * len(header))

    if summary.warnings:
        print("Data quality warnings:")
        for warning in summary.warnings:
            print(f"  - {warning}")

    failing = [row for row in rows if row.get("status") == "FAIL"]
    if failing:
        print()
        print("Top remediation priorities (by severity then volume):")
        for row in failing:
            print(
                f"  {row['control_id']}  {row['severity']:<9} {row['exception_count']:>4} exception(s) "
                f"- {row['name']}"
            )
    print()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    ensure_directories()

    if not (args.generate_data or args.run):
        print(__doc__ or "No action specified.")
        build_parser().print_help()
        return 0

    if args.generate_data:
        generate_dataset(rows=500, output_dir=args.data_dir)
        if not args.run:
            return 0

    try:
        return action_run(
            controls=args.controls,
            data_dir=args.data_dir,
            db_path=args.db,
            report=args.report,
            as_of=args.as_of,
            include_disabled=args.include_disabled,
            store=not args.no_store,
            output_dir=args.output_dir,
        )
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1
    except DataValidationError as exc:
        print(f"DATA ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
