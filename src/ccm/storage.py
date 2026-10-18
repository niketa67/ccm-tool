"""SQLite persistence: the audit trail.

Every monitoring run is stored as three linked records so a reviewer can
reconstruct *what was tested, when, against which data*:

* ``runs``            - one row per execution, including the data snapshot hash.
* ``control_results`` - per-control outcome and thresholds for that run.
* ``exceptions``      - the individual findings, with a workflow status that the
  dashboard can update without touching history.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .engine import RunSummary
from .paths import DB_PATH, ensure_directories

logger = logging.getLogger(__name__)

__all__ = [
    "Storage",
    "SCHEMA_VERSION",
    "open_connection",
    "cell_for_db",
]

SCHEMA_VERSION = 1

#: Exception workflow statuses, in the order the dashboard offers them.
EXCEPTION_STATUSES: tuple[str, ...] = ("Open", "Acknowledged", "Remediated")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    executed_at         TEXT    NOT NULL,
    as_of               TEXT    NOT NULL,
    data_snapshot_hash  TEXT    NOT NULL,
    total_controls      INTEGER NOT NULL,
    passed              INTEGER NOT NULL,
    failed              INTEGER NOT NULL,
    error_count         INTEGER NOT NULL DEFAULT 0,
    total_exceptions    INTEGER NOT NULL,
    effectiveness       REAL    NOT NULL,
    ccm_version         TEXT    NOT NULL,
    warnings            TEXT    NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS control_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    control_id      TEXT    NOT NULL,
    control_name    TEXT    NOT NULL,
    status          TEXT    NOT NULL CHECK (status IN ('PASS','FAIL','ERROR')),
    severity        TEXT    NOT NULL,
    risk_category   TEXT    NOT NULL DEFAULT '',
    total_tested    INTEGER NOT NULL,
    exception_count INTEGER NOT NULL,
    exception_rate  REAL    NOT NULL,
    framework_mapping TEXT  NOT NULL DEFAULT '',
    parameters_json TEXT    NOT NULL DEFAULT '{}',
    executed_at     TEXT    NOT NULL,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    summary_text    TEXT    NOT NULL,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS exceptions (
    exception_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    control_id    TEXT    NOT NULL,
    account_id    TEXT,
    username      TEXT,
    employee_id   TEXT,
    system        TEXT,
    severity      TEXT    NOT NULL,
    reason        TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'Open'
                  CHECK (status IN ('Open','Acknowledged','Remediated')),
    created_at    TEXT    NOT NULL,
    updated_at    TEXT
);

CREATE INDEX IF NOT EXISTS ix_runs_executed_at    ON runs (executed_at);
CREATE INDEX IF NOT EXISTS ix_results_run        ON control_results (run_id);
CREATE INDEX IF NOT EXISTS ix_results_control    ON control_results (control_id);
CREATE INDEX IF NOT EXISTS ix_exceptions_run     ON exceptions (run_id);
CREATE INDEX IF NOT EXISTS ix_exceptions_control ON exceptions (control_id);
CREATE INDEX IF NOT EXISTS ix_exceptions_status  ON exceptions (status);
"""


def open_connection(path: Path | str) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys and row-dict access enabled.

    Args:
        path: Database file path. ``":memory:"`` is honoured for tests.

    Returns:
        A configured :class:`sqlite3.Connection`.
    """
    is_memory = str(path) == ":memory:"
    if not is_memory:
        ensure_directories()
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA foreign_keys = ON")
    connection.row_factory = sqlite3.Row
    connection.executescript(_SCHEMA)
    connection.commit()
    return connection


def cell_for_db(value: Any) -> Any:
    """Convert a pandas/NumPy value into a SQLite-friendly primitive.

    Handles :data:`pandas.NA`, ``numpy`` scalars, timestamps and the ``boolean``
    dtype so writing an exceptions frame never raises on a missing value.
    """
    if value is None:
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):  # arrays / exotic objects
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


@dataclass
class Storage:
    """Thin persistence layer over ``ccm.db``.

    The class holds one SQLite connection *per thread*. Streamlit re-runs the
    script on a fresh thread for every interaction, which would otherwise
    invalidate a cached ``sqlite3.Connection`` ("SQLite objects created in a
    thread can only be used in that same thread"), so keeping the connection in
    a :class:`threading.local` makes the object safe to cache and share.

    Args:
        path: Database location. Defaults to ``<project>/ccm.db``.
    """

    path: Path | str = field(default_factory=lambda: DB_PATH)
    _local: threading.local = field(default_factory=threading.local, repr=False, init=False)

    def __post_init__(self) -> None:
        """Normalise the path; the connection is opened lazily per thread."""
        self.path = Path(self.path) if str(self.path) != ":memory:" else Path(":memory:")

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #
    @property
    def connection(self) -> sqlite3.Connection:
        """A connection bound to the *current* thread, created on first use."""
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = open_connection(self.path)
            setattr(self._local, "connection", connection)
        return connection

    def close(self) -> None:
        """Close this thread's connection, releasing locks and file handles."""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            setattr(self._local, "connection", None)

    def __enter__(self) -> "Storage":
        """Enter a context manager, returning the storage instance."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the current thread's connection on context exit."""
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown guard
        """Best-effort release of the database handle."""
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def save_run(self, summary: RunSummary, *, executed_at: str | None = None) -> int:
        """Persist a run with its control results and exceptions.

        Args:
            summary: The run to persist.
            executed_at: Optional timestamp override (defaults to the summary's).

        Returns:
            The new ``run_id``.
        """
        timestamp = executed_at or summary.executed_at
        conn = self.connection
        try:
            cursor = conn.execute(
                """
                INSERT INTO runs (
                    executed_at, as_of, data_snapshot_hash, total_controls, passed, failed,
                    error_count, total_exceptions, effectiveness, ccm_version, warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    summary.as_of,
                    summary.data_hash,
                    summary.total_controls,
                    summary.passed,
                    summary.failed,
                    summary.error_count,
                    summary.total_exceptions,
                    summary.effectiveness,
                    _ccm_version(),
                    json.dumps(summary.warnings),
                ),
            )
            run_id = int(cursor.lastrowid)  # type: ignore[arg-type]

            for result in summary.results:
                conn.execute(
                    """
                    INSERT INTO control_results (
                        run_id, control_id, control_name, status, severity, risk_category,
                        total_tested, exception_count, exception_rate, framework_mapping,
                        parameters_json, executed_at, duration_ms, summary_text, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        result.control_id,
                        result.name,
                        result.status,
                        result.severity,
                        result.risk_category,
                        result.total_tested,
                        result.exception_count,
                        result.exception_rate,
                        "; ".join(result.framework_mapping),
                        json.dumps(result.parameters_used, sort_keys=True, default=str),
                        result.executed_at.isoformat(),
                        result.duration_ms,
                        result.summary_text,
                        result.error,
                    ),
                )

                self._insert_exceptions(run_id, result)

            conn.commit()
            logger.info("Saved run %d: %d control(s), %d exception(s)", run_id, len(summary.results), summary.total_exceptions)
            summary.run_id = run_id
            return run_id
        except Exception:
            conn.rollback()
            raise

    def _insert_exceptions(self, run_id: int, result) -> None:
        """Insert the exception rows for one control result."""
        frame = result.exceptions
        created_at = result.executed_at.isoformat()
        columns = ["account_id", "username", "employee_id", "system", "severity", "reason"]

        if frame is None or len(frame) == 0:
            return

        payload: list[tuple[Any, ...]] = []
        for _, row in frame.iterrows():
            values = [cell_for_db(row.get(column)) for column in columns]
            payload.append((run_id, result.control_id, *values, "Open", created_at))

        self.connection.executemany(
            """
            INSERT INTO exceptions (
                run_id, control_id, account_id, username, employee_id, system,
                severity, reason, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )

    def update_exception_status(self, exception_id: int, status: str) -> bool:
        """Set the workflow status of one exception.

        Args:
            exception_id: Primary key of the exception row.
            status: One of :data:`EXCEPTION_STATUSES`.

        Returns:
            ``True`` if a row was updated.

        Raises:
            ValueError: If ``status`` is not a valid workflow state.
        """
        if status not in EXCEPTION_STATUSES:
            raise ValueError(f"Invalid status {status!r}; expected one of {', '.join(EXCEPTION_STATUSES)}")

        cursor = self.connection.execute(
            "UPDATE exceptions SET status = ?, updated_at = ? WHERE exception_id = ?",
            (status, datetime.now(timezone.utc).isoformat(), exception_id),
        )
        self.connection.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info("Exception %d status set to %s", exception_id, status)
        return updated

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def list_runs(self, limit: int | None = None) -> pd.DataFrame:
        """Return run headers, newest first.

        Args:
            limit: Optional cap on the number of rows.
        """
        query = "SELECT * FROM runs ORDER BY executed_at DESC, run_id DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        return pd.read_sql_query(query, self.connection)

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """Return the ``runs`` row for ``run_id``, or ``None``."""
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_results(self, run_id: int | None = None) -> pd.DataFrame:
        """Return control results for one run.

        Args:
            run_id: The run to read. Defaults to the most recent run, so callers
                exploring the database do not need to look the id up first.
        """
        resolved = run_id if run_id is not None else self.latest_run_id()
        if resolved is None:
            return pd.DataFrame(columns=["run_id", "control_id"])
        return pd.read_sql_query(
            "SELECT * FROM control_results WHERE run_id = ? ORDER BY control_id",
            self.connection,
            params=(resolved,),
        )

    def get_exceptions(
        self,
        run_id: int | None = None,
        *,
        control_id: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """Return exceptions, optionally filtered by run, control and status.

        Args:
            run_id: Restrict to one run.
            control_id: Restrict to one control.
            status: Restrict to one workflow status.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("e.run_id = ?")
            params.append(run_id)
        if control_id:
            clauses.append("e.control_id = ?")
            params.append(control_id)
        if status:
            clauses.append("e.status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return pd.read_sql_query(
            f"""
            SELECT e.*, cr.severity AS control_severity, cr.control_name,
                   r.executed_at AS run_executed_at
            FROM exceptions e
            LEFT JOIN control_results cr
                   ON cr.run_id = e.run_id AND cr.control_id = e.control_id
            LEFT JOIN runs r ON r.run_id = e.run_id
            {where}
            ORDER BY e.run_id DESC, e.control_id, e.exception_id
            """,
            self.connection,
            params=params,
        )

    def exception_trend(self, limit: int = 20) -> pd.DataFrame:
        """Return exception counts per control per run, oldest first.

        Args:
            limit: Number of most-recent runs to include.

        Returns:
            A DataFrame with columns ``run_id``, ``executed_at``, ``control_id``,
            ``exception_count`` - the input for the Trends chart.
        """
        return pd.read_sql_query(
            f"""
            SELECT r.run_id,
                   r.executed_at,
                   cr.control_id,
                   cr.severity,
                   COALESCE(cr.exception_count, 0) AS exception_count
            FROM runs r
            JOIN control_results cr ON cr.run_id = r.run_id
            WHERE r.run_id IN (SELECT run_id FROM runs ORDER BY executed_at DESC, run_id DESC LIMIT {int(limit)})
            ORDER BY r.run_id, cr.control_id
            """,
            self.connection,
        )

    def latest_run_id(self) -> int | None:
        """Return the most recent ``run_id``, or ``None`` when the database is empty."""
        row = self.connection.execute("SELECT MAX(run_id) AS latest FROM runs").fetchone()
        return int(row["latest"]) if row and row["latest"] is not None else None

    def clear_all(self) -> None:
        """Delete all runs and results. Used by tests and demo re-seeding."""
        with self.connection:
            self.connection.executescript(
                "DELETE FROM exceptions; DELETE FROM control_results; DELETE FROM runs;"
            )
            self.connection.execute("DELETE FROM sqlite_sequence WHERE name IN ('runs','control_results','exceptions')")
        logger.warning("All stored runs deleted")


def _ccm_version() -> str:
    """Return the installed package version, tolerating a missing metadata entry."""
    try:
        from importlib.metadata import version

        return version("ccm-tool")
    except Exception:  # pragma: no cover - not installed as a distribution
        return "1.0.0"
