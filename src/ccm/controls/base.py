"""Control framework primitives: :class:`ControlResult` and :class:`BaseControl`.

A control is a small, self-contained test with a stable identity. Subclasses
implement two methods:

* :meth:`BaseControl.population` - the rows the control is accountable for (the
  denominator of the exception rate).
* :meth:`BaseControl.evaluate` - the rows that breach policy (the numerator).

Everything else - timing, exception-rate maths, PASS/FAIL derivation, evidence
shape, error containment - lives here so that each control stays short enough to
read in one sitting.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Mapping, Sequence

import pandas as pd

from ..config import severity_rank
from ..loader import DataBundle

logger = logging.getLogger(__name__)

__all__ = [
    "ControlStatus",
    "EXCEPTION_COLUMNS",
    "ControlResult",
    "BaseControl",
    "empty_exceptions",
    "normalise_exceptions",
]


class ControlStatus(str, Enum):
    """Outcome of a single control execution.

    ``ERROR`` exists so that an infrastructure failure (a broken column, a bad
    rule file) is never silently reported as a clean ``PASS``.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


#: Columns every exception row must carry, in report order.
EXCEPTION_COLUMNS: tuple[str, ...] = (
    "account_id",
    "username",
    "employee_id",
    "system",
    "severity",
    "reason",
)


def empty_exceptions() -> pd.DataFrame:
    """Return an empty exception frame carrying the mandatory evidence columns."""
    return pd.DataFrame(columns=list(EXCEPTION_COLUMNS))


def normalise_exceptions(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Coerce a control's raw output into the canonical evidence shape.

    Guarantees the mandatory columns exist, puts them first, and fills any gaps
    with :data:`pandas.NA` so downstream CSV/SQLite writes never trip over a
    missing column.
    """
    if frame is None or len(frame) == 0:
        return empty_exceptions()

    frame = frame.copy()
    for column in EXCEPTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    ordered = list(EXCEPTION_COLUMNS) + [column for column in frame.columns if column not in EXCEPTION_COLUMNS]
    return frame.loc[:, ordered].reset_index(drop=True)


@dataclass
class ControlResult:
    """The auditable outcome of one control execution.

    Attributes:
        control_id: Stable control identifier, e.g. ``"CC-01"``.
        status: ``PASS``, ``FAIL`` or ``ERROR``.
        total_tested: Size of the population the control evaluated.
        exception_count: Number of failing rows.
        exception_rate: ``exception_count / total_tested`` (0.0 when untested).
        exceptions: The exception rows, in the canonical evidence shape.
        executed_at: Timezone-aware UTC timestamp of execution.
        parameters_used: Effective thresholds, after config overrides.
        summary_text: One-line human-readable finding.
        name: Control display name.
        description: Full control description.
        risk_category: Risk taxonomy bucket.
        severity: Severity of the control's failure mode.
        framework_mapping: Framework references this control satisfies.
        duration_ms: Wall-clock execution time in milliseconds.
        error: Populated only when ``status`` is ``ERROR``.
    """

    control_id: str
    status: str
    total_tested: int
    exception_count: int
    exception_rate: float
    exceptions: pd.DataFrame
    executed_at: datetime
    parameters_used: dict[str, Any]
    summary_text: str
    name: str = ""
    description: str = ""
    risk_category: str = ""
    severity: str = "Medium"
    framework_mapping: tuple[str, ...] = field(default_factory=tuple)
    duration_ms: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        """``True`` when the control found no exceptions."""
        return self.status == ControlStatus.PASS.value

    @property
    def severity_rank(self) -> int:
        """Numeric severity rank, for sorting most-severe-first."""
        return severity_rank(self.severity)

    def as_row(self) -> dict[str, Any]:
        """Flatten to a dict, excluding the (potentially large) exception frame."""
        return {
            "control_id": self.control_id,
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "risk_category": self.risk_category,
            "total_tested": self.total_tested,
            "exception_count": self.exception_count,
            "exception_rate": self.exception_rate,
            "executed_at": self.executed_at.isoformat(),
            "duration_ms": self.duration_ms,
            "summary_text": self.summary_text,
            "framework_mapping": "; ".join(self.framework_mapping),
            "parameters_used": dict(self.parameters_used),
            "error": self.error,
        }


class BaseControl(ABC):
    """Abstract base class for every CCM control.

    Class attributes describe the control's identity and default policy; the
    YAML config may override severity and any parameter at runtime.

    Class Attributes:
        control_id: Stable identifier used by the registry and config.
        name: Short human-readable name.
        description: What the control asserts and why it matters.
        risk_category: Risk taxonomy bucket.
        severity: Default severity when config does not override it.
        framework_mapping: Default framework references.
        default_parameters: Default thresholds, merged with config values.
    """

    control_id: ClassVar[str] = ""
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    risk_category: ClassVar[str] = "General IT Controls"
    severity: ClassVar[str] = "Medium"
    framework_mapping: ClassVar[tuple[str, ...]] = ()
    default_parameters: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *,
        severity: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
        risk_category: str | None = None,
        framework_mapping: Sequence[str] | None = None,
    ) -> None:
        """Build a control instance, merging config over class defaults.

        Args:
            severity: Severity override from config.
            parameters: Threshold overrides from config, merged over
                ``default_parameters``.
            name: Display-name override.
            description: Description override.
            risk_category: Risk-category override.
            framework_mapping: Framework-mapping override.
        """
        self.severity: str = severity or self.severity
        self.parameters: dict[str, Any] = {**self.default_parameters, **dict(parameters or {})}
        self.name: str = name or self.name
        self.description: str = description or self.description
        self.risk_category: str = risk_category or self.risk_category
        self.framework_mapping: tuple[str, ...] = tuple(framework_mapping or self.framework_mapping)

    # ------------------------------------------------------------------ #
    # Subclass responsibilities
    # ------------------------------------------------------------------ #
    def population(self, data: DataBundle) -> pd.DataFrame:
        """Return the rows this control is accountable for.

        Defaults to every user account. Override when a control only applies to
        a subset (active accounts, privileged accounts, accounts with log
        activity, ...) so the reported exception rate stays honest.

        Args:
            data: The loaded data bundle.

        Returns:
            A DataFrame whose row count becomes ``total_tested``.
        """
        return data.user_accounts

    @abstractmethod
    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Identify policy breaches.

        Args:
            data: The loaded data bundle.

        Returns:
            A DataFrame of exceptions. Implementations must include a
            human-readable ``reason`` column; the mandatory identity columns are
            back-filled by :func:`normalise_exceptions` if omitted.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Template method
    # ------------------------------------------------------------------ #
    def run(self, data: DataBundle) -> ControlResult:
        """Execute the control and package the outcome as evidence.

        Any exception raised by :meth:`evaluate` is captured and returned as an
        ``ERROR`` result rather than propagating, so one broken control cannot
        abort an entire monitoring run.

        Args:
            data: The loaded data bundle.

        Returns:
            A :class:`ControlResult`.
        """
        started = time.perf_counter()
        executed_at = datetime.now(timezone.utc)

        try:
            population = self.population(data)
            total_tested = int(len(population))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("%s: population could not be determined", self.control_id)
            return self._error_result(executed_at, started, f"Population error: {exc}")

        try:
            raw = self.evaluate(data)
            exceptions = normalise_exceptions(raw)
        except Exception as exc:
            logger.exception("%s: evaluation failed", self.control_id)
            return self._error_result(executed_at, started, f"{type(exc).__name__}: {exc}", total_tested)

        # Back-fill the control's severity onto rows that did not set one.
        if "severity" in exceptions.columns:
            exceptions["severity"] = exceptions["severity"].where(exceptions["severity"].notna(), self.severity)
        else:  # pragma: no cover - normalise_exceptions always creates it
            exceptions["severity"] = self.severity

        exception_count = int(len(exceptions))
        exception_rate = round(exception_count / total_tested, 6) if total_tested else 0.0
        status = ControlStatus.FAIL.value if exception_count else ControlStatus.PASS.value

        result = ControlResult(
            control_id=self.control_id,
            status=status,
            total_tested=total_tested,
            exception_count=exception_count,
            exception_rate=exception_rate,
            exceptions=exceptions,
            executed_at=executed_at,
            parameters_used=dict(self.parameters),
            summary_text=self._summarise(total_tested, exception_count, exception_rate),
            name=self.name,
            description=self.description,
            risk_category=self.risk_category,
            severity=self.severity,
            framework_mapping=self.framework_mapping,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        logger.info(
            "%s %s: %d/%d exception(s) (%.2f%%)",
            self.control_id,
            status,
            exception_count,
            total_tested,
            exception_rate * 100,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _error_result(
        self,
        executed_at: datetime,
        started: float,
        message: str,
        total_tested: int = 0,
    ) -> ControlResult:
        """Build an ``ERROR`` result so a failure is visible, not swallowed."""
        return ControlResult(
            control_id=self.control_id,
            status=ControlStatus.ERROR.value,
            total_tested=total_tested,
            exception_count=0,
            exception_rate=0.0,
            exceptions=empty_exceptions(),
            executed_at=executed_at,
            parameters_used=dict(self.parameters),
            summary_text=f"Control could not be evaluated: {message}",
            name=self.name,
            description=self.description,
            risk_category=self.risk_category,
            severity=self.severity,
            framework_mapping=self.framework_mapping,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=message,
        )

    def _summarise(self, total_tested: int, exception_count: int, exception_rate: float) -> str:
        """Compose the one-line finding shown in the CLI and dashboard."""
        if total_tested == 0:
            return "No records in scope - control not applicable for this dataset."
        if exception_count == 0:
            return f"Tested {total_tested:,} record(s); no exceptions identified. Control operating effectively."
        return (
            f"Tested {total_tested:,} record(s); {exception_count:,} exception(s) identified "
            f"({exception_rate * 100:.2f}% deviation rate). Remediation required."
        )
