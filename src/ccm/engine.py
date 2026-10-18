"""Engine that executes control checks against a loaded data bundle.

Two guarantees shape the design:

1. **Isolation.** One control raising an exception must not abort the monitoring
   run. Each control runs inside its own try/except and reports an ``ERROR``
   status instead.
2. **Predictability.** Controls are executed in stable ``control_id`` order, so
   repeated runs against identical data produce identical evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from .config import ControlsConfig, load_controls_config, severity_rank
from .controls.base import ControlResult, ControlStatus
from .controls.registry import build_controls
from .loader import DataBundle

logger = logging.getLogger(__name__)

__all__ = ["RunSummary", "run_controls", "engine_effectiveness"]


@dataclass
class RunSummary:
    """Aggregate outcome of one full monitoring run.

    Attributes:
        run_id: Identifier assigned when the run is persisted (``None`` until
            then); the CLI and dashboard use it to join results to the run.
        executed_at: ISO timestamp of the run.
        as_of: Data reference date the controls evaluated against.
        data_hash: SHA-256 over the source files (evidence integrity seal).
        results: Per-control results in execution order.
        warnings: Data-quality warnings attached to the run.
        error_count: Number of controls that returned ``ERROR``.
    """

    executed_at: str
    as_of: str
    data_hash: str
    results: list[ControlResult] = field(default_factory=list)
    run_id: int | None = None
    warnings: list[str] = field(default_factory=list)
    error_count: int = 0

    @property
    def total_controls(self) -> int:
        """Number of controls that were executed."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Count of controls that found no exceptions."""
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        """Count of controls that found at least one exception."""
        return sum(1 for result in self.results if result.status == ControlStatus.FAIL.value)

    @property
    def total_exceptions(self) -> int:
        """Total exception rows across all controls."""
        return sum(result.exception_count for result in self.results)

    @property
    def exceptions_frame(self) -> pd.DataFrame:
        """All exceptions from the run, tagged with control id and control name."""
        frames = [result.exceptions.assign(control_id=result.control_id, control=result.name) for result in self.results]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        tag_columns = [column for column in ("control_id", "control") if column not in combined.columns]
        return combined.loc[:, tag_columns + [column for column in combined.columns if column not in tag_columns]]

    @property
    def effectiveness(self) -> float:
        """Share of executed controls that passed, as a percentage."""
        if not self.results:
            return 0.0
        return round(100.0 * self.passed / len(self.results), 1)

    def as_table(self) -> pd.DataFrame:
        """One row per control, ready for ``to_string()`` or CSV output."""
        rows = [result.as_row() for result in sorted(self.results, key=lambda r: severity_rank(r.severity), reverse=True)]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, object]:
        """Flatten the summary for SQLite persistence."""
        return {
            "executed_at": self.executed_at,
            "as_of": self.as_of,
            "data_hash": self.data_hash,
            "total_controls": self.total_controls,
            "passed": self.passed,
            "failed": self.failed,
            "error_count": self.error_count,
            "total_exceptions": self.total_exceptions,
            "effectiveness": self.effectiveness,
            "warnings": list(self.warnings),
        }


def _run_single(control, data: DataBundle, *, re_raise: bool = False) -> ControlResult:
    """Execute one control, containing any failure at the control boundary."""
    try:
        return control.run(data)
    except Exception as exc:  # pragma: no cover - BaseControl.run already guards
        logger.exception("Control %s crashed unexpectedly", control.control_id)
        if re_raise:
            raise
        return ControlResult(
            control_id=control.control_id,
            status=ControlStatus.ERROR.value,
            total_tested=0,
            exception_count=0,
            exception_rate=0.0,
            exceptions=pd.DataFrame(),
            executed_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
            parameters_used=dict(getattr(control, "parameters", {})),
            summary_text=f"Control could not be evaluated: {type(exc).__name__}: {exc}",
            name=getattr(control, "name", control.control_id),
            description=getattr(control, "description", ""),
            risk_category=getattr(control, "risk_category", ""),
            severity=getattr(control, "severity", "Medium"),
            framework_mapping=tuple(getattr(control, "framework_mapping", ())),
            error=f"{type(exc).__name__}: {exc}",
        )


def run_controls(
    data: DataBundle,
    config: ControlsConfig | None = None,
    *,
    only: list[str] | None = None,
    include_disabled: bool = False,
) -> RunSummary:
    """Execute every enabled control against ``data``.

    Args:
        data: Loaded and validated data bundle.
        config: Config source. Defaults to a fresh load of ``controls.yaml``.
        only: Optional whitelist of control ids to execute.
        include_disabled: When ``True``, run controls that config disables.

    Returns:
        A :class:`RunSummary`.
    """
    config = config or load_controls_config()
    selected = list(only) if only else None

    controls = build_controls(config, only=selected, include_disabled=include_disabled)
    results: list[ControlResult] = []

    for control in controls:
        result = _run_single(control, data)
        results.append(result)

    results.sort(key=lambda r: (r.control_id,))

    summary = RunSummary(
        executed_at=pd.Timestamp.now(tz="UTC").isoformat(),
        as_of=pd.Timestamp(data.as_of).isoformat(),
        data_hash=data.data_hash,
        results=results,
        warnings=list(data.warnings),
    )
    summary.error_count = sum(1 for result in results if result.status == ControlStatus.ERROR.value)
    logger.info(
        "Run complete: %d control(s), %d PASS / %d FAIL / %d ERROR, %d exception(s)",
        summary.total_controls,
        summary.passed,
        summary.failed,
        summary.error_count,
        summary.total_exceptions,
    )
    return summary


def engine_effectiveness(summary: RunSummary) -> float:
    """Return the run's control effectiveness percentage."""
    return summary.effectiveness
