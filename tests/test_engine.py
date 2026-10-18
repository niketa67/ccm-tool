"""Tests for the engine: orchestration, isolation and summary maths."""

from __future__ import annotations

import pandas as pd
import pytest

from ccm.engine import run_controls
from ccm.controls.base import BaseControl
from ccm.loader import DataBundle

from conftest import bundle


class _ExplodingControl(BaseControl):
    """A stand-in control whose ``evaluate`` always raises."""

    control_id = "XX-99"
    name = "Exploding Control"
    description = "always raises"
    risk_category = "Test"
    severity = "High"
    framework_mapping = ()
    default_parameters: dict = {}

    def __init__(self, **kwargs) -> None:
        """Accept config kwargs like a real control."""
        super().__init__(
            severity=kwargs.get("severity"),
            parameters=kwargs.get("parameters"),
        )

    def population(self, data: DataBundle) -> pd.DataFrame:
        """Return the account table."""
        return data.user_accounts

    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Deliberately raise to prove isolation."""
        raise RuntimeError("this control is intentionally broken")


class _CleanControl(BaseControl):
    """A stand-in control that always passes."""

    control_id = "XX-01"
    name = "Clean Control"
    description = "always passes"
    risk_category = "Test"
    severity = "Low"
    framework_mapping = ("TEST-1",)
    default_parameters: dict = {}

    def __init__(self, **kwargs) -> None:
        """Accept config kwargs like a real control."""
        super().__init__(
            severity=kwargs.get("severity"),
            parameters=kwargs.get("parameters"),
        )

    def population(self, data: DataBundle) -> pd.DataFrame:
        """Return the account table."""
        return data.user_accounts

    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Return no exceptions."""
        return pd.DataFrame(columns=["account_id", "username", "employee_id", "system", "reason"])


@pytest.fixture()
def two_control_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the two stand-in controls for the duration of a test."""
    from ccm.controls import registry

    monkeypatch.setitem(registry._REGISTRY, _ExplodingControl.control_id, _ExplodingControl)  # noqa: SLF001
    monkeypatch.setitem(registry._REGISTRY, _CleanControl.control_id, _CleanControl)  # noqa: SLF001


class TestEngine:
    """Engine behaviour."""

    def test_all_real_controls_run_and_produce_results(self, real_config) -> None:
        """Every built-in control yields exactly one result."""
        summary = run_controls(bundle(), real_config)
        assert len(summary.results) == 8
        assert [result.control_id for result in summary.results] == sorted(
            result.control_id for result in summary.results
        )
        assert all(result.total_tested >= 0 for result in summary.results)

    def test_failing_control_does_not_crash_the_run(self, real_config, two_control_registry) -> None:
        """A control that raises must surface as ERROR, not abort the run."""
        summary = run_controls(bundle(), real_config, only=["XX-99", "CC-01"])
        error_results = [result for result in summary.results if result.status == "ERROR"]
        assert len(error_results) >= 1
        assert "intentionally broken" in error_results[0].summary_text
        # The healthy controls still produced real outcomes alongside the error.
        assert any(result.status in ("PASS", "FAIL") for result in summary.results)

    def test_only_selected_controls_execute(self, real_config) -> None:
        """``only`` restricts execution to the requested controls."""
        summary = run_controls(bundle(), real_config, only=["CC-01"])
        assert [result.control_id for result in summary.results] == ["CC-01"]

    def test_config_severity_overrides_class_default(self, real_config) -> None:
        """A config severity must win over the class constant."""
        real_config.controls["CC-01"] = real_config.controls["CC-01"].__class__(
            control_id="CC-01",
            enabled=True,
            severity="Low",
            parameters=real_config.controls["CC-01"].parameters,
        )
        summary = run_controls(bundle(), real_config, only=["CC-01"])
        assert summary.results[0].severity == "Low"

    def test_summary_counts_are_consistent(self, real_config) -> None:
        """Passed + failed + errored equals the number of controls executed."""
        summary = run_controls(bundle(), real_config)
        assert summary.passed + summary.failed + summary.error_count == summary.total_controls
        assert summary.total_exceptions == sum(result.exception_count for result in summary.results)
        if summary.total_controls:
            expected = round(100 * summary.passed / summary.total_controls, 1)
            assert summary.effectiveness == expected

    def test_exceptions_frame_carries_identity_columns(self, real_config) -> None:
        """The combined exception frame keeps the mandatory evidence columns."""
        summary = run_controls(bundle(), real_config)
        if summary.total_exceptions:
            required = {"account_id", "username", "employee_id", "system", "reason", "control_id"}
            assert required.issubset(summary.exceptions_frame.columns)

    def test_exception_rows_have_readable_reasons(self, real_config) -> None:
        """Every exception must carry a non-empty reason string."""
        summary = run_controls(bundle(), real_config)
        for result in summary.results:
            if result.exception_count:
                reasons = result.exceptions["reason"].astype(str)
                assert reasons.str.len().gt(10).all(), f"{result.control_id} produced a blank reason"
