"""Tests for SQLite persistence and configuration loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccm.config import (
    ConfigError,
    ControlConfig,
    ControlsConfig,
    load_controls_config,
    load_sod_rules,
)
from ccm.engine import run_controls
from ccm.storage import EXCEPTION_STATUSES, Storage, cell_for_db

from conftest import bundle


@pytest.fixture()
def store(tmp_path: Path) -> Storage:
    """A :class:`Storage` bound to a throwaway database."""
    return Storage(tmp_path / "test_ccm.db")


@pytest.fixture()
def populated_store(store: Storage) -> Storage:
    """A store that already contains one persisted run."""
    summary = run_controls(bundle())
    store.save_run(summary)
    return store


class TestStorage:
    """Persistence behaviour."""

    def test_save_and_retrieve_run(self, populated_store: Storage) -> None:
        """A run that is saved is retrievable with correct aggregate figures."""
        run_id = populated_store.latest_run_id()
        assert run_id is not None
        run = populated_store.get_run(run_id)
        assert run is not None
        assert run["total_controls"] == 8
        assert run["passed"] + run["failed"] + run["error_count"] == 8
        assert run["total_exceptions"] > 0
        assert len(run["data_snapshot_hash"]) == 64  # SHA-256 hex digest

    def test_results_link_to_run(self, populated_store: Storage) -> None:
        """Control results are joined to their parent run."""
        run_id = populated_store.latest_run_id()
        results = populated_store.get_results(run_id)
        assert len(results) == 8
        assert set(results["run_id"]) == {run_id}
        assert set(results["status"]).issubset({"PASS", "FAIL", "ERROR"})
        assert "parameters_json" in results.columns

    def test_exceptions_are_persisted_with_required_columns(
        self, populated_store: Storage
    ) -> None:
        """Exceptions carry identity columns and default to 'Open'."""
        exceptions = populated_store.get_exceptions()
        assert len(exceptions) > 0
        required = {"account_id", "username", "employee_id", "system", "reason", "status"}
        assert required.issubset(exceptions.columns)
        assert set(exceptions["status"]) == {"Open"}

    def test_update_exception_status_persists(self, populated_store: Storage) -> None:
        """Status changes are written back and remain after reload."""
        exception_id = int(populated_store.get_exceptions().iloc[0]["exception_id"])
        assert populated_store.update_exception_status(exception_id, "Acknowledged") is True

        # Reopen through a fresh connection to prove it hit the database.
        path = populated_store.path
        populated_store.close()
        reopened = Storage(path)
        row = reopened.get_exceptions()
        assert row.loc[row["exception_id"] == exception_id, "status"].iloc[0] == "Acknowledged"

    def test_invalid_status_is_rejected(self, populated_store: Storage) -> None:
        """An unknown workflow status raises rather than corrupting the data."""
        exception_id = int(populated_store.get_exceptions().iloc[0]["exception_id"])
        with pytest.raises(ValueError, match="Invalid status"):
            populated_store.update_exception_status(exception_id, "Sort-of-dealt-with")

    def test_workflow_statuses_are_supported(self, populated_store: Storage) -> None:
        """Every declared status is accepted by the API."""
        first = populated_store.get_exceptions().iloc[0]
        for status in EXCEPTION_STATUSES:
            assert populated_store.update_exception_status(int(first["exception_id"]), status) is True

    def test_multiple_runs_and_trend(self, populated_store: Storage) -> None:
        """Two runs are stored independently and the trend covers both."""
        summary = run_controls(bundle())
        populated_store.save_run(summary)
        assert populated_store.list_runs().shape[0] == 2

        trend = populated_store.exception_trend(limit=10)
        assert set(trend.columns).issuperset({"run_id", "control_id", "exception_count"})
        assert len(set(trend["run_id"])) == 2

    def test_filtering_by_control_and_status(self, populated_store: Storage) -> None:
        """Read queries honour their filters."""
        exceptions = populated_store.get_exceptions()
        target_control = exceptions.iloc[0]["control_id"]
        filtered = populated_store.get_exceptions(control_id=target_control)
        assert set(filtered["control_id"]) == {target_control}

        acknowledged = populated_store.get_exceptions(status="Acknowledged")
        assert (acknowledged.empty) or set(acknowledged["status"]) == {"Acknowledged"}

    def test_empty_database_returns_empty_frames(self, store: Storage) -> None:
        """A fresh database returns empty structures, not errors."""
        assert store.latest_run_id() is None
        assert store.list_runs().empty
        assert store.get_exceptions().empty
        assert store.exception_trend().empty

    def test_null_employee_id_is_written_as_null(self, store: Storage) -> None:
        """Missing employee ids persist as SQL NULL, not <NA> or nan."""
        summary = run_controls(
            bundle(
                accounts=pd.DataFrame(
                    {
                        "account_id": ["A1"],
                        "username": ["orphan"],
                        "employee_id": [None],
                        "system": ["ERP"],
                        "account_type": ["Standard"],
                        "status": ["Active"],
                        "mfa_enabled": [True],
                        "created_date": ["2025-01-01"],
                        "last_login_date": ["2026-06-01"],
                        "last_password_change": ["2025-01-01"],
                        "last_access_review_date": ["2025-01-01"],
                    }
                )
            )
        )
        store.save_run(summary)
        rows = store.get_exceptions()
        orphan_rows = rows.loc[rows["control_id"] == "CC-07"]
        if len(orphan_rows):
            assert orphan_rows["employee_id"].isna().all()


class TestCellConversion:
    """SQLite value coercion."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (pd.NA, None),
            (pd.NaT, None),
            (float("nan"), None),
            (None, None),
            (7, 7),
            (np.int64(42), 42),
            (1.5, 1.5),
            (pd.Timestamp("2026-06-01"), "2026-06-01T00:00:00"),
            ("text", "text"),
        ],
    )
    def test_conversion(self, value: object, expected: object) -> None:
        """Values round-trip into SQLite-safe primitives."""
        assert cell_for_db(value) == expected


class TestConfigLoading:
    """YAML-driven configuration."""

    def test_all_eight_controls_are_declared(self) -> None:
        """The demo config declares the full control catalogue."""
        config = load_controls_config()
        assert set(config.controls) == {
            "CC-01", "CC-02", "CC-03", "CC-04", "CC-05", "CC-06", "CC-07", "CC-08"
        }
        assert all(config.get(cid).enabled for cid in config.controls)

    def test_severities_are_valid(self) -> None:
        """Every severity must fall inside the accepted scale."""
        valid = {"Critical", "High", "Medium", "Low"}
        assert all(config.severity in valid for config in load_controls_config().controls.values())

    def test_thresholds_are_present_and_wired_to_controls(self) -> None:
        """Configured thresholds match the controls' defaults."""
        config = load_controls_config()
        assert config.get("CC-04").parameters["dormant_days"] >= 1
        assert config.get("CC-05").parameters["review_interval_days"] >= 1
        assert config.get("CC-08").parameters["max_failed_attempts"] >= 1

    def test_duration_threshold_default_is_ninety_days(self) -> None:
        """CC-04's documented default is 90 days."""
        assert load_controls_config().get("CC-04").parameters["dormant_days"] == 90

    def test_overrides_change_run_behaviour(self, real_config) -> None:
        """A config change flows through to the control's parameters."""
        real_config.controls["CC-04"] = ControlConfig(
            control_id="CC-04",
            enabled=True,
            severity=real_config.get("CC-04").severity,
            parameters={"dormant_days": 1},
        )
        summary = run_controls(bundle(), real_config, only=["CC-04"])
        assert summary.results[0].parameters_used["dormant_days"] == 1

    def test_get_unknown_control_returns_none(self) -> None:
        """An undeclared control id yields ``None`` rather than raising."""
        assert load_controls_config().get("CC-99") is None

    def test_disabled_controls_are_excluded(self) -> None:
        """``enabled: false`` moves the control out of the enabled list."""
        config = load_controls_config()
        mutated = ControlsConfig(
            controls={cid: ControlConfig(control_id=cid, enabled=(cid != "CC-04")) for cid in config.controls}
        )
        assert "CC-04" not in mutated.enabled_controls()
        assert "CC-01" in mutated.enabled_controls()

    def test_sod_rules_load(self) -> None:
        """The SoD catalogue is readable and well formed."""
        rules = load_sod_rules()
        assert len(rules) >= 3
        assert all(rule.role_a and rule.role_b for rule in rules)
        assert any(rule.system == "*" for rule in rules)

    def test_missing_controls_file_raises(self, tmp_path: Path) -> None:
        """A missing config file produces a helpful error."""
        with pytest.raises(ConfigError, match="not found"):
            load_controls_config(tmp_path / "missing.yaml")

    def test_invalid_severity_raises(self, tmp_path: Path) -> None:
        """An out-of-range severity is rejected at load time."""
        path = tmp_path / "bad.yaml"
        path.write_text(
            "controls:\n  CC-04:\n    enabled: true\n    severity: Extreme\n    parameters: {}\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="invalid severity"):
            load_controls_config(path)

    def test_missing_enabled_key_raises(self, tmp_path: Path) -> None:
        """A control block without ``enabled`` is rejected."""
        path = tmp_path / "bad2.yaml"
        path.write_text("controls:\n  CC-04:\n    severity: High\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="missing keys"):
            load_controls_config(path)

    def test_non_mapping_parameters_raises(self, tmp_path: Path) -> None:
        """``parameters`` must be a mapping."""
        path = tmp_path / "bad3.yaml"
        path.write_text("controls:\n  CC-04:\n    enabled: true\n    parameters: [1,2]\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="parameters"):
            load_controls_config(path)


class TestParametersSerialised:
    """Parameters are stored as JSON for traceability."""

    def test_parameters_json_round_trips(self, populated_store: Storage) -> None:
        """Stored parameters can be parsed back into a dict."""
        results = populated_store.get_results()
        params = json.loads(results.iloc[0]["parameters_json"])
        assert isinstance(params, dict)
