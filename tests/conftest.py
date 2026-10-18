"""Shared fixtures: tiny hand-crafted data bundles and helper builders.

Controls are unit-tested against small, explicit DataFrames so that every
assertion traces back to a human-written input - no surprises from a generator.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import pytest

from ccm.loader import DataBundle

#: A stable "today" so age-based assertions are deterministic.
AS_OF = pd.Timestamp("2026-06-01")


def hr_frame(**overrides: Any) -> pd.DataFrame:
    """Return a three-row HR master table; override any column with a mapping."""
    employees = pd.DataFrame(
        {
            "employee_id": ["E1", "E2", "E3", "E4"],
            "name": ["Ann Leaver", "Bob Active", "Cara Onleave", "Dave Gone"],
            "email": ["ann@x.com", "bob@x.com", "cara@x.com", "dave@x.com"],
            "department": ["Finance", "IT", "Sales", "IT"],
            "job_title": ["Analyst", "Engineer", "Analyst", "Engineer"],
            "manager_id": ["E2", None, None, None],
            "status": ["Terminated", "Active", "On Leave", "Terminated"],
            "hire_date": ["2020-01-01", "2019-01-01", "2021-01-01", "2018-01-01"],
            "termination_date": ["2026-04-01", None, None, "2026-01-01"],
        }
    )
    for column, values in overrides.items():
        employees[column] = values
    return employees


def accounts_frame(**overrides: Any) -> pd.DataFrame:
    """Return a small user-account table; override any column with a mapping."""
    accounts = pd.DataFrame(
        {
            "account_id": ["A1", "A2", "A3", "A4", "A5"],
            "username": ["ann.leaver", "bob.active", "cara.onleave", "dave.gone", "admin"],
            "employee_id": ["E1", "E2", "E3", "E4", None],
            "system": ["ERP", "ERP", "CRM", "AWS", "ERP"],
            "account_type": ["Standard", "Standard", "Standard", "Privileged", "Shared"],
            "status": ["Active", "Active", "Active", "Active", "Active"],
            "mfa_enabled": [True, True, True, False, True],
            "created_date": ["2020-02-01", "2019-02-01", "2021-02-01", "2018-02-01", "2020-02-01"],
            "last_login_date": ["2026-04-10", "2026-05-20", "2026-05-20", "2026-05-20", "2026-05-20"],
            "last_password_change": ["2026-02-01", "2026-02-01", "2026-02-01", "2026-02-01", "2026-02-01"],
            "last_access_review_date": ["2026-05-01", "2026-05-01", "2026-05-01", "2025-11-01", "2026-05-01"],
        }
    )
    for column, values in overrides.items():
        accounts[column] = values
    return accounts


def assignments_frame(rows: list[tuple[str, str, str]] | None = None) -> pd.DataFrame:
    """Return a role-assignment table.

    Args:
        rows: Tuples of ``(account_id, role_name, system)``.
    """
    rows = rows or []
    return pd.DataFrame(
        {
            "account_id": [row[0] for row in rows],
            "role_name": [row[1] for row in rows],
            "system": [row[2] for row in rows],
            "granted_date": ["2025-01-01"] * len(rows),
            "granted_by": ["SEC-TEAM"] * len(rows),
        }
    )


def login_logs_frame(rows: list[tuple[str, str, bool]] | None = None) -> pd.DataFrame:
    """Return a login-log table.

    Args:
        rows: Tuples of ``(account_id, timestamp, success)``.
    """
    rows = rows or []
    return pd.DataFrame(
        {
            "log_id": [f"L{index + 1}" for index in range(len(rows))],
            "account_id": [row[0] for row in rows],
            "timestamp": [row[1] for row in rows],
            "success": [row[2] for row in rows],
            "source_ip": ["10.0.0.1"] * len(rows),
            "is_after_hours": [False] * len(rows),
        }
    )


def bundle(
    *,
    hr: pd.DataFrame | None = None,
    accounts: pd.DataFrame | None = None,
    assignments: pd.DataFrame | None = None,
    logs: pd.DataFrame | None = None,
    as_of: pd.Timestamp = AS_OF,
) -> DataBundle:
    """Assemble a :class:`DataBundle` with sensible small defaults."""
    tables = {
        "hr_employees": hr if hr is not None else hr_frame(),
        "user_accounts": accounts if accounts is not None else accounts_frame(),
        "access_assignments": assignments if assignments is not None else assignments_frame(),
        "login_logs": logs if logs is not None else login_logs_frame(),
    }
    # Digest the in-memory snapshot the same way the loader digests CSV files, so
    # persistence tests can still assert real evidence-integrity hashes.
    digest = hashlib.sha256()
    for name, frame in sorted(tables.items()):
        digest.update(name.encode("utf-8") + b"\x00")
        digest.update(frame.to_csv(index=False).encode("utf-8") + b"\x1e")
    return DataBundle(
        **tables,
        as_of=as_of,
        data_hash=digest.hexdigest(),
    )


@pytest.fixture(autouse=True)
def _project_env(tmp_path: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the package at a temp home so tests never touch real artefacts."""
    # ``CCM_HOME`` must be set before paths are evaluated; import it lazily here.
    from ccm import paths as project_paths

    monkeypatch.setattr(project_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(project_paths, "RAW_DATA_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(project_paths, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(project_paths, "CONTROLS_CONFIG_PATH", project_paths.PROJECT_ROOT / "config" / "controls.yaml")
    monkeypatch.setattr(project_paths, "SOD_RULES_PATH", project_paths.PROJECT_ROOT / "config" / "sod_rules.yaml")


@pytest.fixture()
def real_config():
    """The real ``controls.yaml`` as a :class:`ControlsConfig`."""
    from ccm.config import load_controls_config

    return load_controls_config()
