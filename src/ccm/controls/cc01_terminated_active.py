"""CC-01 - Terminated Employee Active Accounts.

Leavers are the single most common source of logical-access findings in ITGC
testing: HR terminates the employee, but nobody tells the IAM platform. This
control reconciles the HR master against the account directory.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..loader import DataBundle
from .base import BaseControl, empty_exceptions
from .registry import register_control

logger = logging.getLogger(__name__)


@register_control
class TerminatedEmployeeActiveAccounts(BaseControl):
    """Flag accounts belonging to leavers that are still usable."""

    control_id = "CC-01"
    name = "Terminated Employee Active Accounts"
    description = (
        "Employees who have left the organisation must not retain logical access. "
        "Identifies terminated employees whose accounts remain Active, or whose accounts "
        "show authentication activity after the termination date."
    )
    risk_category = "Logical Access Management"
    severity = "Critical"
    framework_mapping = (
        "ISO 27001 A.5.18 (Access Rights)",
        "SOX ITGC - Logical Access / User Deprovisioning",
        "NIST SP 800-53 AC-2(3) (Disable Accounts)",
    )
    default_parameters: dict[str, Any] = {
        "terminated_statuses": ["Terminated"],
        "flag_post_termination_login": True,
        "post_termination_grace_days": 0,
    }

    def population(self, data: DataBundle) -> pd.DataFrame:
        """Accounts that belong to an employee flagged as terminated in HR."""
        terminated = self._terminated_employees(data)
        if terminated.empty:
            return data.user_accounts.iloc[0:0]
        return data.user_accounts.loc[data.user_accounts["employee_id"].isin(set(terminated["employee_id"]))]

    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Return one row per leaver account that breaches policy."""
        terminated = self._terminated_employees(data)
        if terminated.empty:
            logger.debug("CC-01: no terminated employees in HR extract")
            return empty_exceptions()

        accounts = data.user_accounts
        if accounts.empty:
            return empty_exceptions()

        merged = accounts.merge(
            terminated[["employee_id", "name", "termination_date"]],
            on="employee_id",
            how="inner",
        )
        if merged.empty:
            return empty_exceptions()

        grace_days = int(self.parameters.get("post_termination_grace_days", 0) or 0)
        flag_post_login = bool(self.parameters.get("flag_post_termination_login", True))

        is_active = merged["status"].astype("string").str.casefold().eq("active")
        terminated_on = pd.to_datetime(merged["termination_date"], errors="coerce")
        last_login = pd.to_datetime(merged["last_login_date"], errors="coerce")
        login_cutoff = terminated_on + pd.to_timedelta(grace_days, unit="D")
        logged_in_after = last_login.notna() & terminated_on.notna() & last_login.gt(login_cutoff)

        hits = merged.loc[is_active | (logged_in_after if flag_post_login else False)].copy()
        if hits.empty:
            return empty_exceptions()

        reasons: list[str] = []
        for _, row in hits.iterrows():
            reason = self._reason(row, is_active=bool(row["status"] and str(row["status"]).casefold() == "active"))
            reasons.append(reason)
        hits["reason"] = reasons

        leaves = pd.to_datetime(hits["termination_date"], errors="coerce")
        hits["termination_date"] = leaves.dt.strftime("%Y-%m-%d")
        hits["days_since_termination"] = (data.as_of.normalize() - leaves).dt.days
        hits["last_login_date"] = pd.to_datetime(hits["last_login_date"], errors="coerce").dt.strftime("%Y-%m-%d")

        return hits.loc[
            :,
            [
                "account_id",
                "username",
                "employee_id",
                "system",
                "account_type",
                "status",
                "termination_date",
                "days_since_termination",
                "last_login_date",
                "reason",
            ],
        ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _terminated_employees(self, data: DataBundle) -> pd.DataFrame:
        """HR rows whose status marks them as no longer employed."""
        statuses = {str(value).casefold() for value in (self.parameters.get("terminated_statuses") or [])}
        if not statuses:
            return data.hr_employees.iloc[0:0]
        status = data.hr_employees["status"].astype("string").str.casefold()
        return data.hr_employees.loc[status.isin(statuses)]

    @staticmethod
    def _reason(row: pd.Series, *, is_active: bool) -> str:
        """Compose a human-readable reason for a single leaver-account breach."""
        termination = row.get("termination_date")
        termination_text = (
            pd.Timestamp(termination).strftime("%Y-%m-%d") if pd.notna(termination) else "an unknown date"
        )
        employee = row.get("name") or row.get("employee_id")
        parts: list[str] = []

        if is_active:
            parts.append(
                f"Employee {employee} terminated on {termination_text} but the account is still Active "
                f"in {row.get('system')}"
            )

        last_login = row.get("last_login_date")
        if pd.notna(last_login) and pd.notna(termination):
            days_after = (pd.Timestamp(last_login) - pd.Timestamp(termination)).days
            if days_after > 0:
                parts.append(
                    f"account authenticated {days_after} day(s) after termination "
                    f"(last login {pd.Timestamp(last_login).strftime('%Y-%m-%d')})"
                )

        return "; ".join(parts) if parts else "Leaver account retained access after termination"
