"""CC-04 - Dormant Accounts.

An active account nobody uses is pure attack surface: it will not be noticed if
it is abused, and it is a favourite persistence mechanism for attackers.
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
class DormantAccounts(BaseControl):
    """Flag active accounts idle for longer than the dormancy window."""

    control_id = "CC-04"
    name = "Dormant Accounts"
    description = (
        "Active accounts that have not authenticated within the dormancy window represent "
        "unnecessary attack surface and should be disabled or removed."
    )
    risk_category = "Logical Access Management"
    severity = "Medium"
    framework_mapping = (
        "ISO 27001 A.5.15 (Access Control)",
        "SOX ITGC - Logical Access",
        "NIST SP 800-53 AC-2(3) (Disable Inactive Accounts)",
    )
    default_parameters: dict[str, Any] = {
        "dormant_days": 90,
        "use_created_date_if_never_logged_in": True,
        "active_accounts_only": True,
        "exclude_account_types": ["Service"],
    }

    def population(self, data: DataBundle) -> pd.DataFrame:
        """Active accounts, excluding account types that need no interactive login."""
        accounts = data.user_accounts
        if accounts.empty:
            return accounts

        scoped = accounts
        if bool(self.parameters.get("active_accounts_only", True)):
            scoped = scoped.loc[scoped["status"].astype("string").str.casefold().eq("active")]

        excluded = {str(value).casefold() for value in (self.parameters.get("exclude_account_types") or [])}
        if excluded:
            scoped = scoped.loc[~scoped["account_type"].astype("string").str.casefold().isin(excluded)]
        return scoped

    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Return accounts whose idle period exceeds ``dormant_days``."""
        population = self.population(data)
        if population.empty:
            return empty_exceptions()

        dormant_days = int(self.parameters.get("dormant_days", 90))
        use_created = bool(self.parameters.get("use_created_date_if_never_logged_in", True))

        last_login = pd.to_datetime(population["last_login_date"], errors="coerce")
        created = pd.to_datetime(population["created_date"], errors="coerce")

        # Reference point for "idle since": last login when we have one,
        # otherwise the account creation date (a never-used account is idle
        # from the day it was provisioned).
        reference = last_login
        if use_created:
            reference = last_login.fillna(created)

        idle_days = (data.as_of.normalize() - reference).dt.days
        # A missing reference point means we cannot prove the account is in use,
        # so it is treated as maximally dormant rather than silently passing.
        never_used = last_login.isna() & created.isna()

        breached = idle_days.gt(dormant_days) | never_used
        hits = population.loc[breached].copy()
        if hits.empty:
            return empty_exceptions()

        hits["idle_days"] = idle_days.loc[hits.index]
        hits["last_login_display"] = last_login.loc[hits.index].dt.strftime("%Y-%m-%d")

        reasons: list[str] = []
        for _, row in hits.iterrows():
            if bool(never_used.loc[row.name] if row.name in never_used.index else False):
                reasons.append(
                    f"Active {row['account_type']} account in {row['system']} has no authentication history "
                    f"and no creation date to age against"
                )
                continue
            days = row["idle_days"]
            days_text = "unknown" if pd.isna(days) else f"{int(days)}"
            last = row.get("last_login_display")
            if pd.isna(last):
                reasons.append(
                    f"Active {row['account_type']} account in {row['system']} has never been used "
                    f"({days_text} day(s) since creation, threshold {dormant_days})"
                )
            else:
                reasons.append(
                    f"Active {row['account_type']} account in {row['system']} has not authenticated for "
                    f"{days_text} day(s) (last login {last}, threshold {dormant_days})"
                )
        hits["reason"] = reasons

        return hits.loc[
            :,
            [
                "account_id",
                "username",
                "employee_id",
                "system",
                "account_type",
                "status",
                "last_login_display",
                "idle_days",
                "reason",
            ],
        ]
