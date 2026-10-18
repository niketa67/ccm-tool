"""CC-07 - Orphan Accounts.

An account with no HR counterpart has no accountable owner. It is the classic
signature of a deprovisioning failure, a contractor record that was never
created, or an account somebody stood up and forgot about.
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
class OrphanAccounts(BaseControl):
    """Flag active accounts that cannot be tied to an HR record."""

    control_id = "CC-07"
    name = "Orphan Accounts"
    description = (
        "An active account that cannot be tied back to a current HR record has no accountable "
        "owner - a classic indicator of deprovisioning failure, a rogue account, or a contractor "
        "whose record was never created."
    )
    risk_category = "Logical Access Management"
    severity = "High"
    framework_mapping = (
        "ISO 27001 A.5.18 (Access Rights)",
        "SOX ITGC - Logical Access",
        "NIST SP 800-53 AC-2 (Account Management)",
    )
    default_parameters: dict[str, Any] = {
        "exclude_account_types": ["Service"],
        "allowlist_usernames": ["svc_generic"],
        "active_accounts_only": True,
    }

    def population(self, data: DataBundle) -> pd.DataFrame:
        """Active accounts excluding approved machine and allow-listed identities."""
        accounts = data.user_accounts
        if accounts.empty:
            return accounts

        scoped = accounts
        if bool(self.parameters.get("active_accounts_only", True)):
            scoped = scoped.loc[scoped["status"].astype("string").str.casefold().eq("active")]

        excluded = {str(value).casefold() for value in (self.parameters.get("exclude_account_types") or [])}
        if excluded:
            scoped = scoped.loc[~scoped["account_type"].astype("string").str.casefold().isin(excluded)]

        allowlist = {str(value).casefold() for value in (self.parameters.get("allowlist_usernames") or [])}
        if allowlist:
            scoped = scoped.loc[~scoped["username"].astype("string").str.casefold().isin(allowlist)]
        return scoped

    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Return active accounts with a missing or unresolvable employee link."""
        population = self.population(data)
        if population.empty:
            return empty_exceptions()

        known_employees = set(data.hr_employees["employee_id"].dropna().astype(str))
        employee_id = population["employee_id"].astype("string")
        blank = employee_id.isna() | employee_id.str.strip().eq("")
        # Two flavours of orphan: no employee reference at all, and a reference
        # that points at an employee HR has never heard of.
        dangling = ~blank & ~employee_id.isin(known_employees)

        hits = population.loc[blank | dangling].copy()
        if hits.empty:
            return empty_exceptions()

        reasons: list[str] = []
        for _, row in hits.iterrows():
            raw_id = row["employee_id"]
            if pd.isna(raw_id) or not str(raw_id).strip():
                reasons.append(
                    f"Active {row['account_type']} account in {row['system']} has no employee_id and "
                    f"is not covered by an approved service-account exception"
                )
            else:
                reasons.append(
                    f"Active {row['account_type']} account in {row['system']} references employee_id "
                    f"'{raw_id}', which does not exist in the HR master"
                )
        hits["reason"] = reasons

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
                "last_login_date",
                "reason",
            ],
        ]
