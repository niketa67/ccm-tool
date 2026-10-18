"""CC-03 - Privileged Accounts Without MFA.

Elevated entitlements plus single-factor authentication is the highest-impact
combination in the whole dataset, so this control carries a Critical rating and
sits at the top of the remediation queue.
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
class PrivilegedAccountsWithoutMfa(BaseControl):
    """Flag active privileged accounts that are not protected by MFA."""

    control_id = "CC-03"
    name = "Privileged Accounts Without MFA"
    description = (
        "Accounts with elevated entitlements are the highest-value target for credential theft. "
        "Every active privileged account must be protected by multi-factor authentication."
    )
    risk_category = "Privileged Access Management"
    severity = "Critical"
    framework_mapping = (
        "ISO 27001 A.8.5 (Secure Authentication)",
        "SOX ITGC - Logical Access",
        "NIST SP 800-53 IA-2(1) (Multi-Factor Authentication)",
    )
    default_parameters: dict[str, Any] = {
        "privileged_account_types": ["Privileged"],
        "include_service_accounts": False,
        "active_accounts_only": True,
    }

    def _scope_types(self) -> set[str]:
        """Account types treated as privileged for this control."""
        types = {str(value).casefold() for value in (self.parameters.get("privileged_account_types") or [])}
        if bool(self.parameters.get("include_service_accounts", False)):
            types.add("service")
        return types

    def population(self, data: DataBundle) -> pd.DataFrame:
        """Active accounts of a privileged account type."""
        accounts = self._scoped_accounts(data)
        scope = self._scope_types()
        if not scope:
            return accounts.iloc[0:0]
        return accounts.loc[accounts["account_type"].astype("string").str.casefold().isin(scope)]

    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Return privileged accounts with no MFA evidence."""
        population = self.population(data)
        if population.empty:
            return empty_exceptions()

        flag = population["mfa_enabled"]
        # ``boolean`` dtype keeps genuinely-unknown flags as pd.NA. A blank MFA
        # flag is not evidence of MFA, so it is treated as non-compliant but
        # labelled distinctly so the reader can see the difference.
        unenrolled = flag.eq(False)
        unknown = flag.isna()
        hits = population.loc[unenrolled | unknown].copy()
        if hits.empty:
            return empty_exceptions()

        hits["reason"] = [self._reason(row) for _, row in hits.iterrows()]

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
                "mfa_enabled",
                "last_login_date",
                "reason",
            ],
        ]

    def _scoped_accounts(self, data: DataBundle) -> pd.DataFrame:
        """Return accounts eligible for this control (active-only by default)."""
        accounts = data.user_accounts
        if accounts.empty or not bool(self.parameters.get("active_accounts_only", True)):
            return accounts
        return accounts.loc[accounts["status"].astype("string").str.casefold().eq("active")]

    @staticmethod
    def _reason(row: pd.Series) -> str:
        """Compose a human-readable reason, distinguishing 'disabled' from 'unknown'."""
        if pd.isna(row.get("mfa_enabled")):
            return (
                f"Active {row['account_type']} account has no MFA enrolment record in {row['system']}; "
                f"enrolment could not be evidenced"
            )
        return (
            f"Active {row['account_type']} account has MFA disabled in {row['system']}; "
            f"single-factor credentials grant elevated access"
        )
