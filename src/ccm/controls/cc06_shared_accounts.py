"""CC-06 - Shared / Generic Account Usage.

Shared credentials make every downstream log worthless: you can see that the
account did something, but never which human did it. That breaks non-repudiation
and is a control expectation in every access-management framework.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from ..loader import DataBundle
from .base import BaseControl, empty_exceptions
from .registry import register_control

logger = logging.getLogger(__name__)


@register_control
class SharedOrGenericAccounts(BaseControl):
    """Flag active shared accounts and generic-named identities."""

    control_id = "CC-06"
    name = "Shared / Generic Account Usage"
    description = (
        "Shared or generic accounts break non-repudiation: activity cannot be attributed to a "
        "named individual. Active shared accounts are flagged for remediation (conversion to "
        "named accounts or a credential-vaulting solution)."
    )
    risk_category = "Identity & Non-Repudiation"
    severity = "Medium"
    framework_mapping = (
        "ISO 27001 A.5.16 (Identity Management)",
        "SOX ITGC - Logical Access",
        "NIST SP 800-53 IA-4 / AC-2 (Account Management)",
    )
    default_parameters: dict[str, Any] = {
        "shared_account_types": ["Shared"],
        "generic_username_patterns": [
            "(?i)^admin$",
            "(?i)^test",
            "(?i)^shared",
            "(?i)^svc_generic",
            "(?i)^generic",
            "(?i)^tmp",
        ],
        "active_accounts_only": True,
    }

    def population(self, data: DataBundle) -> pd.DataFrame:
        """Active accounts, the only ones that still expose live risk."""
        accounts = data.user_accounts
        if accounts.empty or not bool(self.parameters.get("active_accounts_only", True)):
            return accounts
        return accounts.loc[accounts["status"].astype("string").str.casefold().eq("active")]

    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Return active accounts that are shared by policy or by naming convention."""
        population = self.population(data)
        if population.empty:
            return empty_exceptions()

        shared_types = {str(value).casefold() for value in (self.parameters.get("shared_account_types") or [])}
        patterns = [re.compile(str(p)) for p in (self.parameters.get("generic_username_patterns") or [])]

        account_types = population["account_type"].astype("string").str.casefold()
        shared_by_type = account_types.isin(shared_types) if shared_types else pd.Series(False, index=population.index)

        usernames = population["username"].astype("string").fillna("")
        matched_patterns: list[str] = []
        shared_by_name = pd.Series(False, index=population.index)
        for pattern in patterns:
            matched = usernames.str.contains(pattern, regex=True, na=False)
            if matched.any():
                shared_by_name = shared_by_name | matched
                matched_patterns.append(pattern.pattern)

        hits = population.loc[shared_by_type | shared_by_name].copy()
        if hits.empty:
            return empty_exceptions()

        reasons: list[str] = []
        triggers: list[str] = []
        for _, row in hits.iterrows():
            is_shared_type = str(row["account_type"]).casefold() in shared_types
            name_hit = any(re.search(pattern, str(row["username"])) for pattern in matched_patterns)
            labels = []
            if is_shared_type:
                labels.append(f"account_type='{row['account_type']}'")
            if name_hit:
                labels.append(f"generic username '{row['username']}'")
            triggers.append(" and ".join(labels))
            reasons.append(
                f"Active {'shared' if is_shared_type else 'generic'} account in {row['system']} "
                f"({', '.join(labels)}); activity cannot be attributed to a named individual"
            )

        hits["trigger"] = triggers
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
                "trigger",
                "last_login_date",
                "reason",
            ],
        ]
