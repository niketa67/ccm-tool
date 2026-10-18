"""CC-05 - Stale Privileged Access Reviews.

Access certification is the compensating control for over-provisioning: if the
recertification never happens, every other access control degrades over time.
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
class StalePrivilegedAccessReviews(BaseControl):
    """Flag privileged accounts whose access review has lapsed."""

    control_id = "CC-05"
    name = "Stale Privileged Access Reviews"
    description = (
        "Privileged entitlements must be recertified on a recurring cadence. Flags privileged "
        "accounts whose last access review is older than the review interval, or that have never "
        "been reviewed."
    )
    risk_category = "Access Certification"
    severity = "High"
    framework_mapping = (
        "ISO 27001 A.5.18 (Access Rights Review)",
        "SOX ITGC - Access Recertification",
        "NIST SP 800-53 AC-2(j) (Review of Account Usage)",
    )
    default_parameters: dict[str, Any] = {
        "review_interval_days": 180,
        "privileged_account_types": ["Privileged"],
        "flag_never_reviewed": True,
        "active_accounts_only": True,
    }

    def population(self, data: DataBundle) -> pd.DataFrame:
        """Active accounts of a type requiring periodic recertification."""
        accounts = data.user_accounts
        if accounts.empty:
            return accounts

        scoped = accounts
        if bool(self.parameters.get("active_accounts_only", True)):
            scoped = scoped.loc[scoped["status"].astype("string").str.casefold().eq("active")]

        types = {str(value).casefold() for value in (self.parameters.get("privileged_account_types") or [])}
        if types:
            scoped = scoped.loc[scoped["account_type"].astype("string").str.casefold().isin(types)]
        return scoped

    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Return accounts whose last review is older than the interval."""
        population = self.population(data)
        if population.empty:
            return empty_exceptions()

        interval = int(self.parameters.get("review_interval_days", 180))
        flag_never = bool(self.parameters.get("flag_never_reviewed", True))

        reviewed = pd.to_datetime(population["last_access_review_date"], errors="coerce")
        days_since = (data.as_of.normalize() - reviewed).dt.days
        never_reviewed = reviewed.isna()

        breached = days_since.gt(interval) | (never_reviewed if flag_never else False)
        hits = population.loc[breached].copy()
        if hits.empty:
            return empty_exceptions()

        hits["days_since_review"] = days_since.loc[hits.index]
        hits["last_review_display"] = reviewed.loc[hits.index].dt.strftime("%Y-%m-%d")

        reasons: list[str] = []
        for _, row in hits.iterrows():
            last = row.get("last_review_display")
            if pd.isna(last):
                reasons.append(
                    f"Privileged account in {row['system']} has never been recertified; "
                    f"review interval is {interval} day(s)"
                )
            else:
                reasons.append(
                    f"Privileged access in {row['system']} last recertified on {last}, "
                    f"{int(row['days_since_review'])} day(s) ago (interval {interval} day(s))"
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
                "last_review_display",
                "days_since_review",
                "reason",
            ],
        ]
