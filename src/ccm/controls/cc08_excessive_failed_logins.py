"""CC-08 - Excessive Failed Logins (Brute Force).

Detects a burst of authentication failures against one account inside a rolling
window. Implemented as an exact sliding window over each account's failure
timeline rather than a bucketed count, so a burst straddling a bucket boundary
is still caught.
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
class ExcessiveFailedLogins(BaseControl):
    """Flag accounts with more than N failures inside a rolling window."""

    control_id = "CC-08"
    name = "Excessive Failed Logins (Brute Force)"
    description = (
        "Repeated authentication failures from a single account in a short window indicate "
        "password-spraying or brute-force activity, or a misconfigured integration. Each "
        "qualifying burst is reported as an exception."
    )
    risk_category = "Security Monitoring"
    severity = "Medium"
    framework_mapping = (
        "ISO 27001 A.8.16 (Monitoring Activities)",
        "SOX ITGC - Logical Access",
        "NIST SP 800-53 AC-7 (Unsuccessful Logon Attempts)",
    )
    default_parameters: dict[str, Any] = {
        "max_failed_attempts": 5,
        "window_hours": 1,
        "consecutive_failures_only": False,
        "lookback_days": 30,
    }

    def _recent_logs(self, data: DataBundle) -> pd.DataFrame:
        """Return login events inside the configured lookback window."""
        logs = data.login_logs
        if logs.empty:
            return logs
        timestamp = pd.to_datetime(logs["timestamp"], errors="coerce")
        recent = logs.loc[timestamp.notna()].copy()
        recent["timestamp"] = timestamp.loc[recent.index]

        lookback_days = int(self.parameters.get("lookback_days", 30) or 0)
        if lookback_days > 0:
            cutoff = data.as_of - pd.Timedelta(days=lookback_days)
            recent = recent.loc[recent["timestamp"] >= cutoff]
        return recent.sort_values(["account_id", "timestamp"], kind="stable")

    def population(self, data: DataBundle) -> pd.DataFrame:
        """Accounts with at least one authentication event in the lookback window."""
        recent = self._recent_logs(data)
        accounts = data.user_accounts
        if recent.empty or accounts.empty:
            return accounts.iloc[0:0]
        with_activity = set(recent["account_id"].dropna().astype(str))
        return accounts.loc[accounts["account_id"].astype(str).isin(with_activity)]

    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Return accounts with a failure burst above the threshold."""
        recent = self._recent_logs(data)
        if recent.empty:
            return empty_exceptions()

        max_attempts = int(self.parameters.get("max_failed_attempts", 5))
        window = pd.Timedelta(hours=int(self.parameters.get("window_hours", 1)))
        consecutive_only = bool(self.parameters.get("consecutive_failures_only", False))

        if consecutive_only:
            # A success breaks the streak, so consecutive failures are grouped by
            # the running count of successes seen so far for that account.
            ordered = recent.sort_values(["account_id", "timestamp"], kind="stable")
            succeeded = ordered["success"].eq(True).fillna(False).astype(bool)
            ordered = ordered.assign(_streak=succeeded.groupby(ordered["account_id"], sort=False).cumsum())
            failures = ordered.loc[ordered["success"].eq(False)].copy()
            group_keys = ["account_id", "_streak"]
        else:
            # ``success`` is a nullable boolean: pd.NA is not evidence of failure,
            # so only an explicit False counts against the account.
            failures = recent.loc[recent["success"].eq(False)].copy()
            group_keys = ["account_id"]

        if failures.empty:
            return empty_exceptions()

        account_index = data.user_accounts.set_index(data.user_accounts["account_id"].astype(str), drop=False)
        rows: list[dict[str, Any]] = []

        for key, group in failures.groupby(group_keys, sort=False, dropna=False):
            account_id = key[0] if isinstance(key, tuple) else key
            times = group["timestamp"].sort_values().reset_index(drop=True)
            best_count, best_start, best_end = self._max_in_window(times, window)
            if best_count <= max_attempts:
                continue

            in_window = group.loc[(group["timestamp"] >= best_start) & (group["timestamp"] <= best_end)]
            ips = sorted({str(ip) for ip in in_window["source_ip"].dropna()}) if "source_ip" in in_window else []
            account = account_index.loc[account_id] if account_id in account_index.index else None

            rows.append(
                {
                    "account_id": account_id,
                    "username": account.get("username") if account is not None else pd.NA,
                    "employee_id": account.get("employee_id") if account is not None else pd.NA,
                    "system": account.get("system") if account is not None else pd.NA,
                    "account_type": account.get("account_type") if account is not None else pd.NA,
                    "status": account.get("status") if account is not None else pd.NA,
                    "failed_attempts": best_count,
                    "threshold": max_attempts,
                    "window_start": best_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "window_end": best_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_ips": ", ".join(ips[:5]) + (" ..." if len(ips) > 5 else ""),
                    "severity": self.severity,
                    "reason": (
                        f"{best_count} failed authentication attempts between "
                        f"{best_start.strftime('%Y-%m-%d %H:%M')} and {best_end.strftime('%H:%M')} "
                        f"(threshold: more than {max_attempts} in "
                        f"{int(window.total_seconds() // 3600)}h)"
                    ),
                }
            )

        if not rows:
            return empty_exceptions()

        return pd.DataFrame(rows).sort_values("failed_attempts", ascending=False).reset_index(drop=True)

    @staticmethod
    def _max_in_window(times: pd.Series, window: pd.Timedelta) -> tuple[int, Any, Any]:
        """Find the densest cluster of timestamps within a rolling window.

        Uses a two-pointer sweep over the sorted timestamps: O(n) per account.

        Args:
            times: Sorted timestamps of failure events.
            window: Maximum width of the window.

        Returns:
            Tuple of ``(count, window_start, window_end)`` for the densest
            cluster. ``(0, NaT, NaT)`` when ``times`` is empty.
        """
        if len(times) == 0:
            return 0, pd.NaT, pd.NaT

        values = times.to_numpy()
        best_count, best_start, best_end = 0, values[0], values[0]
        left = 0
        for right in range(len(values)):
            while values[right] - values[left] > window:
                left += 1
            count = right - left + 1
            if count > best_count:
                best_count, best_start, best_end = count, values[left], values[right]
        return best_count, pd.Timestamp(best_start), pd.Timestamp(best_end)
