"""CC-02 - Segregation of Duties Conflicts.

Evaluates each account's role bundle against the conflicting role pairs declared
in ``config/sod_rules.yaml``. The rule catalogue is data, not code, so risk can
extend coverage without a release.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import SodRule, load_sod_rules
from ..loader import DataBundle
from ..paths import PROJECT_ROOT
from .base import BaseControl, empty_exceptions
from .registry import register_control

logger = logging.getLogger(__name__)


@register_control
class SegregationOfDutiesConflicts(BaseControl):
    """Detect accounts holding both sides of a toxic role pair."""

    control_id = "CC-02"
    name = "Segregation of Duties Conflicts"
    description = (
        "A single user must not be able to both initiate and approve the same business activity. "
        "Evaluates every account's role assignments against a curated catalogue of conflicting "
        "role pairs (e.g. 'Payment Creator' + 'Payment Approver')."
    )
    risk_category = "Segregation of Duties"
    severity = "High"
    framework_mapping = (
        "ISO 27001 A.5.3 (Segregation of Duties)",
        "SOX ITGC - Segregation of Duties",
        "NIST SP 800-53 AC-5 (Separation of Duties)",
    )
    default_parameters: dict[str, Any] = {
        "sod_rules_file": "config/sod_rules.yaml",
        "case_sensitive": False,
        "active_accounts_only": True,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the control and load its SoD rule catalogue."""
        super().__init__(**kwargs)
        self._rules: list[SodRule] | None = None

    # ------------------------------------------------------------------ #
    # Rule loading
    # ------------------------------------------------------------------ #
    @property
    def rules(self) -> list[SodRule]:
        """The loaded SoD rules, resolved once and cached per instance."""
        if self._rules is None:
            self._rules = load_sod_rules(self._rules_path())
        return self._rules

    def _rules_path(self) -> Path:
        """Resolve ``sod_rules_file`` relative to the project root when needed."""
        configured = str(self.parameters.get("sod_rules_file") or "config/sod_rules.yaml")
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    # ------------------------------------------------------------------ #
    # Control logic
    # ------------------------------------------------------------------ #
    def population(self, data: DataBundle) -> pd.DataFrame:
        """Accounts that hold at least one role assignment."""
        accounts = self._scope_accounts(data)
        if accounts.empty or data.access_assignments.empty:
            return accounts
        with_roles = set(data.access_assignments["account_id"].dropna().astype(str))
        return accounts.loc[accounts["account_id"].astype(str).isin(with_roles)]

    def evaluate(self, data: DataBundle) -> pd.DataFrame:
        """Return one exception per (account, conflicting rule) pair."""
        if not self.rules:
            logger.warning("CC-02: no SoD rules configured - nothing to test")
            return empty_exceptions()

        assignments = data.access_assignments
        accounts = self._scope_accounts(data)
        if assignments.empty or accounts.empty:
            return empty_exceptions()

        case_sensitive = bool(self.parameters.get("case_sensitive", False))

        # Build account -> set(role) once; O(assignments) rather than O(accounts x rules).
        role_source = assignments["role_name"].astype("string").fillna("")
        if not case_sensitive:
            role_source = role_source.str.casefold()
        systems_source = assignments["system"].astype("string").fillna("")
        role_lookup: dict[str, set[str]] = {}
        system_lookup: dict[str, set[str]] = {}
        per_account = {
            account_id: group
            for account_id, group in pd.DataFrame(
                {
                    "account_id": assignments["account_id"].astype(str),
                    "role": role_source,
                    "system": systems_source,
                }
            ).groupby("account_id", sort=False)
        }
        for account_id, group in per_account.items():
            # Accumulate with set-comprehension rather than pandas `.agg`,
            # whose callables coerce sets to lists under PyArrow-backed dtypes.
            role_lookup[account_id] = {str(role) for role in group["role"]}
            system_lookup[account_id] = {str(system) for system in group["system"]}

        account_index = accounts.set_index(accounts["account_id"].astype(str), drop=False)

        def _key(role: str) -> str:
            return role if case_sensitive else role.casefold()

        rows: list[dict[str, Any]] = []
        for rule in self.rules:
            role_a, role_b = _key(rule.role_a), _key(rule.role_b)
            for account_id, roles in role_lookup.items():
                if role_a not in roles or role_b not in roles:
                    continue
                if account_id not in account_index.index:
                    continue  # account out of scope (e.g. disabled and control is active-only)

                account = account_index.loc[account_id]
                # Scope is judged on the systems where the suspect roles were
                # actually granted, falling back to the account's home system.
                granted_systems = set(system_lookup.get(account_id, {str(account["system"])}))
                if not any(rule.matches_system(system) for system in granted_systems):
                    continue

                rows.append(
                    {
                        "account_id": account_id,
                        "username": account.get("username"),
                        "employee_id": account.get("employee_id"),
                        "system": account.get("system"),
                        "account_type": account.get("account_type"),
                        "status": account.get("status"),
                        "rule_id": rule.rule_id,
                        "conflicting_roles": f"{rule.role_a} + {rule.role_b}",
                        "risk": rule.risk,
                        "rule_severity": rule.severity,
                        "severity": rule.severity,
                        "reason": (
                            f"Account holds mutually incompatible roles "
                            f"'{rule.role_a}' and '{rule.role_b}' ({rule.rule_id}). {rule.risk}"
                        ).strip(),
                    }
                )

        if not rows:
            return empty_exceptions()

        result = pd.DataFrame(rows)
        # An account may breach several rules; report every breach (one row each)
        # but sort so the same account's breaches appear together.
        return result.sort_values(["account_id", "rule_id"], kind="stable").reset_index(drop=True)

    def _scope_accounts(self, data: DataBundle) -> pd.DataFrame:
        """Return the account rows eligible for SoD testing."""
        accounts = data.user_accounts
        if accounts.empty or not bool(self.parameters.get("active_accounts_only", True)):
            return accounts
        return accounts.loc[accounts["status"].astype("string").str.casefold().eq("active")]
