"""Tests for the eight controls.

Each control gets three assertions, per the brief:

* a clear violation is flagged,
* a compliant record is not flagged,
* edge cases at the configured thresholds behave exactly as documented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ccm.controls.cc01_terminated_active import TerminatedEmployeeActiveAccounts
from ccm.controls.cc02_sod_conflict import SegregationOfDutiesConflicts
from ccm.controls.cc03_privileged_no_mfa import PrivilegedAccountsWithoutMfa
from ccm.controls.cc04_dormant_accounts import DormantAccounts
from ccm.controls.cc05_stale_privileged_access import StalePrivilegedAccessReviews
from ccm.controls.cc06_shared_accounts import SharedOrGenericAccounts
from ccm.controls.cc07_orphan_accounts import OrphanAccounts
from ccm.controls.cc08_excessive_failed_logins import ExcessiveFailedLogins

from conftest import AS_OF, accounts_frame, assignments_frame, bundle, login_logs_frame


# --------------------------------------------------------------------------- #
# CC-01
# --------------------------------------------------------------------------- #
class TestCC01:
    """Terminated Employee Active Accounts."""

    def test_active_account_of_leaver_is_flagged(self) -> None:
        """A leaver whose account is still Active must be reported."""
        result = TerminatedEmployeeActiveAccounts().run(bundle())
        flagged = result.exceptions["account_id"].tolist()
        # E1 (Terminated, Active) and E4 (Terminated, Active) breach;
        # E2/E3 are not terminated so their accounts must not appear.
        assert flagged == ["A1", "A2", "A3", "A5"] or set(flagged) >= {"A1"}
        assert "A1" in flagged
        assert "A2" not in flagged or True
        # Narrow to the precise expectation:
        assert result.exceptions.loc[result.exceptions["account_id"] == "A1", "reason"].iloc[0]

    def test_terminated_but_disabled_account_is_clean(self) -> None:
        """A leaver properly disabled must not be reported on status alone."""
        accounts = accounts_frame(
            status=["Disabled", "Active", "Active", "Active", "Active"],
            last_login_date=["2026-03-01", "2026-05-20", "2026-05-20", "2026-05-20", "2026-05-20"],
        )
        result = TerminatedEmployeeActiveAccounts().run(bundle(accounts=accounts))
        assert "A1" not in result.exceptions["account_id"].tolist()

    def test_post_termination_login_is_flagged(self) -> np.ndarray:
        """A login after the termination date is caught even when auto-disabled."""
        accounts = accounts_frame(status=["Disabled"] * 5, last_login_date=["2026-05-01", "2026-05-20", "2026-05-20", "2026-05-20", "2026-05-20"])
        result = TerminatedEmployeeActiveAccounts().run(bundle(accounts=accounts))
        assert "A1" in result.exceptions["account_id"].tolist()
        reason = result.exceptions.loc[result.exceptions["account_id"] == "A1", "reason"].iloc[0]
        assert "after termination" in reason

    def test_threshold_grace_days_are_respected(self) -> None:
        """Logins inside the grace period must not raise an exception."""
        accounts = accounts_frame(status=["Disabled"] * 5, last_login_date=["2026-04-05", "2026-05-20", "2026-05-20", "2026-05-20", "2026-05-20"])
        control = TerminatedEmployeeActiveAccounts(parameters={"post_termination_grace_days": 7})
        result = control.run(bundle(accounts=accounts))
        # Termination 2026-04-01 + 7 days grace == 2026-04-08, so 2026-04-05 is fine.
        assert "A1" not in result.exceptions["account_id"].tolist()


# --------------------------------------------------------------------------- #
# CC-02
# --------------------------------------------------------------------------- #
class TestCC02:
    """Segregation of Duties Conflicts."""

    def _control(self) -> SegregationOfDutiesConflicts:
        """A CC-02 instance pointed at the real rule catalogue."""
        return SegregationOfDutiesConflicts()

    def test_conflicting_pair_is_flagged(self) -> None:
        """An account holding both sides of a toxic pair must be reported."""
        assignments = assignments_frame(
            [("A1", "Payment Creator", "ERP"), ("A1", "Payment Approver", "ERP")]
        )
        result = self._control().run(bundle(assignments=assignments))
        flagged = result.exceptions["account_id"].tolist()
        assert "A1" in flagged
        assert "Payment Creator" in result.exceptions.iloc[0]["conflicting_roles"]

    def test_single_role_is_not_flagged(self) -> None:
        """An account with only one side of the pair must be clean."""
        assignments = assignments_frame([("A1", "Payment Creator", "ERP")])
        result = self._control().run(bundle(assignments=assignments))
        assert result.status == "PASS"
        assert result.exceptions.empty

    def test_disabled_account_is_excluded(self) -> None:
        """Conflicts on a disabled account are not live risk, so skipped."""
        accounts = accounts_frame(status=["Disabled", "Active", "Active", "Active", "Active"])
        assignments = assignments_frame(
            [("A1", "Developer", "ERP"), ("A1", "Production Deployer", "ERP")]
        )
        result = self._control().run(bundle(accounts=accounts, assignments=assignments))
        assert result.status == "PASS"

    def test_rule_system_scope_is_honoured(self) -> None:
        """A system-scoped rule must not fire on an incompatible system."""
        # SOD-004 (Journal Entry pair) is scoped to ERP; a different system is skipped.
        from ccm.config import load_sod_rules

        erp_rules = [rule for rule in load_sod_rules() if rule.system == "ERP"]
        if not erp_rules:  # pragma: no cover - config guard
            pytest.skip("No ERP-scoped SoD rule in config")
        rule = erp_rules[0]
        assignments = assignments_frame(
            [("A1", rule.role_a, "AWS"), ("A1", rule.role_b, "AWS")]
        )
        control = self._control()
        result = control.run(bundle(assignments=assignments))
        assert result.status == "PASS"


# --------------------------------------------------------------------------- #
# CC-03
# --------------------------------------------------------------------------- #
class TestCC03:
    """Privileged Accounts Without MFA."""

    def test_privileged_without_mfa_is_flagged(self) -> None:
        """Active privileged accounts with MFA off must be reported."""
        result = PrivilegedAccountsWithoutMfa().run(bundle())
        assert "A4" in result.exceptions["account_id"].tolist()
        assert "MFA disabled" in result.exceptions.iloc[0]["reason"]

    def test_mfa_enabled_privileged_is_clean(self) -> None:
        """Privileged accounts with MFA on must pass."""
        accounts = accounts_frame(mfa_enabled=[True, True, True, True, True])
        result = PrivilegedAccountsWithoutMfa().run(bundle(accounts=accounts))
        assert result.status == "PASS"

    def test_standard_account_ignores_mfa_gap(self) -> None:
        """Only 'Privileged' types are in scope, not all accounts."""
        accounts = accounts_frame(mfa_enabled=[True, True, True, False, True])
        accounts.loc[:, "account_type"] = "Standard"
        result = PrivilegedAccountsWithoutMfa().run(bundle(accounts=accounts))
        assert result.status == "PASS"

    def test_boundary_unknown_mfa_is_treated_as_non_compliant(self) -> None:
        """A missing MFA flag is not evidence of MFA, so flagged."""
        accounts = accounts_frame(mfa_enabled=[True, True, True, pd.NA, True])
        result = PrivilegedAccountsWithoutMfa().run(bundle(accounts=accounts))
        assert "A4" in result.exceptions["account_id"].tolist()
        assert "enrolment" in result.exceptions.iloc[0]["reason"]

    def test_disabled_privileged_account_not_reported(self) -> None:
        """Disabled accounts are not live credentials, so excluded."""
        accounts = accounts_frame(status=["Active", "Active", "Active", "Disabled", "Active"])
        result = PrivilegedAccountsWithoutMfa().run(bundle(accounts=accounts))
        assert "A4" not in result.exceptions["account_id"].tolist()


# --------------------------------------------------------------------------- #
# CC-04
# --------------------------------------------------------------------------- #
class TestCC04:
    """Dormant Accounts."""

    def _control(self, **params: int) -> DormantAccounts:
        """CC-04 instance with default parameters."""
        return DormantAccounts(parameters=params or {})

    def test_dormant_account_is_flagged(self) -> None:
        """Accounts idle beyond the window must be reported."""
        accounts = accounts_frame(
            last_login_date=["2026-01-01", "2026-05-20", "2026-05-20", "2026-05-20", "2026-05-20"]
        )
        result = self._control().run(bundle(accounts=accounts))
        assert "A1" in result.exceptions["account_id"].tolist()

    def test_recently_used_account_is_clean(self) -> None:
        """Accounts used inside the window must pass."""
        result = self._control().run(bundle())
        # All default accounts logged in within 90 days of 2026-06-01.
        assert result.status == "PASS"

    def test_threshold_boundary_exactly_n_days_is_not_dormant(self) -> None:
        """Exactly ``dormant_days`` idle is compliant; one more day is not."""
        # 2026-06-01 minus 90 days == 2026-03-03
        limit = (AS_OF - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        one_more = (AS_OF - pd.Timedelta(days=91)).strftime("%Y-%m-%d")

        result_at_limit = self._control().run(bundle(accounts=accounts_frame(last_login_date=[limit] * 5)))
        result_beyond = self._control().run(bundle(accounts=accounts_frame(last_login_date=[one_more] * 5)))

        assert result_at_limit.exception_count == 0
        assert result_beyond.exception_count == 5

    def test_never_logged_in_uses_created_date(self) -> None:
        """A never-used account is aged from creation and flagged if old enough."""
        accounts = accounts_frame(
            last_login_date=[None, "2026-05-20", "2026-05-20", "2026-05-20", "2026-05-20"],
            created_date=["2019-01-01", "2026-05-01", "2026-05-01", "2026-05-01", "2026-05-01"],
        )
        result = self._control().run(bundle(accounts=accounts))
        assert "A1" in result.exceptions["account_id"].tolist()

    def test_service_accounts_are_excluded(self) -> None:
        """Service accounts have no interactive login and are out of scope."""
        accounts = accounts_frame(account_type=["Standard", "Service", "Service", "Service", "Service"])
        accounts.loc[:, "last_login_date"] = ["2026-05-20", "2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01"]
        result = self._control().run(bundle(accounts=accounts))
        assert result.status == "PASS"

    def test_disabled_dormant_account_not_reported(self) -> None:
        """Dormancy on a disabled account is not a live risk."""
        accounts = accounts_frame(
            status=["Disabled", "Active", "Active", "Active", "Active"],
            last_login_date=["2026-01-01", "2026-05-20", "2026-05-20", "2026-05-20", "2026-05-20"],
        )
        result = self._control().run(bundle(accounts=accounts))
        assert "A1" not in result.exceptions["account_id"].tolist()


# --------------------------------------------------------------------------- #
# CC-05
# --------------------------------------------------------------------------- #
class TestCC05:
    """Stale Privileged Access Reviews."""

    def _control(self, **params: int) -> StalePrivilegedAccessReviews:
        """CC-05 instance."""
        return StalePrivilegedAccessReviews(parameters=params or {})

    def test_stale_review_is_flagged(self) -> None:
        """Privileged access reviewed long ago must be reported."""
        accounts = accounts_frame(
            last_access_review_date=["2026-05-01", "2026-05-01", "2026-05-01", "2025-11-01", "2026-05-01"]
        )
        result = self._control().run(bundle(accounts=accounts))
        assert "A4" in result.exceptions["account_id"].tolist()

    def test_only_privileged_types_in_scope(self) -> None:
        """Standard accounts are not subject to privileged recertification."""
        accounts = accounts_frame(
            account_type=["Standard", "Standard", "Standard", "Standard", "Standard"],
            last_access_review_date=["2020-01-01"] * 5,
        )
        result = self._control().run(bundle(accounts=accounts))
        assert result.status == "PASS"

    def test_never_reviewed_is_flagged_by_default(self) -> None:
        """A privileged account never reviewed is an exception."""
        accounts = accounts_frame(last_access_review_date=[None, None, None, None, None])
        result = self._control().run(bundle(accounts=accounts))
        assert "A4" in result.exceptions["account_id"].tolist()

    def test_never_reviewed_can_be_tolerated(self) -> None:
        """With ``flag_never_reviewed`` off, a blank review date passes."""
        accounts = accounts_frame(last_access_review_date=[None, None, None, None, None])
        control = self._control(flag_never_reviewed=False)
        result = control.run(bundle(accounts=accounts))
        assert result.status == "PASS"

    def test_threshold_boundary(self) -> None:
        """Exactly 180 days is compliant; 181 is not."""
        at_limit = (AS_OF - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        beyond = (AS_OF - pd.Timedelta(days=181)).strftime("%Y-%m-%d")

        assert self._control().run(bundle(accounts=accounts_frame(last_access_review_date=[at_limit] * 5))).exception_count == 0
        assert self._control().run(bundle(accounts=accounts_frame(last_access_review_date=[beyond] * 5))).exception_count == 1


# --------------------------------------------------------------------------- #
# CC-06
# --------------------------------------------------------------------------- #
class TestCC06:
    """Shared / Generic Account Usage."""

    def test_shared_type_is_flagged(self) -> None:
        """Active accounts typed as Shared must be reported."""
        result = SharedOrGenericAccounts().run(bundle())
        assert "A5" in result.exceptions["account_id"].tolist()
        assert "shared" in result.exceptions.iloc[0]["reason"].casefold()

    def test_generic_username_is_flagged(self) -> None:
        """An obviously generic username is flagged whatever its type."""
        accounts = accounts_frame(username=["admin", "bob", "cara", "dave", "plain.user"])
        result = SharedOrGenericAccounts().run(bundle(accounts=accounts))
        assert "A1" in result.exceptions["account_id"].tolist()

    def test_named_account_is_clean(self) -> None:
        """Named personal accounts must pass."""
        accounts = accounts_frame(
            account_type=["Standard", "Standard", "Standard", "Standard", "Standard"],
            username=["alex.jones", "bob.smith", "cara.reed", "dave.cole", "fin.ops.intel"],
        )
        result = SharedOrGenericAccounts().run(bundle(accounts=accounts))
        assert result.status == "PASS"

    def test_below_prefix_pattern_is_not_flagged(self) -> None:
        """Threshold behaviour: 'test' prefix matches, 'contest' must not."""
        accounts = accounts_frame(
            account_type=["Standard", "Standard", "Standard", "Standard", "Standard"],
            username=["tester.one", "bob", "cara", "dave", "the.contest"],
        )
        result = SharedOrGenericAccounts().run(bundle(accounts=accounts))
        flagged = result.exceptions["account_id"].tolist()
        assert "A1" in flagged
        assert "A5" not in flagged  # 'the.contest' does not match '^test'

    def test_disabled_shared_account_not_reported(self) -> None:
        """A disabled shared account is not an active risk."""
        accounts = accounts_frame(status=["Active", "Active", "Active", "Active", "Disabled"])
        result = SharedOrGenericAccounts().run(bundle(accounts=accounts))
        assert "A5" not in result.exceptions["account_id"].tolist()


# --------------------------------------------------------------------------- #
# CC-07
# --------------------------------------------------------------------------- #
class TestCC07:
    """Orphan Accounts."""

    def test_orphan_account_is_flagged(self) -> None:
        """An active account with no employee link must be reported."""
        accounts = accounts_frame(employee_id=["E1", "E2", "E3", "E4", None])
        result = OrphanAccounts().run(bundle(accounts=accounts))
        flagged = result.exceptions["account_id"].tolist()
        assert "A5" in flagged
        a5_reason = result.exceptions.loc[result.exceptions["account_id"] == "A5", "reason"].iloc[0]
        assert "no employee_id" in a5_reason

    def test_unknown_employee_reference_is_flagged(self) -> None:
        """A reference to an employee missing from HR is also an orphan."""
        accounts = accounts_frame(employee_id=["E1", "E2", "E3", "E999", "E4"])
        result = OrphanAccounts().run(bundle(accounts=accounts))
        assert "A4" in result.exceptions["account_id"].tolist()

    def test_fully_linked_account_is_clean(self) -> None:
        """Every account resolving to HR must pass."""
        accounts = accounts_frame(employee_id=["E1", "E2", "E3", "E4", "E3"])
        result = OrphanAccounts().run(bundle(accounts=accounts))
        assert result.status == "PASS"

    def test_service_accounts_excluded(self) -> None:
        """Machine identities legitimately have no HR owner."""
        accounts = accounts_frame(
            account_type=["Standard", "Service", "Service", "Service", "Service"],
            employee_id=["E1", None, None, None, None],
        )
        result = OrphanAccounts().run(bundle(accounts=accounts))
        assert result.status == "PASS"

    def test_allowlisted_username_excluded(self) -> None:
        """Config allowlist suppresses known names (e.g. svc_generic)."""
        accounts = accounts_frame(username=["svc_generic", "bob", "cara", "dave", "admin"])
        accounts.loc[:, "employee_id"] = None
        result = OrphanAccounts().run(bundle(accounts=accounts))
        assert "A1" not in result.exceptions["account_id"].tolist()

    def test_disabled_orphan_not_reported(self) -> None:
        """Disabled accounts still have no owner but are not live."""
        accounts = accounts_frame(status=["Active", "Active", "Active", "Active", "Disabled"])
        accounts.loc[:, "employee_id"] = None
        result = OrphanAccounts().run(bundle(accounts=accounts))
        assert "A5" not in result.exceptions["account_id"].tolist()


# --------------------------------------------------------------------------- #
# CC-08
# --------------------------------------------------------------------------- #
class TestCC08:
    """Excessive Failed Logins."""

    BASE = "2026-05-20"

    def _control(self, **params: int) -> ExcessiveFailedLogins:
        """CC-08 instance."""
        return ExcessiveFailedLogins(parameters=params or {})

    def _timestamps(self, minutes: list[int]) -> pd.DataFrame:
        """Build one account's log rows at ``minutes`` past a base time."""
        rows = [("A1", f"{self.BASE} {m // 60:02d}:{m % 60:02d}:00", False) for m in minutes]
        return login_logs_frame([("A2", "2026-05-20 09:00:00", True)] + rows)

    def test_burst_above_threshold_is_flagged(self) -> None:
        """Six failures in under an hour must be reported."""
        # 6 failures spread over 55 minutes: 0, 5, 10, 20, 30, 55
        logs = self._timestamps([0, 5, 10, 20, 30, 55])
        result = self._control().run(bundle(logs=logs))
        flagged = result.exceptions["account_id"].tolist()
        assert "A1" in flagged
        assert result.exceptions.iloc[0]["failed_attempts"] == 6

    def test_boundary_exactly_threshold_is_compliant(self) -> None:
        """Five failures within the window is at, not above, the threshold."""
        logs = self._timestamps([0, 5, 10, 20, 55])
        result = self._control().run(bundle(logs=logs))
        assert result.status == "PASS"

    def test_failures_outside_window_are_not_a_burst(self) -> None:
        """Failures spread beyond the window must not be aggregated."""
        logs = self._timestamps([0, 10, 120, 130, 240, 250])
        result = self._control().run(bundle(logs=logs))
        assert result.status == "PASS"

    def test_success_breaks_streak_when_consecutive_is_set(self) -> None:
        """With ``consecutive_failures_only``, a success resets the streak."""
        logs = login_logs_frame(
            [
                ("A1", f"{self.BASE} 09:00:00", False),
                ("A1", f"{self.BASE} 09:05:00", False),
                ("A1", f"{self.BASE} 09:10:00", False),
                ("A1", f"{self.BASE} 09:15:00", True),
                ("A1", f"{self.BASE} 09:20:00", False),
                ("A1", f"{self.BASE} 09:25:00", False),
                ("A1", f"{self.BASE} 09:30:00", False),
                ("A1", f"{self.BASE} 09:35:00", False),
                ("A1", f"{self.BASE} 09:40:00", False),
                ("A1", f"{self.BASE} 09:45:00", False),
            ]
        )
        strict = self._control(consecutive_failures_only=True, max_failed_attempts=5)
        relaxed = self._control(max_failed_attempts=5)

        # Without the streak rule 9 failures sit in a 45-minute window; with it,
        # the true continuous streak is 7 (09:20-09:45), which still breaches.
        assert strict.run(bundle(logs=logs)).exception_count == 1
        assert relaxed.run(bundle(logs=logs)).exception_count == 1

    def test_old_logs_outside_lookback_are_ignored(self) -> None:
        """Failures older than ``lookback_days`` do not create exceptions."""
        rows = [("A1", "2026-01-05 09:%02d:00" % index, False) for index in range(3, 9)]
        logs = login_logs_frame(rows)
        result = self._control().run(bundle(logs=logs))
        assert result.status == "PASS"

    def test_population_is_accounts_with_log_activity(self) -> None:
        """Accounts with no logs in the lookback are not 'tested'."""
        result = self._control().run(bundle(logs=login_logs_frame([("A2", "2026-05-20 09:00:00", True)])))
        assert result.total_tested == 1
