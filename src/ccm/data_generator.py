"""Synthetic data generator for the CCM demo.

Produces a realistic HR / IAM / SIEM extract with **known, seeded exceptions**
injected into an otherwise clean population. Fixed seeds make every run
reproducible, which matters for two reasons:

* the README's numbers stay true over time;
* a reviewer can diff two runs and see only what they changed.

Clean rows are generated so that they would not trip any control - the exception
counts in the dashboard are then exactly the injection constants below, which is
the strongest possible check that the controls work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from .paths import RAW_DATA_DIR, ensure_directories

logger = logging.getLogger(__name__)

__all__ = [
    "InjectionPlan",
    "InjectionReport",
    "GenerationConfig",
    "generate_dataset",
]

# --------------------------------------------------------------------------- #
# Tunable constants - change these to shape the demo dataset
# --------------------------------------------------------------------------- #
#: Seed for both Faker and NumPy. Fixed so results are reproducible.
RANDOM_SEED = 42

#: Employee count in the generated HR population.
EMPLOYEE_COUNT = 500


@dataclass(frozen=True)
class InjectionPlan:
    """Counts of deliberate control breaches injected into the dataset.

    These are the demo's ground truth: a control whose exception count differs
    from the corresponding constant has a bug in its logic.

    Attributes:
        terminated_with_active_accounts: Leavers whose accounts stay Active (CC-01).
        post_termination_logins: Leavers whose accounts log in after exit (CC-01).
        sod_conflicts: Accounts holding a toxic role pair (CC-02).
        privileged_without_mfa: Active privileged accounts with MFA off (CC-03).
        dormant_accounts: Active accounts idle past the dormancy window (CC-04).
        stale_access_reviews: Privileged accounts with lapsed recertification (CC-05).
        shared_accounts: Active shared/generic accounts (CC-06).
        orphan_accounts: Active accounts with no HR owner (CC-07).
        brute_force_accounts: Accounts with failure bursts (CC-08).
    """

    terminated_with_active_accounts: int = 8
    post_termination_logins: int = 3
    sod_conflicts: int = 12
    privileged_without_mfa: int = 10
    dormant_accounts: int = 25
    stale_access_reviews: int = 7
    shared_accounts: int = 6
    orphan_accounts: int = 5
    brute_force_accounts: int = 4


#: Reference plan used unless overridden.
DEFAULT_INJECTION_PLAN = InjectionPlan()


@dataclass(frozen=True)
class GenerationConfig:
    """Full generation settings.

    Attributes:
        employee_count: Number of HR employees to create.
        seed: Deterministic seed for Faker and NumPy.
        plan: Exception injection counts.
        dormant_days: Dormancy threshold assumed for CC-04 (kept in sync with config).
        review_interval_days: Recertification interval assumed for CC-05.
        as_of: Reference "today" for date generation; defaults to now.
    """

    employee_count: int = EMPLOYEE_COUNT
    seed: int = RANDOM_SEED
    plan: InjectionPlan = DEFAULT_INJECTION_PLAN
    dormant_days: int = 90
    review_interval_days: int = 180
    as_of: datetime = field(default_factory=lambda: datetime.now())


#: Reference date for generating realistic, non-future timestamps.
_MAX_HISTORY_YEARS = 3

_DEPARTMENTS: Sequence[str] = (
    "Finance", "IT", "Sales", "Human Resources", "Operations",
    "Procurement", "Engineering", "Marketing", "Legal", "Customer Service",
)

_JOB_TITLES: Sequence[str] = (
    "Analyst", "Senior Analyst", "Manager", "Senior Manager", "Director",
    "Specialist", "Coordinator", "Consultant", "Team Lead", "Associate",
)

#: Systems in the simulated estate, with their typical account type mix.
_SYSTEMS: Sequence[str] = ("ERP", "CRM", "AWS", "Database", "VPN", "Payments")

#: Roles per system. Most are benign; a few pair up as SoD conflicts.
_ROLES_BY_SYSTEM: dict[str, Sequence[str]] = {
    "ERP": (
        "Journal Entry Preparer", "Journal Entry Approver", "Payroll Administrator",
        "Payroll Approver", "Vendor Master Editor", "Bank Reconciliation Preparer",
        "Bank Reconciliation Approver", "Purchasing Clerk", "Inventory Manager", "Viewer",
    ),
    "CRM": (
        "Customer Master Editor", "Credit Approver", "Sales Representative",
        "Sales Manager", "Support Agent", "Viewer",
    ),
    "AWS": (
        "Developer", "Production Deployer", "Security Administrator",
        "Audit Log Reviewer", "ReadOnly", "DevOps Engineer",
    ),
    "Database": ("DBA", "ReadOnly", "Data Analyst", "Security Administrator", "Audit Log Reviewer"),
    "VPN": ("Standard User", "Network Administrator"),
    "Payments": ("Payment Creator", "Payment Approver", "Treasury Analyst", "Viewer"),
}

#: Role pairs that must never coexist on one account are now read from
#: ``config/sod_rules.yaml`` via :func:`_conflicting_pairs` - a single source of
#: truth shared with the CC-02 control.

#: Usernames that CC-06 treats as generic - avoided for clean identities.
_GENERIC_USERNAME_TOKENS = ("admin", "test", "shared", "svc_generic", "generic", "tmp")


def _conflicting_pairs() -> list[tuple[str, str, str]]:
    """Read the conflicting role pairs (and their system scope) from config.

    Reading these from ``config/sod_rules.yaml`` keeps a single source of truth:
    if risk adds a rule, the generator honours it and the demo data stays
    consistent with what CC-02 will actually test against.

    Returns:
        List of ``(role_a, role_b, system)`` tuples; ``system`` is ``"*"`` for
        rules that apply to any system.
    """
    from .config import load_sod_rules

    return [(str(rule.role_a), str(rule.role_b), str(rule.system)) for rule in load_sod_rules()]


class _InjectionTracker:
    """Accumulates a human-readable summary of what was injected."""

    def __init__(self) -> None:
        """Start with an empty ledger."""
        self._entries: list[tuple[str, str, int]] = []

    def add(self, control_id: str, label: str, count: int) -> None:
        """Record an injection.

        Args:
            control_id: Control the breach targets.
            label: Human-readable description.
            count: Number of breaches injected.
        """
        if count:
            self._entries.append((control_id, label, count))

    def summary(self) -> str:
        """Render the ledger as a printable table."""
        if not self._entries:
            return "No exceptions injected."
        width = max(len(label) for _, label, _ in self._entries) + 2
        lines = ["", "Injected exceptions (the demo's ground truth):"]
        lines.append(f"  {'Control':<8} {'Count':>5}  Detail")
        lines.append(f"  {'-' * 8} {'-' * 5}  {'-' * width}")
        for control_id, label, count in self._entries:
            lines.append(f"  {control_id:<8} {count:>5}  {label}")
        total = sum(count for _, _, count in self._entries)
        lines.append(f"  {'-' * 8} {'-' * 5}  {'-' * width}")
        lines.append(f"  {'TOTAL':<8} {total:>5}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def generate_dataset(
    *,
    rows: int | None = None,
    output_dir: Path | str | None = None,
    config: GenerationConfig | None = None,
) -> Path:
    """Generate the four source CSVs into ``data/raw``.

    Args:
        rows: Employee count override (defaults to :data:`EMPLOYEE_COUNT`).
        output_dir: Destination directory (defaults to ``data/raw``).
        config: Full generation settings; ``rows`` wins when both are given.

    Returns:
        The directory the CSVs were written to.
    """
    base = config or GenerationConfig()
    if rows is not None:
        base = GenerationConfig(
            employee_count=rows,
            seed=base.seed,
            plan=base.plan,
            dormant_days=base.dormant_days,
            review_interval_days=base.review_interval_days,
            as_of=base.as_of,
        )

    Faker.seed(base.seed)
    np.random.seed(base.seed)
    fake = Faker("en_GB")

    directory = Path(output_dir) if output_dir is not None else RAW_DATA_DIR
    ensure_directories()
    directory.mkdir(parents=True, exist_ok=True)

    tracker = _InjectionTracker()

    hr = _generate_employees(base, fake)
    accounts, assignments, logs = _generate_access_data(base, fake, hr, tracker)

    hr.to_csv(directory / "hr_employees.csv", index=False)
    accounts.to_csv(directory / "user_accounts.csv", index=False)
    assignments.to_csv(directory / "access_assignments.csv", index=False)
    logs.to_csv(directory / "login_logs.csv", index=False)

    print("=" * 72)
    print("SYNTHETIC DATA GENERATED")
    print("=" * 72)
    print(f"  Seed              : {base.seed}")
    print(f"  Employees         : {len(hr):,}")
    print(f"  User accounts     : {len(accounts):,}")
    print(f"  Role assignments  : {len(assignments):,}")
    print(f"  Login log records : {len(logs):,}")
    print(f"  Output directory  : {directory}")
    print(tracker.summary())
    print("=" * 72)

    return directory


# --------------------------------------------------------------------------- #
# Employee generation
# --------------------------------------------------------------------------- #
def _generate_employees(config: GenerationConfig, fake: Faker) -> pd.DataFrame:
    """Build the HR master table, including leavers and leave records."""
    rng = np.random.default_rng(config.seed)
    as_of = config.as_of.date()

    plan = config.plan
    # Leavers: those with a live account (breach) and those properly offboarded.
    terminated_breached = plan.terminated_with_active_accounts
    terminated_clean = max(8, terminated_breached)
    on_leave = max(6, config.employee_count // 40)
    active_count = max(0, config.employee_count - terminated_breached - terminated_clean - on_leave)

    statuses: list[str] = (
        ["Active"] * active_count
        + ["Terminated"] * terminated_clean
        + ["Terminated"] * terminated_breached
        + ["On Leave"] * on_leave
    )
    rng.shuffle(statuses)

    rows: list[dict[str, object]] = []
    used_ids: set[str] = set()
    for index in range(config.employee_count):
        employee_id = f"EMP{index + 1:05d}"
        used_ids.add(employee_id)

        first, last = fake.first_name(), fake.last_name()
        status = statuses[index]
        hire_date = as_of - timedelta(days=int(rng.integers(200, 3000)))

        termination_date = None
        if status == "Terminated":
            # Leavers exited between 20 and 400 days ago.
            termination_date = as_of - timedelta(days=int(rng.integers(20, 400)))
            if termination_date < hire_date:
                hire_date = termination_date - timedelta(days=int(rng.integers(300, 1500)))

        rows.append(
            {
                "employee_id": employee_id,
                "name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}{index}@example.com",
                "department": str(rng.choice(_DEPARTMENTS)),
                "job_title": str(rng.choice(_JOB_TITLES)),
                "manager_id": None,
                "status": status,
                "hire_date": hire_date.isoformat(),
                "termination_date": termination_date.isoformat() if termination_date else None,
            }
        )

    frame = pd.DataFrame(rows)
    # Point each employee at another employee as their manager (never themselves).
    manager_pool = frame["employee_id"].to_numpy()
    managers = [str(rng.choice(manager_pool)) for _ in range(len(frame))]
    frame["manager_id"] = [
        manager if manager != employee_id else None
        for manager, employee_id in zip(managers, frame["employee_id"])
    ]
    return frame


# --------------------------------------------------------------------------- #
# Access data generation
# --------------------------------------------------------------------------- #
def _generate_access_data(
    config: GenerationConfig,
    fake: Faker,
    hr: pd.DataFrame,
    tracker: _InjectionTracker,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build accounts, role assignments and login logs.

    Returns:
        Tuple of ``(user_accounts, access_assignments, login_logs)`` frames.
    """
    rng = np.random.default_rng(config.seed + 1)
    plan = config.plan
    as_of = config.as_of

    used_usernames: set[str] = set()

    def unique_username(base: str) -> str:
        """Return a username that does not collide and is not generically named."""
        candidate = base.casefold()
        candidate = "".join(token for token in candidate if token.isalnum() or token == ".")
        if any(candidate.startswith(token) for token in _GENERIC_USERNAME_TOKENS):
            candidate = f"{candidate.split('.')[0]}x.{candidate.split('.')[-1]}" if "." in candidate else f"{candidate}x"
        trial, suffix = candidate, 1
        while trial in used_usernames:
            suffix += 1
            trial = f"{candidate}{suffix}"
        used_usernames.add(trial)
        return trial

    account_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []

    def next_account_id() -> str:
        """Sequential account id."""
        return f"ACC{len(account_rows) + 1:05d}"

    def next_log_id() -> str:
        """Sequential log id."""
        return f"LOG{len(log_rows) + 1:07d}"

    def grant_role(account_id: str, system: str, role_name: str, granted_by: str) -> None:
        """Record a role grant."""
        assignment_rows.append(
            {
                "account_id": account_id,
                "role_name": role_name,
                "system": system,
                "granted_date": (as_of - timedelta(days=int(rng.integers(30, 900)))).date().isoformat(),
                "granted_by": granted_by,
            }
        )

    def add_logins(
        account_id: str,
        *,
        recent: bool,
        success_rate: float = 0.97,
        after_hours_rate: float = 0.08,
        count: int | None = None,
    ) -> None:
        """Append a plausible login history for an account."""
        n = count if count is not None else int(rng.integers(20, 90))
        for _ in range(n):
            if recent:
                timestamp = as_of - timedelta(
                    days=float(rng.integers(0, 30)),
                    hours=float(rng.integers(6, 22)),
                    minutes=float(rng.integers(0, 60)),
                )
            else:
                timestamp = as_of - timedelta(
                    days=float(rng.integers(0, 400)),
                    hours=float(rng.integers(6, 22)),
                    minutes=float(rng.integers(0, 60)),
                )
            log_rows.append(
                {
                    "log_id": next_log_id(),
                    "account_id": account_id,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "success": bool(rng.random() < success_rate),
                    "source_ip": f"10.{int(rng.integers(0, 250))}.{int(rng.integers(0, 250))}.{int(rng.integers(1, 250))}",
                    "is_after_hours": bool(rng.random() < after_hours_rate),
                }
            )

    # ---- 1. Clean majority: one account per active/on-leave employee ---------
    for _, employee in hr.iterrows():
        if str(employee["status"]) == "Terminated":
            continue  # handled below

        system = str(rng.choice(_SYSTEMS))
        is_privileged = bool(rng.random() < 0.08)
        account_type = "Privileged" if is_privileged else "Standard"
        account_id = next_account_id()
        username = unique_username(f"{str(employee['name']).split()[0].lower()}.{str(employee['name']).split()[-1].lower()}")

        created = as_of - timedelta(days=int(rng.integers(180, 1200)))
        last_login = as_of - timedelta(days=int(rng.integers(0, 60)))

        account_rows.append(
            {
                "account_id": account_id,
                "username": username,
                "employee_id": employee["employee_id"],
                "system": system,
                "account_type": account_type,
                "status": "Active",
                "mfa_enabled": True,
                "created_date": created.date().isoformat(),
                "last_login_date": last_login.date().isoformat(),
                "last_password_change": (as_of - timedelta(days=int(rng.integers(5, 120)))).date().isoformat(),
                "last_access_review_date": (
                    as_of - timedelta(days=int(rng.integers(0, config.review_interval_days - 20)))
                ).date().isoformat(),
            }
        )

        roles = list(_ROLES_BY_SYSTEM[system])
        grant_role(account_id, system, str(rng.choice(roles)), "SEC-TEAM")
        add_logins(account_id, recent=True)

    # ---- 2. Terminated employees: mostly offboarded correctly ----------------
    terminated = hr.loc[hr["status"] == "Terminated"].reset_index(drop=True)
    breached = terminated.head(plan.terminated_with_active_accounts)
    clean_leavers = terminated.iloc[len(breached):]

    for _, employee in clean_leavers.iterrows():
        system = str(rng.choice(_SYSTEMS))
        account_id = next_account_id()
        username = unique_username(f"{str(employee['name']).split()[0].lower()}.{str(employee['name']).split()[-1].lower()}")
        termination = pd.Timestamp(employee["termination_date"])
        account_rows.append(
            {
                "account_id": account_id,
                "username": username,
                "employee_id": employee["employee_id"],
                "system": system,
                "account_type": "Standard",
                "status": "Disabled",
                "mfa_enabled": True,
                "created_date": (termination - timedelta(days=400)).date().isoformat(),
                "last_login_date": (termination - timedelta(days=2)).date().isoformat(),
                "last_password_change": (termination - timedelta(days=90)).date().isoformat(),
                "last_access_review_date": (termination - timedelta(days=30)).date().isoformat(),
            }
        )
        grant_role(account_id, system, "Viewer", "SEC-TEAM")

    # ---- 3. CC-01 breaches: leavers with live accounts ----------------------
    for offset, (_, employee) in enumerate(breached.iterrows()):
        system = str(rng.choice(_SYSTEMS))
        account_id = next_account_id()
        username = unique_username(f"{str(employee['name']).split()[0].lower()}.{str(employee['name']).split()[-1].lower()}")
        # Injected leavers exit recently: deprovisioning failures surface in
        # recent exits, and their accounts then read as *active*, not dormant,
        # so CC-01 and CC-04 each report exactly what was injected.
        termination = as_of - timedelta(days=int(rng.integers(10, 70)))
        hr.loc[hr["employee_id"] == employee["employee_id"], "termination_date"] = termination.date().isoformat()

        last_login = termination + timedelta(days=int(rng.integers(1, 20)))
        if offset >= plan.post_termination_logins:
            # Still active, but never used after exit: CC-01 catches it on status.
            # Idle time stays within the dormancy window because termination is
            # recent (see above).
            last_login = termination - timedelta(days=int(rng.integers(1, 20)))

        account_rows.append(
            {
                "account_id": account_id,
                "username": username,
                "employee_id": employee["employee_id"],
                "system": system,
                "account_type": "Standard",
                "status": "Active",
                "mfa_enabled": True,
                "created_date": (termination - timedelta(days=300)).date().isoformat(),
                "last_login_date": last_login.date().isoformat(),
                "last_password_change": (termination - timedelta(days=60)).date().isoformat(),
                "last_access_review_date": (termination - timedelta(days=15)).date().isoformat(),
            }
        )
        grant_role(account_id, system, "Viewer", "SEC-TEAM")
        if last_login >= termination:
            add_logins(account_id, recent=False, count=int(rng.integers(3, 12)))

    tracker.add("CC-01", "terminated employees with active accounts", len(breached))

    # ---- 4. CC-02 breaches: SoD conflicts -----------------------------------
    # Read the rule catalogue from config rather than duplicating it here, and
    # only inject onto accounts whose system matches the rule's declared scope -
    # otherwise the control would (correctly) decline to flag them.
    clean_pool = [row for row in account_rows if row["account_type"] == "Standard"]
    rng.shuffle(clean_pool)
    injected_conflicts = 0
    used_for_sod: set[str] = set()
    for role_a, role_b, rule_system in [pair for pair in _conflicting_pairs()] * 3:
        if injected_conflicts >= plan.sod_conflicts:
            break
        candidates = [
            row for row in clean_pool
            if row["account_id"] not in used_for_sod
            and (rule_system == "*" or str(row["system"]) == rule_system)
        ]
        if not candidates:
            continue
        account = candidates[int(rng.integers(0, len(candidates)))]
        used_for_sod.add(str(account["account_id"]))
        grant_role(str(account["account_id"]), str(account["system"]), role_a, "SEC-TEAM")
        grant_role(str(account["account_id"]), str(account["system"]), role_b, "SEC-TEAM")
        injected_conflicts += 1

    tracker.add("CC-02", "accounts holding conflicting SoD role pairs", injected_conflicts)

    # ---- 5. CC-03 breaches: privileged accounts without MFA -----------------
    privileged = [row for row in account_rows if row["account_type"] == "Privileged" and row["status"] == "Active"]
    for account in privileged[: plan.privileged_without_mfa]:
        account["mfa_enabled"] = False
    tracker.add("CC-03", "privileged accounts without MFA", min(plan.privileged_without_mfa, len(privileged)))

    # ---- 6. CC-04 breaches: dormant accounts --------------------------------
    dormant_pool = [row for row in account_rows if row["status"] == "Active" and row["account_type"] == "Standard"]
    for account in dormant_pool[: plan.dormant_accounts]:
        account["last_login_date"] = (as_of - timedelta(days=int(rng.integers(config.dormant_days + 30, 500)))).date().isoformat()
    tracker.add("CC-04", "active accounts dormant beyond threshold", min(plan.dormant_accounts, len(dormant_pool)))

    # ---- 7. CC-05 breaches: stale privileged access reviews ------------------
    # Slice *after* the CC-03 band so the two controls report on distinct
    # accounts rather than double-counting the same remediation.
    stale_pool = privileged[plan.privileged_without_mfa:]
    for account in stale_pool[: plan.stale_access_reviews]:
        account["last_access_review_date"] = (
            as_of - timedelta(days=int(rng.integers(config.review_interval_days + 30, 700)))
        ).date().isoformat()
    tracker.add(
        "CC-05",
        "privileged accounts with stale access reviews",
        min(plan.stale_access_reviews, len(stale_pool)),
    )

    # ---- 8. CC-06 breaches: shared / generic accounts ------------------------
    # Shared accounts carry an HR custodian (the owning manager) so CC-06 and
    # CC-07 report independent findings instead of double-counting.
    custodians = [str(value) for value in hr["employee_id"].dropna().astype(str).tolist()]
    for index in range(plan.shared_accounts):
        account_id = next_account_id()
        generic_name = ["admin", "shared.finance", "test.user", "svc_generic", "generic.hr", "tmp.backup"][index % 6]
        system = str(rng.choice(_SYSTEMS))
        account_rows.append(
            {
                "account_id": account_id,
                "username": generic_name,
                "employee_id": str(rng.choice(custodians)) if custodians else None,
                "system": system,
                "account_type": "Shared",
                "status": "Active",
                "mfa_enabled": True,
                "created_date": (as_of - timedelta(days=int(rng.integers(200, 900)))).date().isoformat(),
                "last_login_date": (as_of - timedelta(days=int(rng.integers(0, 45)))).date().isoformat(),
                "last_password_change": (as_of - timedelta(days=int(rng.integers(30, 200)))).date().isoformat(),
                "last_access_review_date": (as_of - timedelta(days=int(rng.integers(0, 150)))).date().isoformat(),
            }
        )
        grant_role(account_id, system, "Viewer", "SEC-TEAM")
        add_logins(account_id, recent=True, count=int(rng.integers(5, 20)))
    tracker.add("CC-06", "active shared / generic accounts", plan.shared_accounts)

    # ---- 9. CC-07 breaches: orphan accounts ----------------------------------
    for index in range(plan.orphan_accounts):
        account_id = next_account_id()
        first, last = fake.first_name(), fake.last_name()
        system = str(rng.choice(_SYSTEMS))
        account_rows.append(
            {
                "account_id": account_id,
                "username": unique_username(f"{first.lower()}.{last.lower()}"),
                "employee_id": None,
                "system": system,
                "account_type": "Standard",
                "status": "Active",
                "mfa_enabled": True,
                "created_date": (as_of - timedelta(days=int(rng.integers(100, 700)))).date().isoformat(),
                "last_login_date": (as_of - timedelta(days=int(rng.integers(0, 40)))).date().isoformat(),
                "last_password_change": (as_of - timedelta(days=int(rng.integers(20, 150)))).date().isoformat(),
                "last_access_review_date": (as_of - timedelta(days=int(rng.integers(0, 150)))).date().isoformat(),
            }
        )
        grant_role(account_id, system, "Viewer", "SEC-TEAM")
        add_logins(account_id, recent=True, count=int(rng.integers(5, 25)))
    tracker.add("CC-07", "orphan accounts with no HR owner", plan.orphan_accounts)

    # ---- 10. CC-08 breaches: brute-force failure bursts ----------------------
    burst_targets = [row["account_id"] for row in account_rows if row["account_type"] == "Standard"][: plan.brute_force_accounts]
    for account_id in burst_targets:
        start = as_of - timedelta(days=float(rng.integers(1, 20)), hours=float(rng.integers(1, 6)))
        for attempt in range(int(rng.integers(6, 14))):
            log_rows.append(
                {
                    "log_id": next_log_id(),
                    "account_id": account_id,
                    "timestamp": (start + timedelta(minutes=int(attempt * rng.integers(2, 9)))).strftime("%Y-%m-%d %H:%M:%S"),
                    "success": False,
                    "source_ip": f"203.0.113.{int(rng.integers(1, 250))}",
                    "is_after_hours": True,
                }
            )
    tracker.add("CC-08", "accounts with brute-force failure bursts", len(burst_targets))

    accounts = pd.DataFrame(account_rows)
    assignments = pd.DataFrame(assignment_rows)
    logs = pd.DataFrame(log_rows)

    logger.info(
        "Generated %d accounts, %d assignments, %d log records",
        len(accounts),
        len(assignments),
        len(logs),
    )
    return accounts, assignments, logs
