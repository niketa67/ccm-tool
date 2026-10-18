"""Loading and validation of the CCM source extracts.

The tool reads four flat files (the sort of exports you would pull from an HRIS,
an IAM platform and a SIEM) and normalises them into a single
:class:`DataBundle` that every control receives.

Validation is deliberately split in two:

* **Hard errors** raise :class:`DataValidationError` - a missing file, a missing
  mandatory column, duplicate primary keys. A run against these would be
  misleading, so we stop.
* **Warnings** are logged and surfaced in the run summary - broken referential
  integrity (an account pointing at an unknown employee), unparseable dates, and
  so on. These are frequently *the finding*, not an accident, so they must not
  abort the run.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import RAW_DATA_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "DataValidationError",
    "DataBundle",
    "TableSpec",
    "TABLE_SPECS",
    "load_data_bundle",
    "load_table",
    "compute_data_hash",
    "to_bool_series",
]


class DataValidationError(ValueError):
    """Raised when source data is structurally unusable."""


# --------------------------------------------------------------------------- #
# Table specifications
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TableSpec:
    """Describes the shape and semantics of one source CSV.

    Attributes:
        name: Logical name used as the :class:`DataBundle` attribute.
        filename: CSV filename inside the raw data directory.
        required_columns: Columns that must be present or validation fails.
        primary_key: Column that must be unique and non-null.
        date_columns: Columns parsed to dates (time component discarded).
        datetime_columns: Columns parsed to full timestamps.
        bool_columns: Columns coerced to real booleans.
        nullable: Columns allowed to contain blanks (default: all).
    """

    name: str
    filename: str
    required_columns: tuple[str, ...]
    primary_key: str
    date_columns: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    bool_columns: tuple[str, ...] = ()


TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        name="hr_employees",
        filename="hr_employees.csv",
        required_columns=(
            "employee_id",
            "name",
            "email",
            "department",
            "job_title",
            "manager_id",
            "status",
            "hire_date",
            "termination_date",
        ),
        primary_key="employee_id",
        date_columns=("hire_date", "termination_date"),
    ),
    TableSpec(
        name="user_accounts",
        filename="user_accounts.csv",
        required_columns=(
            "account_id",
            "username",
            "employee_id",
            "system",
            "account_type",
            "status",
            "mfa_enabled",
            "created_date",
            "last_login_date",
            "last_password_change",
            "last_access_review_date",
        ),
        primary_key="account_id",
        date_columns=(
            "created_date",
            "last_login_date",
            "last_password_change",
            "last_access_review_date",
        ),
        bool_columns=("mfa_enabled",),
    ),
    TableSpec(
        name="access_assignments",
        filename="access_assignments.csv",
        required_columns=("account_id", "role_name", "system", "granted_date", "granted_by"),
        primary_key="assignment_id",
        date_columns=("granted_date",),
    ),
    TableSpec(
        name="login_logs",
        filename="login_logs.csv",
        required_columns=(
            "log_id",
            "account_id",
            "timestamp",
            "success",
            "source_ip",
            "is_after_hours",
        ),
        primary_key="log_id",
        datetime_columns=("timestamp",),
        bool_columns=("success", "is_after_hours"),
    ),
)

#: Column order controls expect to receive, keyed by table name.
_EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {spec.name: spec.required_columns for spec in TABLE_SPECS}


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #
_TRUE_TOKENS = frozenset({"true", "1", "yes", "y", "t", "on"})


def to_bool_series(series: pd.Series) -> pd.Series:
    """Coerce a series of assorted truthy/falsy values into real booleans.

    ``astype(bool)`` is unsafe here: the string ``"False"`` is truthy in Python,
    and CSVs routinely carry ``"True"/"False"``, ``"1"/"0"`` or blank cells (a
    blank MFA flag means "no evidence of MFA", which we treat as ``False``).

    Args:
        series: Any series representing a boolean flag.

    Returns:
        A nullable ``boolean`` series (``pd.NA`` preserved only for genuinely
        unknown values - in practice we map blanks to ``False``).
    """
    if series.dtype == bool or str(series.dtype) == "boolean":
        return series.astype("boolean")

    # Fast path: pandas already parsed the column as numeric or boolean-ish.
    if pd.api.types.is_numeric_dtype(series):
        return series.map(lambda value: bool(value) if pd.notna(value) else False).astype("boolean")

    normalised = series.astype("string").str.strip().str.casefold()
    mapped = normalised.map(lambda value: value in _TRUE_TOKENS if pd.notna(value) else False)
    return mapped.astype("boolean")


def _coerce_dates(series: pd.Series, column: str) -> tuple[pd.Series, int]:
    """Parse ``series`` as dates, returning the parsed values and a bad-row count."""
    original_notnull = series.notna() & (series.astype("string").str.strip() != "")
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if getattr(parsed.dtype, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    parsed = parsed.dt.normalize()
    unparseable = int((original_notnull & parsed.isna()).sum())
    if unparseable:
        logger.warning("Column '%s': %d value(s) could not be parsed as dates", column, unparseable)
    return parsed, unparseable


def _coerce_datetimes(series: pd.Series, column: str) -> tuple[pd.Series, int]:
    """Parse ``series`` as timestamps, returning the parsed values and a bad-row count."""
    original_notnull = series.notna() & (series.astype("string").str.strip() != "")
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if getattr(parsed.dtype, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    unparseable = int((original_notnull & parsed.isna()).sum())
    if unparseable:
        logger.warning("Column '%s': %d value(s) could not be parsed as timestamps", column, unparseable)
    return parsed, unparseable


def _clean_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from text columns and turn empty strings into missing."""
    for column in frame.columns:
        if frame[column].dtype == object or isinstance(frame[column].dtype, pd.StringDtype):
            cleaned = frame[column].astype("string").str.strip()
            frame[column] = cleaned.mask(cleaned == "", other=pd.NA)
    return frame


# --------------------------------------------------------------------------- #
# Table loading
# --------------------------------------------------------------------------- #
def load_table(spec: TableSpec, directory: Path | str) -> pd.DataFrame:
    """Load and normalise a single source table.

    Args:
        spec: The :class:`TableSpec` describing the table.
        directory: Folder containing the CSV files.

    Returns:
        A cleaned DataFrame.

    Raises:
        DataValidationError: If the file is missing or lacks required columns.
    """
    path = Path(directory) / spec.filename
    if not path.exists():
        raise DataValidationError(
            f"Required data file not found: {path}\n"
            f"Generate the synthetic extract first:  python run_controls.py --generate-data"
        )

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
    except pd.errors.EmptyDataError:
        # An empty table is legitimate for some datasets (e.g. no login history),
        # so long as the header is present. pandas cannot read a header-only file
        # with dtype=str, so fall back to a column-only frame.
        frame = pd.DataFrame(columns=list(spec.required_columns))
        logger.warning("%s is empty; continuing with zero rows", spec.filename)
        return frame

    missing = [column for column in spec.required_columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"{path.name} is missing required column(s): {missing}")

    # Ensure the primary key column exists even for generated id-less tables.
    if spec.primary_key not in frame.columns:
        frame[spec.primary_key] = np.arange(1, len(frame) + 1).astype(str)

    frame = _clean_strings(frame)

    for column in spec.bool_columns:
        frame[column] = to_bool_series(frame[column])
    for column in spec.date_columns:
        frame[column], _ = _coerce_dates(frame[column], column)
    for column in spec.datetime_columns:
        frame[column], _ = _coerce_datetimes(frame[column], column)

    # Keep a predictable column order: spec columns first, extras afterwards.
    ordered = [column for column in spec.required_columns if column in frame.columns]
    ordered += [column for column in frame.columns if column not in ordered]
    frame = frame.loc[:, ordered].reset_index(drop=True)

    logger.debug("Loaded %s: %d rows x %d columns", spec.filename, len(frame), frame.shape[1])
    return frame


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    """Everything a control needs, plus the metadata that makes it auditable.

    Attributes:
        hr_employees: HR master data.
        user_accounts: Directory accounts across all systems.
        access_assignments: Role grants per account.
        login_logs: Authentication events.
        as_of: The reference timestamp controls evaluate "age" against.
        sources: Mapping of logical table name to the file it was read from.
        data_hash: SHA-256 over the source files (evidence integrity seal).
        warnings: Non-fatal data quality issues found during validation.
    """

    hr_employees: pd.DataFrame
    user_accounts: pd.DataFrame
    access_assignments: pd.DataFrame
    login_logs: pd.DataFrame
    as_of: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now().normalize())
    sources: dict[str, Path] = field(default_factory=dict)
    data_hash: str = ""
    warnings: list[str] = field(default_factory=list)

    def table(self, name: str) -> pd.DataFrame:
        """Return a table by logical name."""
        try:
            return getattr(self, name)
        except AttributeError as exc:  # pragma: no cover - programming error guard
            raise KeyError(f"Unknown table '{name}'") from exc

    @property
    def row_counts(self) -> dict[str, int]:
        """Row count per table - useful for run summaries and dashboards."""
        return {
            "hr_employees": len(self.hr_employees),
            "user_accounts": len(self.user_accounts),
            "access_assignments": len(self.access_assignments),
            "login_logs": len(self.login_logs),
        }

    def active_employees(self) -> pd.DataFrame:
        """HR records for employees not flagged as terminated."""
        status = self.hr_employees["status"].astype("string").str.casefold()
        return self.hr_employees.loc[status.ne("terminated")]

    def hr_index(self) -> pd.DataFrame:
        """HR table indexed by ``employee_id`` for fast lookups in controls."""
        return self.hr_employees.set_index("employee_id", drop=False)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate(bundle: DataBundle) -> list[str]:
    """Run cross-table integrity checks, attaching warnings to the bundle.

    Returns:
        The list of warning strings that were produced.
    """
    warnings: list[str] = []

    for spec in TABLE_SPECS:
        frame = bundle.table(spec.name)
        declared = set(_EXPECTED_COLUMNS[spec.name]) | {spec.primary_key}
        extras = [column for column in frame.columns if column not in declared]

        if len(frame) and spec.primary_key in frame.columns:
            duplicated = frame[spec.primary_key].duplicated().sum()
            if duplicated:
                raise DataValidationError(
                    f"{spec.filename}: {duplicated} duplicate value(s) in primary key '{spec.primary_key}'"
                )
            null_keys = int(frame[spec.primary_key].isna().sum())
            if null_keys:
                raise DataValidationError(
                    f"{spec.filename}: {null_keys} null value(s) in primary key '{spec.primary_key}'"
                )

        if extras:
            warnings.append(f"{spec.filename}: unexpected extra column(s) {extras} were retained")

    # Referential integrity: accounts -> employees
    accounts, employees = bundle.user_accounts, bundle.hr_employees
    if len(accounts) and len(employees):
        known = set(employees["employee_id"].dropna().astype(str))
        linked = accounts["employee_id"].dropna().astype(str)
        unmatched = sorted(set(linked) - known)
        dangling = int(linked.isin(unmatched).sum())
        if unmatched:
            warnings.append(
                f"user_accounts: {dangling} account(s) reference {len(unmatched)} employee_id(s) "
                f"absent from HR (expected for orphan-account testing)"
            )

    # Referential integrity: assignments -> accounts
    assignments = bundle.access_assignments
    if len(assignments) and len(accounts):
        known_accounts = set(accounts["account_id"].dropna().astype(str))
        unknown = ~assignments["account_id"].astype(str).isin(known_accounts)
        if int(unknown.sum()):
            warnings.append(
                f"access_assignments: {int(unknown.sum())} row(s) reference unknown account_id(s)"
            )

    # Referential integrity: logs -> accounts
    logs = bundle.login_logs
    if len(logs) and len(accounts):
        unknown = ~logs["account_id"].astype(str).isin(set(accounts["account_id"].astype(str)))
        if int(unknown.sum()):
            warnings.append(f"login_logs: {int(unknown.sum())} row(s) reference unknown account_id(s)")

    for message in warnings:
        logger.warning("Data quality: %s", message)
    return warnings


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def compute_data_hash(paths: Iterable[Path | str]) -> str:
    """Compute a SHA-256 digest over a set of files.

    The digest covers each file's *name* and *bytes*, so a rename changes the
    hash just as an edit does. The result is stored on every run to prove the
    evidence was produced from a specific, unaltered snapshot of the source data.

    Args:
        paths: Files to include.

    Returns:
        A lowercase hex digest.
    """
    digest = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: p.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        if path.exists():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:  # pragma: no cover - defensive
            digest.update(b"<missing>")
        digest.update(b"\x1e")
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_data_bundle(
    raw_dir: Path | str | None = None,
    *,
    as_of: datetime | pd.Timestamp | str | None = None,
    validate: bool = True,
) -> DataBundle:
    """Load all source tables and return a validated :class:`DataBundle`.

    Args:
        raw_dir: Directory holding the CSVs. Defaults to ``data/raw``.
        as_of: Reference timestamp for age-based tests. Defaults to today at
            midnight, so results are stable within a calendar day.
        validate: When ``True`` (default) run cross-table integrity checks.

    Returns:
        A populated :class:`DataBundle`.

    Raises:
        DataValidationError: If any table is missing or structurally invalid.
    """
    directory = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    if not directory.exists():
        raise DataValidationError(
            f"Raw data directory not found: {directory}\n"
            f"Generate the synthetic extract first:  python run_controls.py --generate-data"
        )

    tables: dict[str, pd.DataFrame] = {}
    sources: dict[str, Path] = {}
    for spec in TABLE_SPECS:
        tables[spec.name] = load_table(spec, directory)
        sources[spec.name] = directory / spec.filename

    if as_of is None:
        reference = pd.Timestamp.now().normalize()
    else:
        reference = pd.Timestamp(as_of)
        if reference.tzinfo is not None:
            reference = reference.tz_convert(None)
        reference = reference.normalize()

    bundle = DataBundle(
        as_of=reference,
        sources=sources,
        data_hash=compute_data_hash(sources.values()),
        **tables,
    )
    if validate:
        bundle.warnings = _validate(bundle)
    return bundle
