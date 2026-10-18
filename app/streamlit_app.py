"""Streamlit dashboard for the Continuous Controls Monitoring tool.

Launch with:

    streamlit run app/streamlit_app.py

Five views, navigated from the sidebar:

1. **Overview**             - KPI cards, exceptions-by-control bar, pass/fail donut.
2. **Control Results**      - per-control outcomes plus their exception detail.
3. **Exceptions Workbench** - filterable findings with persistable workflow status.
4. **Trends**               - exception counts per control across historical runs.
5. **Run and Export**       - re-run from the UI, download evidence, view config.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The app script lives in ``app/``; make the ``src`` package importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pandas as pd

from ccm import __version__
from ccm.config import load_controls_config
from ccm.engine import run_controls
from ccm.loader import DataValidationError, load_data_bundle
from ccm.paths import RAW_DATA_DIR
from ccm.reporting import ReportWriter
from ccm.storage import EXCEPTION_STATUSES, Storage

# Must be the first Streamlit call in the file.
st.set_page_config(
    page_title="CCM - Controls Monitoring",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
GREEN = "#2e7d32"
AMBER = "#ed9c28"
RED = "#c62828"
GREY = "#78909c"
INK = "#1f2430"

STATUS_COLOURS = {"PASS": GREEN, "FAIL": AMBER, "ERROR": GREY}
SEVERITY_COLOURS = {
    "Critical": "#c62828",
    "High": "#ef6c00",
    "Medium": "#f9a825",
    "Low": "#1565c0",
}
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]

NAVIGATION = {
    "Overview": 1,
    "Control Results": 2,
    "Exceptions Workbench": 3,
    "Trends": 4,
    "Run and Export": 5,
}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_storage() -> Storage:
    """One persistent SQLite connection shared by the whole session."""
    return Storage()


@st.cache_data(show_spinner="Loading source extracts...")
def get_bundle(cache_key: tuple) -> object:
    """Load and validate the source data; invalidated via a file signature."""
    return load_data_bundle(RAW_DATA_DIR)


def _raw_signature() -> tuple:
    """Cheap signature of the raw files, used to bust the data cache."""
    return tuple(
        (path.name, path.stat().st_size, path.stat().st_mtime)
        for path in sorted(RAW_DATA_DIR.glob("*.csv"))
    )


@st.cache_data(show_spinner=False)
def load_run(run_id: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(control_results, exceptions)`` for one run."""
    storage = get_storage()
    if run_id is None:
        return pd.DataFrame(columns=["control_id"]), pd.DataFrame()
    return storage.get_results(run_id), storage.get_exceptions(run_id=run_id)


@st.cache_data(show_spinner=False)
def load_trend(limit: int) -> pd.DataFrame:
    """Rows of ``run x control`` exception counts, oldest first."""
    return get_storage().exception_trend(limit=limit)


# --------------------------------------------------------------------------- #
# Small presentational helpers
# --------------------------------------------------------------------------- #
def kpi_card(label: str, value: object, colour: str) -> str:
    """Return the HTML for one KPI card.

    Text colours are theme-independent (Streamlit's default theme may be dark or
    light), so labels use mid-gray and the caller supplies an accent colour only
    for the value.
    """
    return (
        f'''<div style="background:rgba(127,140,160,0.12);border:1px solid rgba(127,140,160,0.35);border-radius:10px;padding:12px 14px;min-width:0;overflow:hidden">
        <div style="font-size:10.5px;color:#90a4ae;white-space:normal;line-height:1.3">{label}</div>
        <div style="font-size:28px;font-weight:700;color:{colour};margin-top:4px;white-space:nowrap;line-height:1.15">{value}</div></div>'''
    )


def status_badge(status: str) -> str:
    """Return the HTML badge for a control status."""
    colour = STATUS_COLOURS.get(status, GREY)
    return (
        f'<span style="color:#fff;background:{colour};padding:2px 10px;'
        f'border-radius:999px;font-size:11px;font-weight:700">{status}</span>'
    )


def severity_badge(severity: str) -> str:
    """Return the HTML badge for a severity label."""
    colour = SEVERITY_COLOURS.get(severity, GREY)
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 10px;'
        f'border-radius:999px;font-size:11px;font-weight:600">{severity}</span>'
    )


def _plotly_layout(fig: go.Figure, *, title: str, height: int) -> None:
    """Apply the dashboard-wide chart styling."""
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=46, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=INK),
        yaxis=dict(gridcolor="#eceff3", zerolinecolor="#eceff3"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )


def _clean(obj: object) -> object:
    """Map pandas missing values to ``None`` for display."""
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        return obj
    return obj


def render_header() -> None:
    """Draw the application masthead."""
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:14px;margin-bottom:6px">
          <span style="font-size:34px">&#128737;</span>
          <div>
            <div style="font-size:26px;font-weight:700;line-height:1.1">
              Continuous Controls Monitoring</div>
            <div style="font-size:13px;color:#607d8b">
              Automated ITGC control testing &middot; v{__version__}</div>
          </div></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> tuple[str, int | None]:
    """Return ``(selected_view_label, selected_run_id)``."""
    with st.sidebar:
        runs = get_storage().list_runs()

        labels: dict[str, int] = {}
        for _, row in runs.iterrows():
            stamp = str(row["executed_at"])[:19].replace("T", " ")
            # Streamlit renders selectbox labels as plain text, so use a literal
            # middle dot rather than an HTML entity.
            labels[f"#{int(row['run_id'])} · {stamp} · {int(row['total_exceptions'])} exceptions"] = int(row["run_id"])

        st.markdown("###")
        view = st.radio("View", list(NAVIGATION), label_visibility="collapsed")

        st.divider()
        st.markdown("**Monitoring run**")
        if labels:
            chosen = st.selectbox("Run", list(labels), label_visibility="collapsed")
            run_id = labels[chosen]
        else:
            run_id = None
            st.info("No runs stored yet. Open **Run and Export** to create one.")

        st.divider()
        if run_id:
            row = runs.loc[runs["run_id"] == run_id].iloc[0]
            st.caption(
                f"Data snapshot SHA-256\n\n`{str(row['data_snapshot_hash'])[:20]}…`\n\n"
                f"Source: `{RAW_DATA_DIR}`"
            )
        else:
            st.caption(f"Source: `{RAW_DATA_DIR}`")
    return view, run_id


# --------------------------------------------------------------------------- #
# View 1 - Overview
# --------------------------------------------------------------------------- #
def render_overview(run_id: int | None) -> None:
    """KPIs plus severity-coloured exception and outcome charts."""
    results, exceptions = load_run(run_id)
    if results.empty:
        st.info("Run a monitoring cycle to populate this view.")
        return

    st.markdown("### 1. Overview")

    passed = int((results["status"] == "PASS").sum())
    failed = int((results["status"] == "FAIL").sum())
    errored = int((results["status"] == "ERROR").sum())
    total = len(results)
    exception_total = int(len(exceptions))
    effectiveness = round(100.0 * passed / total, 1) if total else 0.0

    columns = st.columns(5, gap="small")
    cards = [
        ("Controls", total, GREY),
        ("Passed", passed, GREEN if passed else GREY),
        ("Failed", failed, AMBER if failed else GREY),
        ("Exceptions", exception_total, RED if exception_total else GREY),
        # "Effective" rather than "Effectiveness": the word must fit a 130px card.
        ("Effective", f"{effectiveness}%", GREEN if effectiveness >= 80 else AMBER),
    ]
    for column, (label, value, colour) in zip(columns, cards):
        column.markdown(kpi_card(label, value, colour), unsafe_allow_html=True)

    chart_columns = st.columns([3, 2])

    by_control = (
        results.groupby(["control_id", "severity"], as_index=False)["exception_count"]
        .sum()
        .sort_values("exception_count", ascending=False)
        .reset_index(drop=True)
    )
    fig_bar = px.bar(
        by_control,
        x="control_id",
        y="exception_count",
        color="severity",
        color_discrete_map=SEVERITY_COLOURS,
        category_orders={"severity": SEVERITY_ORDER},
        labels={"control_id": "Control", "exception_count": "Exceptions", "severity": "Severity"},
    )
    _plotly_layout(fig_bar, title="Exceptions by control, by severity", height=380)
    chart_columns[0].plotly_chart(fig_bar, key="overview_bar")

    fig_donut = go.Figure(
        go.Pie(
            labels=["Passed", "Failed", "Error"],
            values=[passed, failed, errored],
            hole=0.6,
            marker=dict(colors=[GREEN, AMBER, GREY]),
            sort=False,
        )
    )
    fig_donut.update_traces(textinfo="percent", textposition="inside", insidetextorientation="horizontal")
    fig_donut.update_layout(
        title="Control outcomes",
        height=380,
        margin=dict(l=10, r=10, t=46, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=12, color=INK)),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=INK),
    )
    chart_columns[1].plotly_chart(fig_donut, key="overview_donut")

    with st.container(border=True):
        st.markdown("#### Per-control results")
        slim = results[["control_id", "control_name", "severity", "status", "total_tested", "exception_count"]].copy()
        slim["exception_rate"] = (slim["exception_count"] / slim["total_tested"] * 100).map(
            lambda value: f"{value:.2f}%" if pd.notna(value) else "-"
        )
        slim["status_badge"] = slim["status"].map(status_badge)
        slim["severity_badge"] = slim["severity"].map(severity_badge)
        st.dataframe(
            slim[
                ["control_id", "control_name", "severity_badge", "status_badge", "total_tested", "exception_count", "exception_rate"]
            ].rename(
                columns={
                    "severity_badge": "Severity",
                    "status_badge": "Status",
                    "total_tested": "Tested",
                    "exception_count": "Exceptions",
                }
            ),
            hide_index=True,
        )


# --------------------------------------------------------------------------- #
# View 2 - Control Results
# --------------------------------------------------------------------------- #
def render_control_results(run_id: int | None) -> None:
    """Per-control table plus the selected control's exception detail."""
    results, exceptions = load_run(run_id)
    if results.empty:
        st.info("No stored results yet. Open **Run and Export**.")
        return

    st.markdown("### 2. Control results")

    display = results.copy()
    display["Status"] = display["status"].map(status_badge)
    display["Severity"] = display["severity"].map(severity_badge)
    display["Rate"] = (display["exception_rate"] * 100).map(lambda value: f"{value:.2f}%")
    st.dataframe(
        display[
            ["control_id", "control_name", "Severity", "Status", "total_tested", "exception_count", "Rate", "framework_mapping"]
        ].rename(
            columns={
                "control_id": "Control",
                "control_name": "Name",
                "total_tested": "Tested",
                "exception_count": "Exceptions",
                "framework_mapping": "Framework mapping",
            }
        ),
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn(width="large"),
            "Framework mapping": st.column_config.TextColumn(width="large"),
        },
    )

    st.divider()
    control_id = st.selectbox(
        "Inspect a control",
        list(results["control_id"]),
        format_func=lambda cid: f"{cid} - {results.loc[results['control_id'] == cid, 'control_name'].iloc[0]}",
    )
    row = results.loc[results["control_id"] == control_id].iloc[0]

    st.markdown(f"#### {row['control_name']}")
    st.write(row["summary_text"])
    st.caption(str(row["risk_category"]))
    st.markdown(f"**Severity:** {row['severity']}")
    st.markdown(f"**Framework mapping:** {row['framework_mapping']}")
    if _clean(row["error"]):
        st.error(f"Evaluation error: {row['error']}")

    with st.expander("Thresholds applied in this run"):
        st.code(str(row["parameters_json"]), language="json")

    detail = exceptions.loc[exceptions["control_id"] == control_id]
    st.markdown(f"#### Exceptions ({len(detail):,})")
    if detail.empty:
        st.success("No exceptions identified for this control.")
        return
    st.dataframe(detail, hide_index=True, height=360)


# --------------------------------------------------------------------------- #
# View 3 - Exceptions Workbench
# --------------------------------------------------------------------------- #
def render_workbench(run_id: int | None) -> None:
    """Filterable exceptions with persistable workflow status."""
    st.markdown("### 3. Exceptions workbench")

    exceptions = load_run(run_id)[1]
    if exceptions is None or exceptions.empty:
        st.info("No exceptions recorded for the selected run.")
        return

    columns = st.columns(4)
    with columns[0]:
        control_filter = st.multiselect("Control", sorted(exceptions["control_id"].unique()), placeholder="All")
    with columns[1]:
        system_filter = st.multiselect(
            "System", sorted(str(value) for value in exceptions["system"].dropna().unique()), placeholder="All"
        )
    with columns[2]:
        severity_filter = st.multiselect(
            "Severity", [sev for sev in SEVERITY_ORDER if sev in set(exceptions["severity"])], placeholder="All"
        )
    with columns[3]:
        status_filter = st.multiselect(
            "Status", list(EXCEPTION_STATUSES), default=["Open", "Acknowledged"], placeholder="All"
        )

    filtered = exceptions.copy()
    if control_filter:
        filtered = filtered.loc[filtered["control_id"].isin(control_filter)]
    if system_filter:
        filtered = filtered.loc[filtered["system"].isin(system_filter)]
    if severity_filter:
        filtered = filtered.loc[filtered["severity"].isin(severity_filter)]
    if status_filter:
        filtered = filtered.loc[filtered["status"].isin(status_filter)]

    if filtered.empty:
        st.warning("No exceptions match the current filters.")
        return

    st.caption(f"Showing {len(filtered):,} of {len(exceptions):,} exceptions.")

    columns_to_show = [
        "exception_id", "control_id", "account_id", "username", "employee_id",
        "system", "severity", "reason", "status",
    ]
    editable = filtered[columns_to_show].copy().sort_values("exception_id")

    edited = st.data_editor(
        editable,
        key=f"workbench_{run_id}",
        hide_index=True,
        height=380,
        disabled=[column for column in columns_to_show if column != "status"],
        column_config={
            "exception_id": st.column_config.TextColumn("ID", width="small"),
            "severity": st.column_config.SelectboxColumn(
                label="Severity", options=SEVERITY_ORDER, disabled=True, width="small"
            ),
            "status": st.column_config.SelectboxColumn(
                label="Status", options=list(EXCEPTION_STATUSES), required=True, width="small"
            ),
            "reason": st.column_config.TextColumn(width="large"),
        },
    )

    changed: list[tuple[int, str]] = []
    for _, row in edited.iterrows():
        original = filtered.loc[filtered["exception_id"] == row["exception_id"], "status"]
        if not original.empty and str(original.iloc[0]) != str(row["status"]):
            changed.append((int(row["exception_id"]), str(row["status"])))

    if changed and st.button(f"Apply {len(changed)} status change(s)", type="primary"):
        storage = get_storage()
        for exception_id, status in changed:
            storage.update_exception_status(exception_id, status)
        load_run.clear()
        st.toast(f"{len(changed)} exception(s) updated.")
        st.success(
            f"Updated {len(changed)} exception(s) to "
            f"{', '.join(sorted({status for _, status in changed}))}."
        )

    with st.expander("Status codes"):
        st.markdown(
            "**Open** - identified, not yet triaged.  \n"
            "**Acknowledged** - risk accepted or investigation under way.  \n"
            "**Remediated** - corrected; evidence retained for the auditor."
        )


# --------------------------------------------------------------------------- #
# View 4 - Trends
# --------------------------------------------------------------------------- #
def render_trends(run_id: int | None) -> None:
    """Exception counts per control across historical runs."""
    trend = load_trend(limit=30)
    if trend.empty:
        st.info("No historical data yet. Run the controls repeatedly to build a trend.")
        return

    st.markdown("### 4. Trends")
    st.caption("Each point is an exception count from one monitoring run, showing whether remediation is actually closing findings.")

    labelled = trend.copy()
    labelled["run_label"] = labelled["run_id"].map(lambda value: f"#{int(value)}")

    fig_line = px.line(
        labelled,
        x="run_label",
        y="exception_count",
        color="control_id",
        markers=True,
        labels={"run_label": "Monitoring run", "exception_count": "Exceptions", "control_id": "Control"},
    )
    _plotly_layout(fig_line, title="Exception count by control across runs", height=420)
    st.plotly_chart(fig_line, key="trend_line")

    runs_sorted = sorted(set(trend["run_id"]))
    if len(runs_sorted) >= 2:
        latest_run, previous_run = runs_sorted[-1], runs_sorted[-2]
        latest = (
            trend.loc[trend["run_id"] == latest_run, ["control_id", "severity", "exception_count"]]
            .rename(columns={"exception_count": "current"})
        )
        prior = (
            trend.loc[trend["run_id"] == previous_run, ["control_id", "exception_count"]]
            .rename(columns={"exception_count": "previous"})
        )
        movement = latest.merge(prior, on="control_id", how="outer").fillna(0)
        movement["current"] = movement["current"].astype(int)
        movement["previous"] = movement["previous"].astype(int)
        movement["delta"] = movement["current"] - movement["previous"]

        def _colour(value: object) -> str:
            """Colour a delta red when rising and green when falling."""
            number = _clean(value)
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                if number > 0:
                    return f"color:{RED}"
                if number < 0:
                    return f"color:{GREEN}"
            return ""

        st.markdown(f"#### Run #{latest_run} vs run #{previous_run}")
        st.dataframe(
            movement.style.map(_colour, subset=["delta"]),
            hide_index=True,
        )

    runs = get_storage().list_runs()
    st.markdown("#### Run history")
    st.dataframe(
        runs[
            ["run_id", "executed_at", "data_snapshot_hash", "total_controls", "passed", "failed", "total_exceptions"]
        ],
        hide_index=True,
        height=260,
        column_config={"data_snapshot_hash": st.column_config.TextColumn("Data SHA-256", width="medium")},
    )


# --------------------------------------------------------------------------- #
# View 5 - Run and Export
# --------------------------------------------------------------------------- #
def render_run_and_export(run_id: int | None) -> None:
    """Trigger a fresh monitoring run and download its evidence."""
    st.markdown("### 5. Run and export")
    st.caption(
        "Re-running executes every enabled control against the current `data/raw` "
        "extracts, stores the results in SQLite and regenerates the evidence pack."
    )

    config = load_controls_config()

    left, right = st.columns(2)
    if left.button("Run all controls", type="primary", key="run_all"):
        try:
            with st.spinner("Executing controls..."):
                bundle = get_bundle(_raw_signature())
                summary = run_controls(bundle, config=config)
                with Storage() as storage:
                    persisted_id = storage.save_run(summary)
                ReportWriter(summary, config=config).write_all()
            load_run.clear()
            load_trend.clear()
            st.success(
                f"Run #{persisted_id}: {summary.total_controls} controls - {summary.passed} PASS / "
                f"{summary.failed} FAIL / {summary.error_count} ERROR, {summary.total_exceptions} exceptions."
            )
            st.dataframe(summary.as_table(), hide_index=True)
        except DataValidationError as exc:
            st.error(f"Could not run controls: {exc}")

    if right.button("Generate synthetic data", key="generate"):
        from ccm.data_generator import generate_dataset

        with st.spinner("Generating seeded synthetic data..."):
            generate_dataset()
        load_run.clear()
        get_bundle.clear()
        st.info(
            f"Regenerated `{RAW_DATA_DIR}`. Press **Run all controls** to evaluate it."
        )

    st.divider()
    st.markdown("#### Download evidence")

    runs = get_storage().list_runs()
    if runs.empty:
        st.info("Nothing to download yet - run the controls first.")
    else:
        run_id = int(runs.iloc[0]["run_id"])
        try:
            writer = ReportWriter(_summary_for_run(run_id), config=config)
            html_path = writer.folder / "evidence_report.html"
            with st.spinner("Rendering evidence pack..."):
                writer.write_all()
            csv_path = writer.folder / f"exceptions_all_{writer.stamp}.csv"

            export_left, export_right = st.columns(2)
            if html_path.exists():
                export_left.download_button(
                    "Download HTML report",
                    data=html_path.read_bytes(),
                    file_name=html_path.name,
                    mime="text/html",
                    key=f"download_html_{run_id}",
                )
            if csv_path.exists():
                export_right.download_button(
                    "Download all exceptions (CSV)",
                    data=csv_path.read_bytes(),
                    file_name=csv_path.name,
                    mime="text/csv",
                    key=f"download_csv_{run_id}",
                )
            st.caption(f"Evidence folder: `{writer.folder}`")
        except Exception as exc:  # pragma: no cover - display-level guard
            st.error(f"Could not build evidence pack: {exc}")

    with st.expander("Current controls.yaml configuration"):
        st.code(load_controls_config().as_dict(), language="yaml")

    with st.expander("How to add a control"):
        st.code(
            "1. Create src/ccm/controls/cc09_my_check.py:\n\n"
    + "@register_control\nclass MyCheck(BaseControl):\n"
    + "    control_id = \"CC-09\"\n"
    + "    name = \"My Check\"\n"
    + "    severity = \"Medium\"\n"
    + "    default_parameters = {\"threshold\": 7}\n\n"
    + "    def population(self, data):\n"
    + "        return data.user_accounts\n\n"
    + "    def evaluate(self, data):\n"
    + "        ...\n"
    + "        return violations                      # includes a `reason` column\n\n"
    + "2. Import it in src/ccm/controls/__init__.py\n\n"
    + "3. Add a CC-09 block to config/controls.yaml            # threshold + severity + mapping\n\n"
    + "Done - the engine, dashboard and reports pick it up automatically.",
            language="python",
        )


def _summary_for_run(run_id: int):
    """Rebuild a summary object from stored rows for the report writer."""
    from ccm.controls.base import ControlResult
    from ccm.engine import RunSummary

    storage = get_storage()
    run = storage.get_run(run_id)
    results_frame = storage.get_results(run_id)
    if run is None or results_frame.empty:
        raise ValueError(f"Run {run_id} has no stored results")

    exceptions = storage.get_exceptions(run_id=run_id)
    results: list[ControlResult] = []
    for _, row in results_frame.iterrows():
        control_exceptions = exceptions.loc[exceptions["control_id"] == row["control_id"]].drop(
            columns=[column for column in ("control_severity", "control_name", "run_executed_at") if column in exceptions.columns],
            errors="ignore",
        )
        results.append(
            ControlResult(
                control_id=row["control_id"],
                status=str(row["status"]),
                total_tested=int(row["total_tested"]),
                exception_count=int(row["exception_count"]),
                exception_rate=float(row["exception_rate"]),
                exceptions=control_exceptions,
                executed_at=pd.Timestamp(row["executed_at"]).to_pydatetime(),
                parameters_used=_json_load(str(row["parameters_json"])),
                summary_text=str(row["summary_text"]),
                name=str(row["control_name"]),
                description="",
                risk_category=str(row["risk_category"]),
                severity=str(row["severity"]),
                framework_mapping=tuple(
                    part.strip() for part in str(row["framework_mapping"]).split(";") if part.strip()
                ),
                duration_ms=int(row["duration_ms"]),
                error=row["error"],
            )
        )

    return RunSummary(
        executed_at=str(run["executed_at"]),
        as_of=str(run["as_of"]),
        data_hash=str(run["data_snapshot_hash"]),
        run_id=run_id,
        results=results,
        warnings=[str(item) for item in _json_load(str(run["warnings"]))],
        error_count=sum(1 for item in results if item.status == "ERROR"),
    )


def _json_load(text: str):
    """Parse a JSON string, tolerating invalid or empty values."""
    import json

    try:
        return json.loads(text or "[]")
    except (ValueError, TypeError):
        return []


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
HANDLERS = {
    "Overview": render_overview,
    "Control Results": render_control_results,
    "Exceptions Workbench": render_workbench,
    "Trends": render_trends,
    "Run and Export": render_run_and_export,
}


def main() -> None:
    """Render the dashboard."""
    render_header()
    view, run_id = render_sidebar()
    st.divider()
    HANDLERS[view](run_id)


if __name__ == "__main__":
    main()
