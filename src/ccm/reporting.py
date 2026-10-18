"""Evidence report generation.

An auditor needs three things from a monitoring tool: *what ran*, *against which
data*, and *what was found*. This module turns a :class:`~ccm.engine.RunSummary`
into Markdown, HTML and CSV artefacts that answer all three, with the data
snapshot hash printed on the front page so a reviewer can re-derive the inputs.

Outputs land in ``reports/`` with timestamped filenames::

    reports/run_2026-06-01_143210/
        evidence_report.html
        evidence_report.md
        exceptions_CC-01_2026-06-01_143210.csv
        exceptions_all_2026-06-01_143210.csv
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import ControlsConfig, load_controls_config
from .engine import RunSummary
from .paths import REPORT_DIR, ensure_directories

logger = logging.getLogger(__name__)

__all__ = ["ReportWriter", "generate_reports"]

_STATUS_BADGE = {
    "PASS": '<span class="badge pass">PASS</span>',
    "FAIL": '<span class="badge fail">FAIL</span>',
    "ERROR": '<span class="badge error">ERROR</span>',
}

_SEVERITY_CLASS = {
    "Critical": "crit",
    "High": "high",
    "Medium": "med",
    "Low": "low",
}


def _escape(value: object) -> str:
    """HTML-escape a value for safe embedding in the report."""
    text = "" if value is None else str(value)
    if text in ("nan", "<NA>", "NaT"):
        return "&mdash;"
    return html.escape(text)


def _md_cell(value: object) -> str:
    """Render a value as a Markdown table cell."""
    text = "" if value is None else str(value)
    if text in ("nan", "<NA>", "NaT"):
        return "-"
    return text.replace("|", "\\|").replace("\n", " ")


def _pct(rate: float) -> str:
    """Format an exception rate as a percentage string."""
    return f"{rate * 100:.2f}%"


class ReportWriter:
    """Builds Markdown, HTML and CSV evidence artefacts for one run.

    Args:
        summary: The run to report on.
        config: Config snapshot embedded in the report (defaults to a fresh load).
        output_dir: Destination folder; defaults to ``reports/``.
    """

    def __init__(
        self,
        summary: RunSummary,
        config: ControlsConfig | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        """Store the run and prepare the output directory."""
        self.summary = summary
        self.config = config or load_controls_config()
        self.output_dir = Path(output_dir) if output_dir is not None else REPORT_DIR
        ensure_directories()

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        self.stamp = stamp
        self.folder = self.output_dir / f"run_{stamp}_{summary.run_id or 'adhoc'}"
        self.folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def write_all(self) -> dict[str, Path]:
        """Write Markdown, HTML and CSV evidence for this run.

        Returns:
            Mapping of artefact name to the path that was written.
        """
        written: dict[str, Path] = {}
        written["markdown"] = self.write_markdown()
        written["html"] = self.write_html()
        written["exceptions_csv"] = self.write_exceptions_csv()
        written.update(self.write_per_control_csvs())
        logger.info("Evidence report written to %s", self.folder)
        return written

    def write_markdown(self) -> Path:
        """Write ``evidence_report.md``."""
        path = self.folder / "evidence_report.md"
        path.write_text(self._markdown(), encoding="utf-8")
        return path

    def write_html(self) -> Path:
        """Write ``evidence_report.html``."""
        path = self.folder / "evidence_report.html"
        path.write_text(self._html(), encoding="utf-8")
        return path

    def write_exceptions_csv(self, control_id: str | None = None) -> Path:
        """Write a CSV of exceptions, optionally restricted to one control.

        Args:
            control_id: When given, export only that control's exceptions.

        Returns:
            Path of the CSV that was written.
        """
        frame = self.summary.exceptions_frame
        if control_id:
            frame = frame.loc[frame["control_id"] == control_id]
        label = f"exceptions_{control_id or 'all'}_{self.stamp}.csv"
        path = self.folder / label
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def write_per_control_csvs(self) -> dict[str, Path]:
        """Write one CSV per failing control, the format auditors ask for."""
        written: dict[str, Path] = {}
        for result in self.summary.results:
            if result.exception_count == 0:
                continue
            written[result.control_id] = self.write_exceptions_csv(result.control_id)
        return written

    # ------------------------------------------------------------------ #
    # Markdown
    # ------------------------------------------------------------------ #
    def _markdown(self) -> str:
        """Compose the Markdown report body."""
        summary = self.summary
        lines: list[str] = []
        add = lines.append

        add("# Continuous Controls Monitoring - Evidence Report")
        add("")
        add(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_")
        add("")
        add("## 1. Run metadata")
        add("")
        add("| Field | Value |")
        add("| --- | --- |")
        add(f"| Run ID | {summary.run_id if summary.run_id else 'not persisted'} |")
        add(f"| Executed at | {summary.executed_at} |")
        add(f"| Data as-of date | {summary.as_of} |")
        add(f"| Controls executed | {summary.total_controls} |")
        add(f"| Passed | {summary.passed} |")
        add(f"| Failed | {summary.failed} |")
        add(f"| Errors | {summary.error_count} |")
        add(f"| Total exceptions | {summary.total_exceptions} |")
        add(f"| Control effectiveness | {summary.effectiveness}% |")
        add(f"| Data snapshot SHA-256 | `{summary.data_hash}` |")
        add("")

        if summary.warnings:
            add("### Data quality warnings")
            add("")
            for warning in summary.warnings:
                add(f"- {warning}")
            add("")

        add("## 2. Control results")
        add("")
        add("| Control | Name | Severity | Status | Tested | Exceptions | Rate | Framework mapping |")
        add("| --- | --- | --- | --- | ---: | ---: | ---: | --- |")
        for result in summary.results:
            add(
                f"| {result.control_id} | {_md_cell(result.name)} | {result.severity} | "
                f"**{result.status}** | {result.total_tested:,} | {result.exception_count:,} | "
                f"{_pct(result.exception_rate)} | {_md_cell('; '.join(result.framework_mapping))} |"
            )
        add("")

        add("## 3. Control descriptions and thresholds")
        add("")
        for result in summary.results:
            add(f"### {result.control_id} - {result.name}")
            add("")
            add(f"- **Risk category:** {result.risk_category}")
            add(f"- **Severity:** {result.severity}")
            add(f"- **Framework mapping:** {'; '.join(result.framework_mapping)}")
            add(f"- **Outcome:** {result.summary_text}")
            if result.error:
                add(f"- **Error:** {result.error}")
            add(f"- **Parameters:** `{_md_cell(result.parameters_used)}`")
            add("")

        add("## 4. Exception detail")
        add("")
        for result in summary.results:
            add(f"### {result.control_id} - {result.name} ({result.exception_count} exception(s))")
            add("")
            if result.exception_count == 0:
                add("_No exceptions identified._")
                add("")
                continue
            frame = result.exceptions
            header = [column for column in frame.columns if column != "severity"]
            add("| " + " | ".join(header) + " |")
            add("| " + " | ".join(["---"] * len(header)) + " |")
            for _, row in frame.iterrows():
                add("| " + " | ".join(_md_cell(row[column]) for column in header) + " |")
            add("")

        add("## 5. Configuration snapshot")
        add("")
        add("```yaml")
        add(_dump_config_yaml(self.config))
        add("```")
        add("")
        add("---")
        add("_Automated evidence generated by the CCM tool. Retain with audit working papers._")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    # HTML
    # ------------------------------------------------------------------ #
    def _html(self) -> str:
        """Compose a self-contained HTML report (no external assets)."""
        summary = self.summary

        rows: list[str] = []
        for result in summary.results:
            badge = _STATUS_BADGE.get(result.status, result.status)
            sev_class = _SEVERITY_CLASS.get(result.severity, "low")
            rows.append(
                "<tr>"
                f"<td><b>{_escape(result.control_id)}</b></td>"
                f"<td>{_escape(result.name)}</td>"
                f'<td class="sev {sev_class}">{_escape(result.severity)}</td>'
                f"<td>{badge}</td>"
                f'<td class="num">{result.total_tested:,}</td>'
                f'<td class="num">{result.exception_count:,}</td>'
                f'<td class="num">{_pct(result.exception_rate)}</td>'
                f'<td class="map">{_escape("; ".join(result.framework_mapping))}</td>'
                "</tr>"
            )

        detail_blocks: list[str] = []
        for result in summary.results:
            if result.exception_count == 0:
                detail_blocks.append(
                    f"<h3>{_escape(result.control_id)} &ndash; {_escape(result.name)}</h3>"
                    '<p class="ok">No exceptions identified.</p>'
                )
                continue
            frame = result.exceptions
            header = [column for column in frame.columns if column != "severity"]
            thead = "".join(f"<th>{_escape(column)}</th>" for column in header)
            body = "".join(
                "<tr>" + "".join(f"<td>{_escape(row[column])}</td>" for column in header) + "</tr>"
                for _, row in frame.iterrows()
            )
            detail_blocks.append(
                f"<h3>{_escape(result.control_id)} &ndash; {_escape(result.name)} "
                f'<span class="badge fail">{result.exception_count}</span></h3>'
                f'<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>'
            )

        warnings_html = ""
        if summary.warnings:
            items = "".join(f"<li>{_escape(warning)}</li>" for warning in summary.warnings)
            warnings_html = f"<h2>Data quality warnings</h2><ul>{items}</ul>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CCM Evidence Report - {html.escape(summary.executed_at[:19])}</title>
<style>
  :root {{ --bg:#f7f8fa; --ink:#1f2430; --muted:#5b6270; --line:#d9dee7;
          --green:#1c7c3c; --red:#c0392b; --amber:#b9770e; --grey:#6b7280; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:32px; background:var(--bg); color:var(--ink);
         font:14px/1.55 "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; background:#fff; padding:32px 40px;
          border:1px solid var(--line); border-radius:10px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin-top:32px; border-bottom:2px solid var(--line); padding-bottom:6px; }}
  h3 {{ font-size:15px; margin-top:24px; }}
  .sub {{ color:var(--muted); margin:0 0 24px; font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0 24px; font-size:13px; }}
  th, td {{ border:1px solid var(--line); padding:7px 10px; text-align:left;
           vertical-align:top; word-break:break-word; }}
  th {{ background:#eef1f6; font-weight:600; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.map {{ font-size:12px; color:var(--muted); }}
  .badge {{ display:inline-block; padding:2px 9px; border-radius:999px;
           font-size:11px; font-weight:700; letter-spacing:.04em; }}
  .badge.pass {{ background:#e3f4e9; color:var(--green); }}
  .badge.fail {{ background:#fdecea; color:var(--red); }}
  .badge.error {{ background:#eceff3; color:var(--grey); }}
  .sev.crit {{ color:#fff; background:var(--red); }}
  .sev.high {{ color:#fff; background:var(--amber); }}
  .sev.med  {{ color:#4a3a00; background:#fff2cc; }}
  .sev.low  {{ color:#12314a; background:#dbeafe; }}
  .kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 8px; }}
  .kpi {{ flex:1 1 140px; border:1px solid var(--line); border-radius:8px;
         padding:12px 14px; background:#fbfcfd; }}
  .kpi .label {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em;
                color:var(--muted); }}
  .kpi .value {{ font-size:22px; font-weight:700; margin-top:2px; }}
  .hash {{ font-family:Consolas, "Courier New", monospace; font-size:12px;
          background:#f2f4f8; padding:2px 6px; border-radius:4px; }}
  code {{ font-family:Consolas, "Courier New", monospace; font-size:12px; }}
  pre {{ background:#f6f8fa; padding:14px; border-radius:6px; overflow:auto;
        font-size:12px; }}
  p.ok {{ color:var(--green); font-weight:600; }}
  footer {{ margin-top:32px; padding-top:14px; border-top:1px solid var(--line);
           color:var(--muted); font-size:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Continuous Controls Monitoring &ndash; Evidence Report</h1>
  <p class="sub">Generated {_escape(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))}</p>

  <div class="kpis">
    <div class="kpi"><div class="label">Controls executed</div><div class="value">{summary.total_controls}</div></div>
    <div class="kpi"><div class="label">Passed</div><div class="value">{summary.passed}</div></div>
    <div class="kpi"><div class="label">Failed</div><div class="value">{summary.failed}</div></div>
    <div class="kpi"><div class="label">Exceptions</div><div class="value">{summary.total_exceptions:,}</div></div>
    <div class="kpi"><div class="label">Effectiveness</div><div class="value">{summary.effectiveness}%</div></div>
  </div>

  <table>
    <tr><th style="width:230px">Executed at</th><td>{_escape(summary.executed_at)}</td></tr>
    <tr><th>Data as-of date</th><td>{_escape(summary.as_of)}</td></tr>
    <tr><th>Data snapshot SHA-256</th><td><span class="hash">{_escape(summary.data_hash)}</span></td></tr>
    <tr><th>Run ID</th><td>{summary.run_id if summary.run_id else 'not persisted'}</td></tr>
  </table>

  {warnings_html}

  <h2>Control results</h2>
  <table>
    <thead><tr><th>Control</th><th>Name</th><th>Severity</th><th>Status</th>
    <th>Tested</th><th>Exceptions</th><th>Rate</th><th>Framework mapping</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>

  <h2>Control descriptions</h2>
  <table>
    <thead><tr><th>Control</th><th>Description</th><th>Thresholds</th><th>Outcome</th></tr></thead>
    <tbody>
      {''.join(
        f"<tr><td><b>{_escape(r.control_id)}</b><br><span style='color:var(--muted);font-size:12px'>"
        f"{_escape(r.risk_category)}</span></td><td>{_escape(r.description)}</td>"
        f"<td><code>{_escape(r.parameters_used)}</code></td><td>{_escape(r.summary_text)}</td></tr>"
        for r in summary.results)}
    </tbody>
  </table>

  <h2>Exception detail</h2>
  {''.join(detail_blocks)}

  <h2>Configuration snapshot</h2>
  <pre>{html.escape(_dump_config_yaml(self.config))}</pre>

  <footer>Automated evidence generated by the CCM tool. Retain with audit working papers.</footer>
</div>
</body>
</html>
"""


def _dump_config_yaml(config: ControlsConfig) -> str:
    """Serialise a config snapshot to YAML for the report appendix."""
    import yaml  # local import: keeps yaml out of the module's import surface

    return yaml.safe_dump(config.as_dict(), sort_keys=False, allow_unicode=True)


def generate_reports(
    summary: RunSummary,
    config: ControlsConfig | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Convenience wrapper: build a :class:`ReportWriter` and write everything.

    Args:
        summary: The run to report on.
        config: Optional config snapshot to embed.
        output_dir: Optional destination directory.

    Returns:
        Mapping of artefact name to written path.
    """
    return ReportWriter(summary, config=config, output_dir=output_dir).write_all()
