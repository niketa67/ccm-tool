# Continuous Controls Monitoring (CCM) Tool

**Config-driven automated testing of identity & access controls — with an audit-ready evidence trail, an exceptions workbench and a live dashboard.**

Built as a portfolio project for IT Risk / Risk Technology roles: it demonstrates
the full arc of controls monitoring work — *design a test, codify it, execute it
over real-shaped data, triage the findings, and package the evidence* — in a
single reproducible tool. No external APIs, no paid services, no cloud.

---

## Why this exists

In a first-line risk or audit function, a large share of ITGC testing is still
done by hand: export a user-access report, filter for leavers, eyeball role
combinations, paste into a spreadsheet, repeat every quarter. This tool
automates that loop:

| Manual process | CCM Tool |
| --- | --- |
| Filter leavers against the account list | `CC-01 Terminated Employee Active Accounts` |
| Check who can create *and* approve payments | `CC-02 Segregation of Duties Conflicts` |
| Ask the infra team about MFA on admin accounts | `CC-03 Privileged Accounts Without MFA` |
| Hunt for accounts idle > 90 days | `CC-04 Dormant Accounts` |
| Chase overdue privileged recertifications | `CC-05 Stale Privileged Access Reviews` |
| Find shared `admin`/`svc` logins | `CC-06 Shared / Generic Account Usage` |
| Reconcile accounts back to HR | `CC-07 Orphan Accounts` |
| Review SIEM logs for password spraying | `CC-08 Excessive Failed Logins` |

Each control is evaluated against four synthetic HR/IAM/SIEM extracts
(~500 employees, 511 accounts, 26.7k login records), producing **PASS/FAIL**
verdicts, a rate per control, row-level exceptions with human-readable reasons,
and a persisted run history in SQLite.

## Demo results (from a seeded, reproducible dataset)

The data generator injects a **known number of control breaches** into an
otherwise clean population. Those counts are the ground truth the engine must
recover exactly — the strongest available proof that the control logic is
correct, not just that it runs:

| Control | Risk category | Severity | Injected | Found by engine |
| --- | --- | --- | ---: | ---: |
| CC-01 Terminated Employee Active Accounts | Logical Access Management | Critical | 8 | **8** |
| CC-02 Segregation of Duties Conflicts | Segregation of Duties | High | 12 | **12** |
| CC-03 Privileged Accounts Without MFA | Privileged Access Management | Critical | 10 | **10** |
| CC-04 Dormant Accounts | Logical Access Management | Medium | 25 | **25** |
| CC-05 Stale Privileged Access Reviews | Access Certification | High | 7 | **7** |
| CC-06 Shared / Generic Account Usage | Identity & Non-Repudiation | Medium | 6 | **6** |
| CC-07 Orphan Accounts | Logical Access Management | High | 5 | **5** |
| CC-08 Excessive Failed Logins (Brute Force) | Security Monitoring | Medium | 4 | **4** |
| | | | **77** | **77 — 100% detection** |

Every remaining account is deliberately clean, so any exception above these
counts would indicate a false positive; any below, a missed finding.

## Quick start

```bash
# 1. Environment (Python 3.10+; developed on 3.12)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux

# 2. Generate the synthetic data extract (seeded, reproducible)
python run_controls.py --generate-data

# 3. Run every enabled control and store results in SQLite
python run_controls.py --run

# 4. Run again and emit the evidence pack
python run_controls.py --run --report

# 5. Run a subset of controls
python run_controls.py --run --controls CC-01 CC-03

# 6. Launch the dashboard
.venv\Scripts\streamlit run app\streamlit_app.py
```

Optional seeds: `python scripts/seed_history.py` writes four monitoring runs
(different as-of dates) so the dashboard's Trends view has history on first open.

## Architecture

```
                ┌──────────────────── config layer ────────────────────┐
                │  controls.yaml (thresholds, severity, on/off)        │
                │  sod_rules.yaml (toxic role pairs)                   │
                └──────────────────────────┬───────────────────────────┘
                                           │
data/raw/*.csv ──▶ loader ──▶ DataBundle ──▶ engine ──▶ RunSummary ──▶ storage ──▶ ccm.db
 (HR, accounts,     (validate +    (typed)    │              │             ▲            │
 roles, logs)        hash)                    │              │             └── SQLite:
                                              │              │                 runs /
                                              ▼              │                 control_results /
                                     controls/*.py           │                 exceptions
                                     8 × BaseControl         ▼
                                     (register via          reporting ──▶ reports/run_<ts>/
                                      decorator)                          evidence_report.md
                                                                          evidence_report.html
                                                                          exceptions_*.csv

app/streamlit_app.py ──▶ reads ccm.db + re-runs controls, edits exception status
tests/ ──▶ 80 pytest cases over every control, engine, storage and config
```

Design facts worth knowing before reading the code:

- **Policy lives in YAML, logic in Python.** Every threshold, severity and
  on/off switch is read from `config/controls.yaml`; the classes never
  hardcode numbers. Changing `dormant_days: 90` to `45` changes behaviour on
  the next run — no code edits, and the thresholds used are re-stored with each
  run for traceability.
- **One contract for all controls.** Each control subclasses `BaseControl` and
  implements just two methods: `population(data)` (what was tested) and
  `evaluate(data)` (what breaches it). The base class handles timing, exception
  rate math, PASS/FAIL derivation and error containment.
- **A broken control surfaces as `ERROR`, not silence.** A control that raises
  is contained at its boundary and reported with status `ERROR` and its stack
  reason, so a run never aborts halfway and never fakes a PASS.
- **Every result is reproducible.** The loader computes a SHA-256 over the four
  source CSVs; that hash is stamped on the run, every report and the dashboard,
  so a reviewer can always answer *"what data produced these numbers?"*.

## The control catalogue

| ID | Control | Tests | Threshold / parameters (from YAML) |
| --- | --- | --- | --- |
| CC-01 | Terminated Employee Active Accounts | Leavers whose account is still `Active`, or authenticated after `termination_date` (0-day grace, configurable) | `terminated_statuses`, `flag_post_termination_login` |
| CC-02 | Segregation of Duties Conflicts | Accounts holding both sides of a toxic pair; honours per-rule system scope and active-only | 8 rule pairs in `sod_rules.yaml`, e.g. *Payment Creator + Payment Approver* |
| CC-03 | Privileged Accounts Without MFA | Active `Privileged` accounts with `mfa_enabled` false **or missing** | `privileged_account_types` |
| CC-04 | Dormant Accounts | Active accounts idle > 90 days; never-used accounts aged from `created_date`; `Service` accounts excluded | `dormant_days: 90` |
| CC-05 | Stale Privileged Access Reviews | Privileged accounts not recertified within 180 days, or never reviewed | `review_interval_days: 180` |
| CC-06 | Shared / Generic Account Usage | Active `Shared`-type accounts and generic usernames (`(?i)^admin$`, `^test`, `^svc_generic`, …) | type list + regex patterns |
| CC-07 | Orphan Accounts | Active accounts with no `employee_id`, an id missing from HR, or an identity that cannot be attributed; `Service` accounts + config allowlist excluded | `exclude_account_types`, `allowlist_usernames` |
| CC-08 | Excessive Failed Logins | > 5 failed attempts within a rolling 1-hour window, over a 30-day lookback; optional strict-consecutive mode | `max_failed_attempts: 5`, `window_hours: 1` |

Framework mappings (ISO 27001, SOX ITGC, NIST SP 800-53) are declared per
control in config and printed on every report, because the first question an
auditor asks a monitoring tool is *which control objective does this satisfy?*

## Dashboard

Five views, navigated from the sidebar (plots are Plotly; red/amber/green scheme
by severity and status):

1. **Overview** — KPI cards (controls, passed, failed, exceptions, effectiveness %), exceptions-by-control bar coloured by severity, pass/fail donut, per-control table.
2. **Control Results** — outcome table plus a per-control drill-down: summary text, thresholds actually applied in that run, framework mapping, full exception detail.
3. **Exceptions Workbench** — filter by control/system/severity/status, then change an exception's workflow status (`Open → Acknowledged → Remediated`) inline; changes are persisted to SQLite immediately, without rewriting history.
4. **Trends** — exception count per control across stored runs, a latest-vs-previous movement table, and the run history with data hashes (so a trend spike can always be traced to its input snapshot).
5. **Run and Export** — trigger a fresh run from the browser, monitor payloads, download the HTML report and full exception CSV, view the live config, and read the recipe for adding control #9.

## The evidence pack

`--report` writes `reports/run_<timestamp>_<run_id>/`:

- `evidence_report.md` — human-readable working paper (run metadata, results, per-control thresholds, full exception tables, the exact config snapshot).
- `evidence_report.html` — self-contained styled version of the same (no external assets, prints cleanly).
- `exceptions_all_*.csv` — every exception row, one line per finding.
- `exceptions_CC-0X_*.csv` — one file per failing control, the format remediation owners actually ask for.

Each report header carries the run id, execution timestamp, data as-of date and
the SHA-256 of the source extract, so it can stand alone in audit working
papers.

## Testing

**80 tests, all passing.** Coverage per the testing philosophy *"a control is
only as good as its demonstrated edge-case behaviour"*:

- **Controls** (`tests/test_controls.py`): every control is tested three ways —
  a clear violation is flagged, a compliant record is not flagged, and boundary
  behaviour matches the documented threshold (e.g. CC-04 flags 91 idle days but
  not exactly 90; CC-08 flags 6 failures in an hour but not exactly 5; CC-02
  honours per-rule system scope and skips disabled accounts).
- **Engine** (`tests/test_engine.py`): execution order, `only` subsetting, config
  severity overrides, summary arithmetic, and — critically — that an exploding
  control yields `ERROR` while the rest of the run completes.
- **Storage** (`tests/test_storage.py`): save/load round-trip, exception status
  updates persisting across a connection reopen, invalid statuses rejected,
  filters, trends and `NULL`-safe value coercion.
- **Config** (`tests/test_storage.py::TestConfigLoading`): catalogue completeness,
  threshold wiring, and failure cases (missing file, bad severity, bad schema)
  raising `ConfigError` with helpful messages.

```bash
python -m pytest tests -q          # 80 passed
```

## Engineering notes

Decisions made here that would come up in a technical interview:

- **pandas 3.x semantics.** String columns default to `str`/`string` dtypes, so
  the codebase normalises through explicit helpers instead of `astype(bool)`
  (which coerces missing values to `True`, a classic silent-data-corruption
  trap). Missing-value handling is centralised in `loader.to_bool_series` and
  `storage.cell_for_db`.
- **Thread-safe storage for Streamlit.** Streamlit re-runs the script on a new
  thread per interaction, which invalidates a plain cached `sqlite3.Connection`
  (`SQLite objects created in a thread can only be used in that same thread`).
  `Storage` therefore keeps one connection per thread via `threading.local`,
  making it safe to cache at the application level.
- **Set semantics over row-by-row matching.** CC-02 and CC-08 build their
  lookups once (`groupby` / rolling windows) rather than nesting loops over
  accounts × rules, keeping the full run to a couple of seconds.
- **Deterministic generation.** Seeded Faker + NumPy (`seed: 42`) means the
  README's numbers above are stale-proof: anyone re-running `--generate-data`
  recovers the same 77 exceptions.
- **Reporting without a template engine.** The HTML report is composed directly
  (escaped via `html.escape`), so the dependency list stays minimal and no
  untrusted data can break out of the table cells.

## Extending: add a control in three steps

1. **Implement** `src/ccm/controls/cc09_my_control.py`:

   ```python
   from .base import BaseControl
   from .registry import register_control

   @register_control
   class MyNinthControl(BaseControl):
       control_id = "CC-09"
       name = "..."
       description = "What business risk this tests."
       risk_category = "..."
       severity = "Medium"
       framework_mapping = ("ISO 27001 A.x.y (...)",)
       default_parameters = {"threshold": 7}

       def population(self, data):          # what was tested
           return data.user_accounts

       def evaluate(self, data):            # return violation rows incl. `reason`
           ...
   ```

2. **Register** it by importing the module in `src/ccm/controls/__init__.py`.
3. **Declare policy** — add a `CC-09:` block to `config/controls.yaml` with its
   threshold, severity and framework mapping.

The engine, dashboard and report writer pick it up automatically — no other
code changes. `python run_controls.py --run --controls CC-09` runs it in
isolation.

## Project layout

```
ccm-tool/
├── run_controls.py            # CLI entry point (generate / run / report)
├── config/
│   ├── controls.yaml          # control catalogue: thresholds, severity, mappings
│   └── sod_rules.yaml         # toxic role pairs consumed by CC-02
├── src/ccm/
│   ├── paths.py               # centralised pathlib locations
│   ├── config.py              # YAML loading + validation (ConfigError on bad input)
│   ├── loader.py              # CSVs → DataBundle + SHA-256 + validation
│   ├── engine.py              # run orchestration + RunSummary
│   ├── storage.py             # SQLite persistence + thread-safe connections
│   ├── reporting.py           # Markdown / HTML / CSV evidence writer
│   ├── data_generator.py      # seeded synthetic data + exception injection
│   └── controls/              # 8 controls + base class + registry
├── app/streamlit_app.py       # 5-view dashboard
├── scripts/seed_history.py    # optional: seed demo trend history
├── tests/                     # 80 pytest cases
├── data/raw/                  # generated extracts (gitignored)
├── reports/                   # evidence packs (gitignored)
└── ccm.db                     # monitoring history (gitignored)
```

## Assumptions & limitations

By design, so the scope stays honest for an MVP:

- **Single-file extracts, one snapshot per run.** The loaders assume one CSV per
  source with a stable column contract. Multi-entity or incremental pipelines
  (e.g. delta HR/CMDB feeds) are a roadmap item.
- **`as_of` drives all age math.** Age thresholds compare against the data's
  as-of date (default: today), which keeps runs reproducible — but the source
  extracts' dates must be internally consistent with it.
- **Simulation, not live integration.** The demo data is synthetic. Attaching
  the controls to a real directory (Okta, AD, Workday export) means replacing
  the CSV loader with an API client — the control layer would not change,
  because it is written against the typed `DataBundle`, not the file format.
- **One shared SQLite file.** Fine for a single analyst or a small team; a
  multi-user deployment would move to Postgres and add authentication — the
  `Storage` interface would stay the same.
- **Exception statuses are workflow-only.** `Acknowledged` / `Remediated` record
  triage state; there is no approval chain, SLA clock or ticket-system push.
- **The seeded repo state.** `data/raw/`, `reports/` and `ccm.db` are gitignored
  outputs; a fresh clone rebuilds all three with the two commands in Quick
  Start.

## Roadmap

- Trend charts that show **remediation closing findings** by comparing runs
  across remediation cycles (the schema already supports it).
- A Jira/ServiceNow exporter so workbench items become tickets.
- Control-effectiveness KPI roll-up over time (percent effectiveness per period).
- A first-class SoD rule-builder UI instead of hand-edited YAML.
- Dockerised one-command deployment with authentication.
- Optional streaming mode: evaluate new login logs on a schedule (cc-08 becomes
  a real-time security alert rather than a periodic review).

---

*Built with Python, pandas, NumPy, PyYAML, SQLite3, Streamlit, Plotly, Faker
and pytest. Author: an IT-risk analyst who got tired of filtering spreadsheets
by hand.*
