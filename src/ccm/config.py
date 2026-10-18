"""Configuration loading and validation for the CCM tool.

Two YAML documents drive the entire framework:

* ``config/controls.yaml``  - which controls run, their severity, thresholds and
  framework mappings.
* ``config/sod_rules.yaml`` - the conflicting role pairs used by control CC-02.

Nothing about *policy* is hard-coded in the Python control classes: the classes
implement the logic, the YAML implements the policy. That split is what lets a
risk analyst retune a threshold (or switch a control off) without a code review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .paths import CONTROLS_CONFIG_PATH, SOD_RULES_PATH

logger = logging.getLogger(__name__)

__all__ = [
    "SEVERITY_ORDER",
    "SEVERITY_RANK",
    "ConfigError",
    "ControlConfig",
    "ControlsConfig",
    "SodRule",
    "load_controls_config",
    "load_sod_rules",
    "severity_rank",
]

#: Ordered most-severe first; used for sorting in reports and dashboards.
SEVERITY_ORDER: tuple[str, ...] = ("Critical", "High", "Medium", "Low")

#: Numeric rank per severity (higher == more severe) for sorting/comparison.
SEVERITY_RANK: dict[str, int] = {sev: len(SEVERITY_ORDER) - i for i, sev in enumerate(SEVERITY_ORDER)}

_REQUIRED_CONTROL_KEYS = ("enabled",)


class ConfigError(ValueError):
    """Raised when a configuration document is missing or malformed."""


@dataclass(frozen=True)
class ControlConfig:
    """Resolved policy for a single control.

    Attributes:
        control_id: Stable identifier such as ``"CC-01"``.
        enabled: Whether the engine should execute this control.
        severity: Severity override from config (``Low`` to ``Critical``).
        parameters: Control-specific thresholds passed to the control class.
        name: Optional display-name override.
        description: Optional description override.
        risk_category: Optional risk-category override.
        framework_mapping: Optional framework-mapping override.
    """

    control_id: str
    enabled: bool = True
    severity: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    description: str | None = None
    risk_category: str | None = None
    framework_mapping: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ControlsConfig:
    """The whole ``controls.yaml`` document, resolved into typed objects."""

    controls: Mapping[str, ControlConfig]
    meta: Mapping[str, Any] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def get(self, control_id: str) -> ControlConfig | None:
        """Return the config for ``control_id``, or ``None`` if not declared."""
        return self.controls.get(control_id)

    def enabled_controls(self) -> list[str]:
        """Control ids that are switched on, in config-declaration order."""
        return [cid for cid, cfg in self.controls.items() if cfg.enabled]

    def disabled_controls(self) -> list[str]:
        """Control ids explicitly switched off in config."""
        return [cid for cid, cfg in self.controls.items() if not cfg.enabled]

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict view, suitable for rendering in the dashboard."""
        return {
            "meta": dict(self.meta),
            "defaults": dict(self.defaults),
            "controls": {
                cid: {
                    "enabled": cfg.enabled,
                    "severity": cfg.severity,
                    "parameters": dict(cfg.parameters),
                }
                for cid, cfg in self.controls.items()
            },
        }


@dataclass(frozen=True)
class SodRule:
    """One mutually-incompatible role pair from ``sod_rules.yaml``."""

    rule_id: str
    role_a: str
    role_b: str
    system: str = "*"
    risk: str = ""
    severity: str = "High"
    framework: str = ""

    def matches_system(self, system: str) -> bool:
        """Return ``True`` if the rule applies to ``system`` (``*`` == any)."""
        return self.system in ("*", "", None) or self.system == system


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path``, raising :class:`ConfigError` on failure."""
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - malformed-YAML guard
        raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc

    if document is None:
        raise ConfigError(f"Configuration file is empty: {path}")
    if not isinstance(document, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(document).__name__}")
    return document


def _normalise_severity(severity: Any, *, context: str) -> str | None:
    """Validate and canonicalise a severity string."""
    if severity is None:
        return None
    text = str(severity).strip().capitalize()
    if text not in SEVERITY_RANK:
        raise ConfigError(
            f"{context}: invalid severity {severity!r}. Expected one of {', '.join(SEVERITY_ORDER)}."
        )
    return text


def load_controls_config(path: Path | str | None = None) -> ControlsConfig:
    """Load and validate ``controls.yaml``.

    Args:
        path: Optional override for the config location. Defaults to
            ``config/controls.yaml`` next to the project root.

    Returns:
        A fully resolved :class:`ControlsConfig`.

    Raises:
        ConfigError: If the file is missing, malformed, or a control block is
            invalid (unknown severity, non-mapping ``parameters``, ...).
    """
    config_path = Path(path) if path is not None else CONTROLS_CONFIG_PATH
    document = _read_yaml(config_path)

    raw_controls = document.get("controls")
    if not isinstance(raw_controls, dict) or not raw_controls:
        raise ConfigError(f"{config_path}: 'controls' must be a non-empty mapping")

    resolved: dict[str, ControlConfig] = {}
    for control_id, block in raw_controls.items():
        control_id = str(control_id).strip()
        if not isinstance(block, dict):
            raise ConfigError(f"{config_path}: control {control_id} must be a mapping")

        missing = [key for key in _REQUIRED_CONTROL_KEYS if key not in block]
        if missing:
            raise ConfigError(f"{config_path}: control {control_id} is missing keys: {missing}")

        parameters = block.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ConfigError(f"{config_path}: control {control_id} 'parameters' must be a mapping")

        framework = block.get("framework_mapping")
        if framework is not None and isinstance(framework, str):
            framework = (framework,)
        elif isinstance(framework, Sequence):
            framework = tuple(str(item) for item in framework)

        resolved[control_id] = ControlConfig(
            control_id=control_id,
            enabled=bool(block.get("enabled", True)),
            severity=_normalise_severity(block.get("severity"), context=f"{config_path}: {control_id}"),
            parameters=dict(parameters),
            name=block.get("name"),
            description=(block.get("description") or "").strip() or None,
            risk_category=block.get("risk_category"),
            framework_mapping=framework,
        )

    logger.debug("Loaded %d control definitions from %s", len(resolved), config_path)
    return ControlsConfig(
        controls=resolved,
        meta=document.get("meta") or {},
        defaults=document.get("defaults") or {},
        source_path=config_path,
    )


def load_sod_rules(path: Path | str | None = None) -> list[SodRule]:
    """Load the segregation-of-duties rule catalogue.

    Args:
        path: Optional override; defaults to ``config/sod_rules.yaml``.

    Returns:
        A list of :class:`SodRule` objects. An empty rule set is valid and simply
        means CC-02 has nothing to test against.
    """
    rules_path = Path(path) if path is not None else SOD_RULES_PATH
    document = _read_yaml(rules_path)

    raw_rules = document.get("rules") or []
    if not isinstance(raw_rules, list):
        raise ConfigError(f"{rules_path}: 'rules' must be a list")

    rules: list[SodRule] = []
    for index, block in enumerate(raw_rules):
        if not isinstance(block, dict):
            raise ConfigError(f"{rules_path}: rule #{index} must be a mapping")
        role_a, role_b = block.get("role_a"), block.get("role_b")
        if not role_a or not role_b:
            raise ConfigError(f"{rules_path}: rule #{index} requires both 'role_a' and 'role_b'")

        rules.append(
            SodRule(
                rule_id=str(block.get("id") or f"SOD-{index + 1:03d}"),
                role_a=str(role_a),
                role_b=str(role_b),
                system=str(block.get("system") or "*"),
                risk=str(block.get("risk") or "").strip(),
                severity=_normalise_severity(block.get("severity"), context=f"{rules_path}: rule #{index}")
                or "High",
                framework=str(block.get("framework") or "").strip(),
            )
        )

    logger.debug("Loaded %d SoD rules from %s", len(rules), rules_path)
    return rules


def severity_rank(severity: str | None) -> int:
    """Return a sortable rank for ``severity`` (unknown values rank lowest)."""
    if severity is None:
        return 0
    return SEVERITY_RANK.get(str(severity).strip().capitalize(), 0)
