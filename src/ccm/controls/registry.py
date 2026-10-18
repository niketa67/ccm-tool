"""Registry that auto-discovers controls by ``control_id``.

Control classes register themselves with the :func:`register_control` decorator
at import time. ``controls/__init__.py`` imports every control module, so
importing the package is enough to populate the registry.

Adding a new control therefore takes two steps and no central bookkeeping:

1. Create ``controls/ccXX_my_check.py`` with a class decorated by
   ``@register_control``.
2. Add a ``CC-XX`` block to ``config/controls.yaml``.
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence, Type

from ..config import ControlsConfig
from .base import BaseControl

logger = logging.getLogger(__name__)

__all__ = [
    "register_control",
    "get_control_class",
    "iter_control_classes",
    "registered_control_ids",
    "build_controls",
]

_REGISTRY: dict[str, Type[BaseControl]] = {}


def register_control(cls: Type[BaseControl]) -> Type[BaseControl]:
    """Class decorator that adds ``cls`` to the global control registry.

    Args:
        cls: A concrete :class:`BaseControl` subclass.

    Returns:
        The class unchanged, so it can be used as a decorator.

    Raises:
        ValueError: If the class has no ``control_id``, or if that id is
            already taken by a different class.
    """
    control_id = getattr(cls, "control_id", "") or ""
    if not control_id:
        raise ValueError(f"{cls.__name__} must define a non-empty 'control_id' to be registered")

    existing = _REGISTRY.get(control_id)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Duplicate control_id '{control_id}': already registered to {existing.__name__}, "
            f"cannot re-register {cls.__name__}"
        )

    _REGISTRY[control_id] = cls
    logger.debug("Registered control %s (%s)", control_id, cls.__name__)
    return cls


def get_control_class(control_id: str) -> Type[BaseControl]:
    """Return the registered class for ``control_id``.

    Raises:
        KeyError: If no such control is registered.
    """
    try:
        return _REGISTRY[control_id]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown control '{control_id}'. Registered controls: {known}") from exc


def iter_control_classes() -> Iterator[Type[BaseControl]]:
    """Yield registered control classes in ``control_id`` order."""
    for control_id in sorted(_REGISTRY):
        yield _REGISTRY[control_id]


def registered_control_ids() -> list[str]:
    """Return every registered control id, sorted."""
    return sorted(_REGISTRY)


def build_controls(
    config: ControlsConfig,
    *,
    only: Sequence[str] | None = None,
    include_disabled: bool = False,
) -> list[BaseControl]:
    """Instantiate controls according to configuration.

    Args:
        config: Loaded ``controls.yaml``.
        only: Optional whitelist of control ids. Ids in this list that are not
            registered raise :class:`KeyError`; ids that are registered but
            missing from config fall back to class defaults.
        include_disabled: When ``True``, instantiate controls even if config
            disables them (used by the CLI's ``--include-disabled`` flag).

    Returns:
        A list of ready-to-run control instances, ordered by ``control_id``.

    Raises:
        KeyError: If ``only`` names an unregistered control.
    """
    if only:
        unknown = sorted(set(only) - set(_REGISTRY))
        if unknown:
            raise KeyError(f"Unknown control id(s): {', '.join(unknown)}. Registered: {', '.join(registered_control_ids())}")

    controls: list[BaseControl] = []
    for control_id, cls in ((cid, _REGISTRY[cid]) for cid in sorted(_REGISTRY)):
        if only and control_id not in only:
            continue

        control_config = config.get(control_id)
        if control_config is None:
            if only:
                logger.warning("%s is not declared in config; running with class defaults", control_id)
                controls.append(cls())
            else:
                logger.debug("%s is registered but absent from config; skipped", control_id)
            continue

        if not control_config.enabled and not (include_disabled or only):
            logger.debug("%s is disabled in config; skipped", control_id)
            continue

        controls.append(
            cls(
                severity=control_config.severity,
                parameters=control_config.parameters,
                name=control_config.name,
                description=control_config.description,
                risk_category=control_config.risk_category,
                framework_mapping=control_config.framework_mapping,
            )
        )

    return controls
