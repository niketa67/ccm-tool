"""Control implementations.

Importing this package registers every built-in control into the registry, so
the engine can discover them by ``control_id``.

Adding a control: create ``ccXX_name.py`` with a ``@register_control``-decorated
class and import it here.
"""

from __future__ import annotations

from . import cc01_terminated_active, cc02_sod_conflict, cc03_privileged_no_mfa  # noqa: F401
from . import cc04_dormant_accounts, cc05_stale_privileged_access  # noqa: F401
from . import cc06_shared_accounts, cc07_orphan_accounts, cc08_excessive_failed_logins  # noqa: F401
from .base import (
    EXCEPTION_COLUMNS,
    BaseControl,
    ControlResult,
    ControlStatus,
    empty_exceptions,
    normalise_exceptions,
)
from .registry import build_controls, get_control_class, iter_control_classes, registered_control_ids

__all__ = [
    "BaseControl",
    "ControlResult",
    "ControlStatus",
    "EXCEPTION_COLUMNS",
    "empty_exceptions",
    "normalise_exceptions",
    "register_control",
    "registered_control_ids",
    "build_controls",
    "get_control_class",
    "iter_control_classes",
]

#: Control modules auto-registered on import (documentation for readers).
CONTROL_MODULES = tuple(
    (
        cc01_terminated_active,
        cc02_sod_conflict,
        cc03_privileged_no_mfa,
        cc04_dormant_accounts,
        cc05_stale_privileged_access,
        cc06_shared_accounts,
        cc07_orphan_accounts,
        cc08_excessive_failed_logins,
    )
)
