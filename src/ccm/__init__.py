"""Continuous Controls Monitoring (CCM) tool.

An automated, config-driven framework that tests IT general controls over user
access data (HR records, user accounts, role assignments and login logs) and
produces auditable PASS/FAIL evidence for every control.

Typical programmatic use::

    from ccm import engine, loader, storage

    bundle = loader.load_data_bundle()
    summary = engine.run_controls(bundle)
    storage.save_run(summary)
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
