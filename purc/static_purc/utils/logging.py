"""
Logging configuration for the PURCSolver package.

Provides the same ``get_logger`` / ``logger`` / ``configure_logging`` surface as
the PUM and LaplacianSolve packages so the codebases share a consistent style.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

_DEFAULT_FORMAT = "%(asctime)s %(name)s %(levelname)-5s %(message)s"


def _configure_base_logger() -> logging.Logger:
    """
    Attach a single stream handler to the root logger (idempotently).

    Returns:
        The base ``"purc"`` logger.

    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root_logger.addHandler(handler)
    return logging.getLogger("purc")


logger = _configure_base_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a module-aware logger with shared top-level configuration.

    Args:
        name: Module name (e.g., ``__name__``).  If *None*, returns the base
            ``"purcsolver"`` logger.

    Returns:
        A :class:`logging.Logger` instance.

    """
    if not name:
        return logger
    return logging.getLogger(name)


def configure_logging(
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    *,
    package_levels: Optional[Dict[str, int]] = None,
) -> None:
    """
    Configure the root logger format and per-package levels.

    Args:
        level: Root logger level (default ``INFO``).
        fmt: Format string.  ``None`` uses the project default.
        package_levels: Per-package level overrides, e.g.
            ``{"purcsolver.solvers": logging.DEBUG}``.

    """
    fmt = fmt or _DEFAULT_FORMAT
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers for idempotent reconfiguration.
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(stream_handler)

    if package_levels:
        for pkg, pkg_level in package_levels.items():
            logging.getLogger(pkg).setLevel(pkg_level)
