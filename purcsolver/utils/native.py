"""
Lazy access to the compiled native core extension.

Keeps a single cached handle to ``purcsolver._purcsolver_core`` and raises a
clear, actionable error when the extension has not been built (rather than
failing obscurely at first use).  Mirrors LaplacianSolve's ``_loader`` and PUM's
``native_choice`` patterns.

Native is the shipping hot path; the pure-numpy implementations elsewhere in the
package exist as a dev fallback and as the parity oracle for native tests.  Every
hot call site selects between them via :func:`native_available`.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

from .logging import get_logger

_logger = get_logger(__name__)

_core: Optional[ModuleType] = None


class NativeUnavailableError(ImportError):
    """Raised when the compiled native core is required but not importable."""


def native_core() -> ModuleType:
    """
    Import and cache the native core module.

    Returns:
        The ``purcsolver._purcsolver_core`` extension module.

    Raises:
        NativeUnavailableError: If the compiled extension is not present.

    """
    global _core
    if _core is None:
        try:
            _core = importlib.import_module("purcsolver._purcsolver_core")
        except ImportError as exc:  # pragma: no cover - exercised only when unbuilt
            raise NativeUnavailableError(
                "The PURCSolver native core is not built. Install the package in "
                "editable mode with: pip install --no-build-isolation -e ."
            ) from exc
    return _core


def native_available() -> bool:
    """
    Return whether the native core extension can be imported.

    Returns:
        ``True`` if the extension is importable, else ``False``.

    """
    try:
        native_core()
        return True
    except ImportError:
        return False
