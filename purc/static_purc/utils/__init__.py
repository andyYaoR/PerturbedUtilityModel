"""Internal utilities: logging, type aliases, and native-core access."""

from __future__ import annotations

from .logging import configure_logging, get_logger, logger
from .native import NativeUnavailableError, native_available, native_core

__all__ = [
    "logger",
    "get_logger",
    "configure_logging",
    "native_available",
    "native_core",
    "NativeUnavailableError",
]
