"""
Lazy access to the compiled native core extension.

Keeps a single cached handle to ``purc.laplaciansolve._laplaciansolve_core`` and
raises a clear, actionable error when the extension has not been built (rather
than failing obscurely at first use).
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
import re
import sys
from types import ModuleType
from typing import List, Optional

from .logging import get_logger

_logger = get_logger(__name__)

_core: Optional[ModuleType] = None


def native_core() -> ModuleType:
    """
    Import and cache the native core module.

    Returns:
        The ``purc.laplaciansolve._laplaciansolve_core`` extension module.

    Raises:
        ImportError: If the compiled extension is not present.

    """
    global _core
    if _core is None:
        try:
            _core = importlib.import_module("purc.laplaciansolve._laplaciansolve_core")
        except ImportError as exc:  # pragma: no cover - exercised only when unbuilt
            raise ImportError(
                "The LaplacianSolve native core is not built. Install the package "
                "in editable mode with: pip install --no-build-isolation -e ."
            ) from exc
    return _core


def has_native_core() -> bool:
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


_cholmod: Optional[ModuleType] = None
_torch_openmp_preloaded = False

# Shared-library basenames that *are* an OpenMP runtime (LLVM libomp, GNU libgomp,
# Intel libiomp5).  Used both to find torch's bundled runtime and to detect when
# more than one distinct OpenMP runtime has been mapped into the process.
_OPENMP_LIB_RE = re.compile(r"\b(?:libomp|libgomp|libiomp5?(?:md)?)\b", re.IGNORECASE)


def _torch_lib_dir() -> Optional[str]:
    """
    Locate torch's bundled-library directory without importing torch.

    Returns:
        The path to ``<torch>/lib`` if torch is installed, else ``None``.

    """
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.join(list(spec.submodule_search_locations)[0], "lib")


def _preload_torch_openmp() -> None:
    """
    Preload torch's bundled OpenMP runtime so CHOLMOD reuses the *same* runtime.

    PyTorch's wheel bundles an OpenMP runtime whose install-name matches the path
    the system ``libcholmod`` links, so loading it first (``RTLD_GLOBAL``) makes a
    subsequently-loaded ``libcholmod`` reuse it instead of mapping a second copy -
    one runtime, no ``OMP Error #15``, no SuiteSparse rebuild.  This is the single
    process-wide OpenMP runtime the whole package shares (see docs/PURC_INTEGRATION).

    A no-op if torch is not installed (CHOLMOD then loads the only OpenMP runtime
    in the process, which is equally safe).  Best-effort: never fatal.
    """
    global _torch_openmp_preloaded
    if _torch_openmp_preloaded:
        return
    _torch_openmp_preloaded = True  # set first: a failed attempt should not retry
    lib_dir = _torch_lib_dir()
    if not lib_dir or not os.path.isdir(lib_dir):
        return
    # torch's bundled OpenMP runtime filename varies by platform/wheel.
    candidates = (
        "libomp.dylib",
        "libiomp5.dylib",
        "libgomp.dylib",  # macOS
        "libgomp.so.1",
        "libomp.so",
        "libomp.so.5",
        "libiomp5.so",  # Linux
        "libiomp5md.dll",
        "libomp.dll",  # Windows
    )
    for name in candidates:
        path = os.path.join(lib_dir, name)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                _logger.debug("Preloaded torch OpenMP runtime: %s", path)
            except OSError as exc:  # pragma: no cover - platform-dependent
                _logger.debug("Could not preload torch OpenMP runtime %s: %s", path, exc)
            return


def _loaded_openmp_runtimes() -> List[str]:
    """
    Return the realpaths of OpenMP runtime libraries currently mapped.

    Returns:
        Sorted distinct realpaths of mapped OpenMP runtimes (empty if the platform
        is not introspectable - in which case the single-runtime guard is skipped).

    """
    paths = set()
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL(None)
            libc._dyld_image_count.restype = ctypes.c_uint32
            libc._dyld_get_image_name.restype = ctypes.c_char_p
            libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
            for i in range(libc._dyld_image_count()):
                name = libc._dyld_get_image_name(i)
                if name and _OPENMP_LIB_RE.search(os.path.basename(name.decode())):
                    paths.add(os.path.realpath(name.decode()))
        except Exception:  # pragma: no cover - diagnostic best-effort
            return []
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/self/maps") as fh:
                for line in fh:
                    region = line.rstrip().split(" ", 5)
                    path = region[-1].strip() if len(region) == 6 else ""
                    if path and _OPENMP_LIB_RE.search(os.path.basename(path)):
                        paths.add(os.path.realpath(path))
        except OSError:  # pragma: no cover - diagnostic best-effort
            return []
    return sorted(paths)


def _verify_single_openmp_runtime() -> None:
    """
    Fail loudly if more than one distinct OpenMP runtime is mapped.

    The preload above unifies torch + CHOLMOD on one runtime; a second runtime
    here means another library linked its own (e.g. a different libomp/libgomp,
    or MKL's libiomp5).  Rather than risk the documented "degraded performance or
    incorrect results", raise a clear, actionable error (no silent unsafe
    coexistence).  Set ``LAPLACIANSOLVE_ALLOW_MULTIPLE_OPENMP=1`` to downgrade to a
    warning and proceed at your own risk.

    Raises:
        ImportError: If multiple OpenMP runtimes are mapped and the override env
            var is not set.

    """
    runtimes = _loaded_openmp_runtimes()
    if len(runtimes) <= 1:
        return
    listing = "\n  ".join(runtimes)
    msg = (
        f"Multiple OpenMP runtimes are loaded in this process:\n  {listing}\n"
        "This can degrade performance or silently produce incorrect results. "
        "LaplacianSolve preloads torch's OpenMP runtime so CHOLMOD reuses it; a "
        "second runtime means another library linked a different OpenMP. Ensure "
        "torch (or its OpenMP) is loaded before any library that links its own."
    )
    if os.environ.get("LAPLACIANSOLVE_ALLOW_MULTIPLE_OPENMP"):
        _logger.warning(msg)
    else:
        raise ImportError(
            msg + "\nSet LAPLACIANSOLVE_ALLOW_MULTIPLE_OPENMP=1 to proceed anyway."
        )


def cholmod_core() -> ModuleType:
    """
    Import and cache the CHOLMOD direct-solver extension.

    Before importing, torch's OpenMP runtime is preloaded so the system
    ``libcholmod`` reuses it (a single process-wide OpenMP runtime), then a guard
    verifies exactly one OpenMP runtime is mapped.

    Returns:
        The ``purc.laplaciansolve._laplaciansolve_cholmod`` extension module.

    Raises:
        ImportError: If CHOLMOD support was not built (SuiteSparse absent), or if
            multiple OpenMP runtimes are detected (see
            :func:`_verify_single_openmp_runtime`).

    """
    global _cholmod
    if _cholmod is None:
        # 1) Unify on torch's OpenMP runtime: load it first so libcholmod reuses it.
        _preload_torch_openmp()
        # 2) Safety net: should a second OpenMP runtime still map (e.g. an exotic
        #    platform where the install-names do not match), this keeps the import
        #    from hard-aborting with "OMP Error #15" so the guard below can raise a
        #    clean, actionable Python error instead of a SIGABRT.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        try:
            _cholmod = importlib.import_module("purc.laplaciansolve._laplaciansolve_cholmod")
        except ImportError as exc:  # pragma: no cover - exercised only when absent
            raise ImportError(
                "The CHOLMOD direct solver is not built (SuiteSparse not found at "
                "build time). Install SuiteSparse and reinstall, or use method='approxchol'."
            ) from exc
        _verify_single_openmp_runtime()
    return _cholmod


def has_cholmod() -> bool:
    """
    Return whether the CHOLMOD direct-solver extension is available.

    Returns:
        ``True`` if the CHOLMOD extension is importable, else ``False``.

    """
    try:
        cholmod_core()
        return True
    except ImportError:
        return False
