#!/usr/bin/env python3
"""
Pre-commit hook: enforce directory boundary rules.

Mirrors the equivalent hook in the PUM / LaplacianSolve packages.  It keeps
native source under ``src/purc/static_purc/`` restricted to C/C++/CUDA files and
prevents committing CMake / compiled build artifacts anywhere in the tree.
Exit code 0 = pass, 1 = violations found.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
violations = []

# File suffixes that may live in the native source tree.
NATIVE_SUFFIXES = (".cpp", ".hpp", ".h", ".cu", ".cuh")
# Names that are legitimate config (not build artifacts) inside the native tree.
NATIVE_ALLOWED_NAMES = ("CMakeLists.txt",)
# Compiled / generated artifacts that must never be committed anywhere.
ARTIFACT_SUFFIXES = (".so", ".dylib", ".pyd", ".o", ".obj", ".a")
ARTIFACT_NAMES = (
    "CMakeCache.txt",
    "cmake_install.cmake",
    "build.ninja",
    ".ninja_deps",
    ".ninja_log",
    ".skbuild-info.json",
    "libnanobind-static.a",
)


def check(condition: bool, msg: str) -> None:
    """
    Record a violation message when *condition* is falsey.

    Args:
        condition: Predicate that must hold for the rule to pass.
        msg: Human-readable description recorded when the rule fails.

    """
    if not condition:
        violations.append(msg)


# 1. src/purc/static_purc/ should only contain native source + CMakeLists.
native_src = ROOT / "src" / "purc" / "static_purc"
if native_src.exists():
    for f in native_src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(ROOT)
            ok = f.suffix in NATIVE_SUFFIXES or f.name in NATIVE_ALLOWED_NAMES
            check(ok, f"{rel}: only {NATIVE_SUFFIXES} or CMakeLists.txt allowed")

# 2. No compiled or CMake build artifacts anywhere in the tree (skip build/).
for f in ROOT.rglob("*"):
    if not f.is_file():
        continue
    parts = f.relative_to(ROOT).parts
    if "build" in parts or ".git" in parts:
        continue
    if f.suffix in ARTIFACT_SUFFIXES or f.name in ARTIFACT_NAMES:
        check(False, f"{f.relative_to(ROOT)}: build artifact should not be committed")

if violations:
    print("Directory boundary violations found:")
    for v in violations:
        print(f"  - {v}")
    sys.exit(1)
else:
    print("All directory boundaries OK.")
    sys.exit(0)
