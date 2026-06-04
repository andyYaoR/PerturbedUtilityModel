"""
Reference oracles for correctness and performance cross-checks.

``cvxpy_oracle.py`` (M1) builds the equivalent DCP convex program for
perturbations that CVXPY can express (quadratic, entropy, Tsallis).
``scipy_oracle.py`` (M1) is an independent separable solver (trust-constr plus a
dense dual active-set Newton) used as ground truth for the polynomial sieve and
as a cross-check elsewhere; it deliberately does *not* call LaplacianSolve so it
cannot co-validate solver bugs.  M0 ships this package as a placeholder.
"""

from __future__ import annotations

__all__: list[str] = []
