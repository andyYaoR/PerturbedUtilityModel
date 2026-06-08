"""
Pure NumPy/SciPy reference implementation (the correctness oracle).

This subpackage is a faithful, readable port of the ``approxchol_lap`` solver of
``Laplacians.jl`` plus the PCG driver, the SDDM<->Laplacian embedding, and the
exact forest factorization.  Later stages reimplement the same algorithms in
C++/CUDA and validate the compiled backends against this reference.
"""

from __future__ import annotations

from .approx_chol import ApproxCholPQ, LLmatp, LLp, approx_chol, build_llmatp
from .components import components, vec_to_comps
from .forest_ldl import forest_ldl, forest_solve, is_forest
from .graph import flip_index, force_lap, lap, to_csc, validate_adjacency
from .ldl_solve import LDLinv, ldl_solve
from .params import ApproxCholParams
from .pcg import PCGResult, cg, pcg
from .rng import ArrayStream, GeneratorStream, SampleStream, as_stream
from .sddm import adj_val_and_excess, extend_matrix, sddm_wrap_lap
from .solver import (
    approxchol_lap,
    approxchol_lap_pcg,
    approxchol_sddm,
)

__all__ = [
    # Parameters / RNG
    "ApproxCholParams",
    "SampleStream",
    "GeneratorStream",
    "ArrayStream",
    "as_stream",
    # Graph utilities
    "lap",
    "force_lap",
    "flip_index",
    "to_csc",
    "validate_adjacency",
    "components",
    "vec_to_comps",
    # Factorization + solve
    "LLp",
    "LLmatp",
    "ApproxCholPQ",
    "build_llmatp",
    "approx_chol",
    "LDLinv",
    "ldl_solve",
    "cg",
    "pcg",
    "PCGResult",
    # Forest fast path
    "is_forest",
    "forest_ldl",
    "forest_solve",
    # SDDM embedding
    "adj_val_and_excess",
    "extend_matrix",
    "sddm_wrap_lap",
    # High-level solvers
    "approxchol_lap",
    "approxchol_sddm",
    "approxchol_lap_pcg",
]
