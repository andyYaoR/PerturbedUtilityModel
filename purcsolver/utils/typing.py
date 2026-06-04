"""
Shared type aliases for PURCSolver.

The package convention is numpy/scipy arrays on CPU (no autograd is required; the
linear solve is delegated to LaplacianSolve which consumes numpy/scipy).  These
aliases keep signatures readable and leave room for a torch path later without an
API change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import scipy.sparse as sp

    SparseMatrix = sp.spmatrix
else:
    SparseMatrix = object

# A dense real array (function values, multipliers, weights, ...).
ArrayLike = np.ndarray

# Model parameters theta = (beta, gamma): beta are the K utility coefficients,
# gamma are the shape parameters of the perturbation (e.g. the polynomial-sieve
# coefficients gamma_3..gamma_L).  Either may be empty for a fixed sub-model.
Theta = Tuple[ArrayLike, ArrayLike]

# Anything we accept where a real vector is expected.
RealVector = Union[ArrayLike, "list[float]", Tuple[float, ...]]
