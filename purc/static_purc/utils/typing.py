"""
Shared type aliases for PURCSolver.

The package is torch-native: tensors are the primary data type on the public API
and throughout the solver (CPU ``float64`` by default, written
device-agnostically).  NumPy/SciPy appear only behind zero-copy bridges (see
:mod:`purcsolver.utils.torch_compat`) for the few dependencies that require them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    import scipy.sparse as sp

    SparseMatrix = sp.spmatrix
else:
    SparseMatrix = object

# A real tensor (function values, multipliers, weights, ...).  Torch is primary;
# numpy arrays are accepted at the boundary and coerced zero-copy.
ArrayLike = Union[torch.Tensor, np.ndarray]

# Model parameters theta = (beta, gamma): beta are the K utility coefficients,
# gamma are the perturbation shape parameters.  Either may be empty.
Theta = Tuple[ArrayLike, ArrayLike]
