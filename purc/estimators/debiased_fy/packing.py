"""
Flat-vector layout for the parameter ``theta = (beta, gamma)`` (torch).

A single source of truth for the ordering ``theta = [beta (K), gamma (L-2)]`` so
the loss gradient, the finite-difference Hessian, and the sandwich variance all
agree on indexing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...static_purc.utils.torch_compat import DEFAULT_DTYPE, as_tensor


@dataclass(frozen=True)
class ParamLayout:
    """
    Index layout of the flat parameter vector ``theta = [beta, gamma]``.

    Attributes:
        n_beta: Number of utility coefficients ``K``.
        n_gamma: Number of sieve coefficients ``L - 2`` (degrees ``3..L``).

    """

    n_beta: int
    n_gamma: int

    @property
    def size(self) -> int:
        """Total parameter dimension ``K + (L - 2)``."""
        return self.n_beta + self.n_gamma

    @property
    def beta_slice(self) -> slice:
        """Slice of ``theta`` holding ``beta``."""
        return slice(0, self.n_beta)

    @property
    def gamma_slice(self) -> slice:
        """Slice of ``theta`` holding ``gamma``."""
        return slice(self.n_beta, self.size)

    def pack(self, beta, gamma) -> torch.Tensor:
        """
        Concatenate ``(beta, gamma)`` into a flat tensor.

        Args:
            beta: Utility coefficients, shape ``[K]``.
            gamma: Sieve coefficients, shape ``[L-2]``.

        Returns:
            The flat ``theta``, shape ``[K + L - 2]``.

        """
        b = as_tensor(beta).to(DEFAULT_DTYPE).reshape(-1)
        g = as_tensor(gamma).to(DEFAULT_DTYPE).reshape(-1)
        return torch.cat([b, g])

    def unpack(self, theta):
        """
        Split a flat ``theta`` into ``(beta, gamma)``.

        Args:
            theta: Flat parameter vector, shape ``[K + L - 2]``.

        Returns:
            ``(beta, gamma)`` tensors (clones, safe to mutate).

        """
        theta = as_tensor(theta).to(DEFAULT_DTYPE).reshape(-1)
        return theta[self.beta_slice].clone(), theta[self.gamma_slice].clone()
