r"""
Symbolic perturbation compiler.

Turns a symbolic kernel ``h(xi)`` into everything the solver needs, automatically:

1. derive ``h' = dh/dxi`` and ``h'' = d^2h/dxi^2`` symbolically (SymPy);
2. lambdify ``h, h', h''`` to vectorized NumPy callables;
3. certify strict convexity on the box -- exactly via the Bernstein condition
   when ``h''`` is polynomial, else by dense sampling;
4. **attempt a closed-form inverse** of the strictly-monotone ``h'(xi) = eta``.
   When SymPy returns elementary roots (e.g. degree ``<= 4`` polynomials), the
   compiler builds a vectorized selector that evaluates the candidate radicals
   and picks the unique real root inside the box; otherwise it falls back to the
   safeguarded-Newton root-find.

The result is a :class:`CompiledKernel` descriptor and a ready-to-use
:class:`SymbolicPerturbation`, so a user-supplied strictly-convex ``h`` gets an
optimized inverse with no hand-coding -- the generalization of the hand-written
closed-form kernels.  (For the production polynomial sieve the analytic
implementation in ``polynomial_sieve`` is used directly; the compiler is the
general path and the closed-form-vs-root-find cross-check.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import sympy as sp

from ..utils.typing import ArrayLike
from ._bernstein import is_convex
from ._rootfind import solve_monotone
from .base import SeparablePerturbation

_IMAG_TOL = 1e-9
_BOX_TOL = 1e-9


@dataclass
class CompiledKernel:
    """
    Compiled artifacts for a symbolic kernel ``h(xi)``.

    Attributes:
        h: Vectorized ``h(xi)``.
        hp: Vectorized ``h'(xi)``.
        hpp: Vectorized ``h''(xi)``.
        inverse: Vectorized ``(eta, lo, hi) -> (xi, interior)`` closed-form
            inverse of ``h'`` if one was found, else ``None``.
        degree: Polynomial degree of ``h'`` (``None`` if not polynomial).
        has_closed_form: Whether :attr:`inverse` is available.
        convex_on_box: Whether ``h''`` was certified positive on the box.
        expr: The original ``h`` expression.
        symbol: The ``xi`` symbol.

    """

    h: Callable[[ArrayLike], ArrayLike]
    hp: Callable[[ArrayLike], ArrayLike]
    hpp: Callable[[ArrayLike], ArrayLike]
    inverse: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]]
    degree: Optional[int]
    has_closed_form: bool
    convex_on_box: Optional[bool]
    expr: sp.Expr
    symbol: sp.Symbol


def _polynomial_degree(expr: sp.Expr, xi: sp.Symbol) -> Optional[int]:
    """
    Return the polynomial degree of ``expr`` in ``xi``, or ``None``.

    Args:
        expr: A SymPy expression.
        xi: The variable.

    Returns:
        The integer degree if ``expr`` is a polynomial in ``xi``, else ``None``.

    """
    poly = sp.Poly(expr, xi) if expr.is_polynomial(xi) else None
    return int(poly.degree()) if poly is not None else None


def _build_inverse(
    hp_expr: sp.Expr, xi: sp.Symbol
) -> Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]]:
    """
    Build a vectorized closed-form inverse of ``h'(xi) = eta`` if SymPy can.

    Args:
        hp_expr: The symbolic first derivative ``h'``.
        xi: The variable.

    Returns:
        A callable ``(eta, lo, hi) -> (xi, interior)`` using the closed-form
        radicals, or ``None`` if no elementary solution was found.

    """
    eta = sp.Symbol("_eta_target", real=True)
    try:
        solutions: List[sp.Expr] = sp.solve(sp.Eq(hp_expr, eta), xi)
    except Exception:  # pragma: no cover - SymPy failure is a clean fallback
        return None
    if not solutions:
        return None

    candidate_funcs = [sp.lambdify(eta, sol, "numpy") for sol in solutions]
    hp_func = sp.lambdify(xi, hp_expr, "numpy")

    def inverse(eta_val: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        eta_arr, lo_arr, hi_arr = np.broadcast_arrays(
            np.asarray(eta_val, dtype=float),
            np.asarray(lo, dtype=float),
            np.asarray(hi, dtype=float),
        )
        ones = np.ones(eta_arr.shape)
        # Saturation decided by the monotone boundary values.
        g_lo = hp_func(lo_arr) * ones
        g_hi = hp_func(hi_arr) * ones
        below = eta_arr <= g_lo
        above = eta_arr >= g_hi

        xi_out = np.where(below, lo_arr, np.where(above, hi_arr, np.nan))
        need = ~(below | above)
        for func in candidate_funcs:
            with np.errstate(all="ignore"):
                cand = np.asarray(func(eta_arr), dtype=complex) * ones
            real_part = cand.real
            valid = (
                need
                & np.isnan(xi_out)
                & (np.abs(cand.imag) < _IMAG_TOL)
                & (real_part >= lo_arr - _BOX_TOL)
                & (real_part <= hi_arr + _BOX_TOL)
            )
            xi_out = np.where(valid, np.clip(real_part, lo_arr, hi_arr), xi_out)
        # Any coordinate still unresolved (no candidate in box) falls back to the
        # nearest bound; this should not happen for a strictly monotone h'.
        xi_out = np.where(
            np.isnan(xi_out), np.clip(0.5 * (lo_arr + hi_arr), lo_arr, hi_arr), xi_out
        )
        interior = (xi_out > lo_arr) & (xi_out < hi_arr)
        return xi_out, interior

    return inverse


def _check_convex_on_box(
    hpp_expr: sp.Expr, xi: sp.Symbol, domain: Tuple[float, float]
) -> Optional[bool]:
    """
    Certify ``h'' > 0`` on the box, exactly for polynomials else by sampling.

    Args:
        hpp_expr: The symbolic second derivative ``h''``.
        xi: The variable.
        domain: The box ``(lo, hi)``.

    Returns:
        ``True``/``False`` from the Bernstein test on ``[0,1]`` when applicable,
        else a sampled verdict, or ``None`` if it cannot be assessed.

    """
    lo, hi = domain
    if hpp_expr.is_polynomial(xi) and (lo, hi) == (0.0, 1.0):
        # h'' = 1 + sum gamma_l (l-1) xi^{l-2}; recover the gamma_l (l-1) coeffs.
        poly = sp.Poly(hpp_expr, xi)
        coeffs = {p: float(c) for (p,), c in poly.terms()}
        const = coeffs.get(0, 0.0)
        if abs(const - 1.0) < 1e-12:
            max_pow = max(coeffs) if coeffs else 0
            gamma = np.zeros(max_pow)  # gamma_3..gamma_{max_pow+2}
            for power, c in coeffs.items():
                if power >= 1:
                    gamma[power - 1] = c / (power + 1)  # c = gamma*(l-1), l-1=power+1
            return is_convex(gamma)
    hpp_func = sp.lambdify(xi, hpp_expr, "numpy")
    grid = np.linspace(lo, hi, 257)[1:-1]
    return bool(np.all(np.asarray(hpp_func(grid), dtype=float) > 0))


def compile_kernel(
    h_expr: sp.Expr, xi: sp.Symbol, *, domain: Tuple[float, float] = (0.0, 1.0)
) -> CompiledKernel:
    """
    Compile a symbolic kernel ``h(xi)`` into a :class:`CompiledKernel`.

    Args:
        h_expr: The kernel ``h`` as a SymPy expression in ``xi`` (parameters
            already substituted to numbers).
        xi: The ``xi`` symbol.
        domain: The per-coordinate box ``(lo, hi)`` for the convexity check.

    Returns:
        The compiled kernel artifacts.

    """
    hp_expr = sp.diff(h_expr, xi)
    hpp_expr = sp.diff(hp_expr, xi)
    h = sp.lambdify(xi, h_expr, "numpy")
    hp = sp.lambdify(xi, hp_expr, "numpy")
    hpp = sp.lambdify(xi, hpp_expr, "numpy")
    degree = _polynomial_degree(hp_expr, xi)
    inverse = _build_inverse(hp_expr, xi)
    convex = _check_convex_on_box(hpp_expr, xi, domain)
    return CompiledKernel(
        h=h,
        hp=hp,
        hpp=hpp,
        inverse=inverse,
        degree=degree,
        has_closed_form=inverse is not None,
        convex_on_box=convex,
        expr=h_expr,
        symbol=xi,
    )


class SymbolicPerturbation(SeparablePerturbation):
    """
    A separable perturbation built from a symbolic kernel via the compiler.

    Uses the auto-derived closed-form inverse when available, else the
    safeguarded-Newton root-find -- both selectable to enable cross-checking.

    Args:
        h_expr: The kernel ``h`` as a SymPy expression in ``xi``.
        xi: The ``xi`` symbol (a default ``Symbol('xi')`` if omitted).
        domain: The per-coordinate box used for the convexity certificate.
        method: ``"auto"`` (closed-form if available), ``"closed"`` (force
            closed-form; error if none), or ``"rootfind"`` (force the iteration).

    """

    cvxpy_expressible = False

    def __init__(
        self,
        h_expr: sp.Expr,
        xi: Optional[sp.Symbol] = None,
        *,
        domain: Tuple[float, float] = (0.0, 1.0),
        method: str = "auto",
    ) -> None:
        xi = xi if xi is not None else sp.Symbol("xi", real=True)
        self.kernel = compile_kernel(h_expr, xi, domain=domain)
        self.default_domain = domain
        if method not in ("auto", "closed", "rootfind"):
            raise ValueError(f"unknown method {method!r}")
        if method == "closed" and not self.kernel.has_closed_form:
            raise ValueError("no closed-form inverse was found for this kernel")
        self.method = method
        self.has_closed_form_recovery = self.kernel.has_closed_form and method != "rootfind"

    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return the compiled kernel value at ``xi``."""
        del params
        return np.asarray(self.kernel.h(np.asarray(xi, dtype=float)), dtype=float)

    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h'(xi)``."""
        del params
        return np.asarray(self.kernel.hp(np.asarray(xi, dtype=float)), dtype=float)

    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h''(xi)``."""
        del params
        xi = np.asarray(xi, dtype=float)
        return np.asarray(self.kernel.hpp(xi), dtype=float) * np.ones_like(xi)

    def primal_recovery(
        self,
        eta: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
        params: Any,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Recover ``xi*(eta)`` via the closed-form inverse or the root-find."""
        del params
        use_closed = self.kernel.has_closed_form and self.method != "rootfind"
        if use_closed:
            return self.kernel.inverse(eta, lo, hi)
        return solve_monotone(
            lambda z: self.hprime(z, None), lambda z: self.hsecond(z, None), eta, lo, hi
        )

    def gamma_feasible(self, params: Any) -> bool:
        """Return the compile-time convexity verdict (``True`` if undetermined)."""
        del params
        return True if self.kernel.convex_on_box is None else self.kernel.convex_on_box
