"""
Registry of swappable separable perturbations.

Concrete kernels (quadratic, entropy, modified entropy, Tsallis, polynomial
sieve) are registered here and resolved by name via :func:`get_perturbation`.
M0 ships the registry and the :class:`SeparablePerturbation` ABC only; concrete
kernels land in M1 (closed-form) and M3 (sieve + symbolic compiler).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

from .base import SeparablePerturbation

# name -> SeparablePerturbation subclass
PERTURBATIONS: Dict[str, Type[SeparablePerturbation]] = {}


def register_perturbation(
    name: str,
) -> Callable[[Type[SeparablePerturbation]], Type[SeparablePerturbation]]:
    """
    Class decorator that registers a perturbation under ``name``.

    Args:
        name: Registry key (e.g. ``"quadratic"``).

    Returns:
        The decorator that records the class and returns it unchanged.  The
        returned decorator raises ``ValueError`` if ``name`` is already taken.

    """

    def _decorator(cls: Type[SeparablePerturbation]) -> Type[SeparablePerturbation]:
        if name in PERTURBATIONS:
            raise ValueError(f"perturbation {name!r} already registered")
        PERTURBATIONS[name] = cls
        return cls

    return _decorator


def get_perturbation(name: str, **kwargs: Any) -> SeparablePerturbation:
    """
    Instantiate a registered perturbation by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the perturbation constructor.

    Returns:
        A new perturbation instance.

    Raises:
        KeyError: If ``name`` is not registered.

    """
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; registered: {sorted(PERTURBATIONS)}")
    return PERTURBATIONS[name](**kwargs)


__all__ = [
    "SeparablePerturbation",
    "PERTURBATIONS",
    "register_perturbation",
    "get_perturbation",
]

# Import concrete kernels for their registration side effects.  Kept at the
# bottom so ``register_perturbation`` is defined before these modules import it.
from . import entropy, modified_entropy, quadratic  # noqa: E402,F401
