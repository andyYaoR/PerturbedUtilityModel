"""
Registry of constraint-polytope geometries.

M0 ships the :class:`Polytope` ABC only.  Concrete geometries land in M1
(:class:`GeneralPolytope`) and M4 (:class:`IncidencePolytope`), along with
builder helpers (``from_dense``, ``from_incidence``) that auto-detect node-arc
incidence structure and select the network fast path.
"""

from __future__ import annotations

from .base import Polytope
from .general import GeneralPolytope

__all__ = ["Polytope", "GeneralPolytope"]
