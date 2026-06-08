"""
Random-number sources for the reference approxChol build.

The approximate-Cholesky factorization samples one survivor edge per eliminated
edge using ``rand()`` (a uniform double in ``[0, 1)``).  To make the build both
reproducible *and* checkable against Julia's ``Laplacians.jl``, the build
consumes an abstract :class:`SampleStream` rather than calling a global RNG:

* :class:`GeneratorStream` wraps a seeded NumPy generator for normal use.
* :class:`ArrayStream` replays a fixed array of doubles, so the C++/Python ports
  and Julia can be driven by an identical sample sequence and produce a
  bit-identical ``LDLinv`` factorization (see ``scripts/gen_julia_fixtures.jl``).
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class SampleStream(Protocol):
    """A source of uniform ``[0, 1)`` doubles consumed by the elimination."""

    def rand(self) -> float:
        """
        Return the next uniform sample in ``[0, 1)``.

        Returns:
            The next sample value.

        """
        ...


class GeneratorStream:
    """Uniform sampler backed by a seeded :class:`numpy.random.Generator`."""

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the stream.

        Args:
            seed: Seed for :func:`numpy.random.default_rng`.  ``None`` draws a
                fresh nondeterministic seed.

        """
        self._gen = np.random.default_rng(seed)
        self._count = 0

    def rand(self) -> float:
        """
        Return the next uniform sample in ``[0, 1)``.

        Returns:
            A freshly drawn uniform double.

        """
        self._count += 1
        return float(self._gen.random())

    @property
    def count(self) -> int:
        """
        Number of samples drawn so far.

        Returns:
            The running draw count.

        """
        return self._count


class ArrayStream:
    """Deterministic sampler that replays a pre-generated sequence of doubles."""

    def __init__(self, samples: Sequence[float]) -> None:
        """
        Initialize the stream.

        Args:
            samples: The doubles to replay, in order.

        """
        self._samples = np.asarray(samples, dtype=np.float64)
        self._pos = 0

    def rand(self) -> float:
        """
        Return the next replayed sample.

        Returns:
            The next value in the replay buffer.

        Raises:
            IndexError: If the replay buffer is exhausted.

        """
        if self._pos >= self._samples.size:
            raise IndexError(
                f"ArrayStream exhausted after {self._pos} samples; "
                "the factorization requested more randomness than provided."
            )
        value = float(self._samples[self._pos])
        self._pos += 1
        return value

    @property
    def count(self) -> int:
        """
        Number of samples consumed so far.

        Returns:
            The running consume position.

        """
        return self._pos


def as_stream(rng: SampleStream | int | None) -> SampleStream:
    """
    Coerce a seed / ``None`` / stream into a :class:`SampleStream`.

    Args:
        rng: A :class:`SampleStream`, an integer seed, or ``None``.

    Returns:
        A usable sample stream.  Integers and ``None`` become a
        :class:`GeneratorStream`; an existing stream is returned unchanged.

    """
    if rng is None or isinstance(rng, int):
        return GeneratorStream(rng)
    return rng
