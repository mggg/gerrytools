from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.random import Generator


def resolve_numpy_rng(
    *,
    seed: int | None = None,
    rng: Generator | None = None,
    field_name: str = "seed",
) -> tuple[Generator, int | None]:
    """Build or validate a NumPy generator from a seed or existing generator."""
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError(f"{field_name} must be an integer or None.")
        seed = int(seed)

    if rng is not None and seed is not None:
        raise ValueError(f"Pass either {field_name} or rng, not both.")

    if rng is None:
        return np.random.default_rng(seed), seed

    return rng, seed


def spawn_child_seeds(rng: Generator, count: int) -> list[int]:
    """Derive deterministic child seeds from a parent NumPy generator."""
    if count <= 0:
        return []

    return [
        int(value)
        for value in rng.integers(0, np.iinfo(np.uint64).max, size=count, dtype=np.uint64)
    ]
