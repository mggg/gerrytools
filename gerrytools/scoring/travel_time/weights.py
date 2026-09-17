"""Trip weight modes for travel-time scores.

A *trip* has two endpoints ``a`` and ``b`` (unit ids) with populations
``pop_a`` and ``pop_b``. The weight mode chooses how much that trip counts
in a weighted mean or weighted max.
"""

from __future__ import annotations

from typing import Literal

WeightMode = Literal["none", "p+p", "p*p"]

WEIGHT_MODES: tuple[WeightMode, ...] = ("none", "p+p", "p*p")


def trip_weight(pop_a: float, pop_b: float, mode: WeightMode) -> float:
    """Return the weight for a trip between two units.

    Args:
        pop_a: Population at endpoint a (nonnegative).
        pop_b: Population at endpoint b (nonnegative).
        mode: ``"none"`` → 1; ``"p+p"`` → pop_a + pop_b; ``"p*p"`` → pop_a * pop_b.

    Returns:
        Nonnegative weight. May be 0 when populations are 0 under ``p+p`` / ``p*p``.
    """
    if mode == "none":
        return 1.0
    if mode == "p+p":
        return float(pop_a) + float(pop_b)
    if mode == "p*p":
        return float(pop_a) * float(pop_b)
    raise ValueError(f"Unknown weight mode {mode!r}; expected one of {WEIGHT_MODES}")
