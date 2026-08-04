"""Shared geometry helpers for the plotting and latex modules."""

from __future__ import annotations

import math


def line_segment_through_unit_square(
    slope: float, *, round_to: int | None = None
) -> tuple[float, float, float, float]:
    """Compute line endpoints inside the unit square for a line through ``(0.5, 0.5)``.

    Args:
        slope (float): Slope of the line.
        round_to (int | None, optional): Decimal places used to round each
            coordinate, or None for no rounding. The TikZ emitters round to
            keep the generated LaTeX compact. Defaults to None.

    Returns:
        tuple[float, float, float, float]: Endpoints in ``(x0, y0, x1, y1)`` order.

    Raises:
        ValueError: If ``slope`` is NaN.
    """
    # NaN fails both the isinf and abs comparisons, which would silently yield NaN endpoints.
    if math.isnan(slope):
        raise ValueError("slope cannot be NaN")
    if math.isinf(slope):
        starting_x, starting_y = 0.5, 0.0
        ending_x, ending_y = 0.5, 1.0
    elif abs(slope) >= 1:
        # Steep lines exit through the top and bottom edges.
        starting_x, starting_y = 0.5 - (0.5 / slope), 0.0
        ending_x, ending_y = 0.5 + (0.5 / slope), 1.0
    else:
        # Shallow lines (including slope 0) exit through the side edges.
        starting_x, starting_y = 0.0, 0.5 - (0.5 * slope)
        ending_x, ending_y = 1.0, 0.5 + (0.5 * slope)

    if round_to is None:
        return (starting_x, starting_y, ending_x, ending_y)
    return (
        round(starting_x, round_to),
        round(starting_y, round_to),
        round(ending_x, round_to),
        round(ending_y, round_to),
    )
