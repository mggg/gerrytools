"""Shared electoral math for the plotting (Matplotlib) and latex (TikZ) backends.

Pure functions over plain arrays and sequences, imported by both backends so the math cannot
drift between them. Values are kept at full float precision; each emitter owns whatever
formatting or rounding its output format requires.

This module owns shared plotting and LaTeX transformations only. Partisan and election scoring
formulas belong in ``gerrytools.scoring``.
"""

from __future__ import annotations

from numbers import Integral
from typing import Iterable, Sequence

import numpy as np

from gerrytools import _partisan_math


def _validated_opposition_counts(pov_counts: np.ndarray, total_counts: np.ndarray) -> np.ndarray:
    """Validate aligned, positive totals and return per-district opposition counts.

    Raises:
        ValueError: If the arrays do not align, contain nonpositive totals, or a district's
            party-of-interest count exceeds its total.
    """
    if pov_counts.shape != total_counts.shape:
        raise ValueError("pov_counts and total_counts must have the same shape.")
    if np.any(total_counts <= 0):
        raise ValueError("total_counts must be positive for all districts.")
    if np.any(pov_counts > total_counts):
        raise ValueError("pov_counts cannot exceed total_counts.")
    return total_counts - pov_counts


def seats_votes_curve_values(
    pov_counts: np.ndarray, total_counts: np.ndarray
) -> tuple[list[float], list[float]]:
    """Compute standard uniform-swing seats-votes step-curve positions.

    The district shares and turnout-weighted overall share use the arithmetic shared with
    :mod:`gerrytools.scoring.formulas`, treating ``total - pov`` as the opposition tally.

    Args:
        pov_counts (np.ndarray): Per-district party-of-interest vote totals.
        total_counts (np.ndarray): Per-district total vote totals.

    Returns:
        tuple[list[float], list[float]]: Vote-share breakpoints (x) and seat-share breakpoints
        (y, ``0..1`` stepped by district rank).

    Raises:
        ValueError: If the arrays do not align, contain nonpositive totals, or a district's
            party-of-interest count exceeds its total.
    """
    opposition_counts = _validated_opposition_counts(pov_counts, total_counts)
    vote_shares = _partisan_math.district_vote_shares(pov_counts, opposition_counts)
    overall_percent = float(_partisan_math.overall_vote_share(pov_counts, opposition_counts))
    breakpoints = _partisan_math.swing_breakpoints(vote_shares, overall_percent)
    vote_share_shift_positions = [0.0] + sorted(map(float, breakpoints)) + [1.0]

    n_seats = len(vote_shares)
    seat_shares_shift_positions = [0.0] + list(map(float, np.arange(n_seats + 1) / n_seats))
    return vote_share_shift_positions, seat_shares_shift_positions


def overall_election_point(pov_counts: np.ndarray, total_counts: np.ndarray) -> tuple[float, float]:
    """Compute the overall election-result point ``(vote share, seat share)``.

    The vote share is the party-of-interest share of all votes cast; the seat share is the
    fraction of districts the party strictly wins. Both use the arithmetic shared with
    :mod:`gerrytools.scoring.formulas`, with ``total - pov`` as the opposition tally.

    Args:
        pov_counts (np.ndarray): Per-district party-of-interest vote totals.
        total_counts (np.ndarray): Per-district total vote totals.

    Returns:
        tuple[float, float]: The ``(vote share, seat share)`` marker position.

    Raises:
        ValueError: If the arrays do not align, contain nonpositive totals, or a district's
            party-of-interest count exceeds its total.
    """
    opposition_counts = _validated_opposition_counts(pov_counts, total_counts)
    total_vote_share = float(_partisan_math.overall_vote_share(pov_counts, opposition_counts))
    total_seat_share = float(np.mean(_partisan_math.district_wins(pov_counts, opposition_counts)))
    return total_vote_share, total_seat_share


def normalize_paintball_data(
    voteshares: Sequence[float],
    seats: Sequence[float],
    total_seats: int | None = None,
) -> tuple[list[float], list[float]]:
    """Validate and normalize incoming vote-share and seat data for a paintball plot.

    ``voteshares`` values must lie in [0, 1]. ``seats`` values are either interpreted directly
    as seat shares in [0, 1] (when ``total_seats`` is None), or as seat counts normalized by
    ``total_seats`` (when provided), and the resulting shares must also lie in [0, 1].

    Args:
        voteshares (Sequence[float]): Vote-share values in ``[0, 1]``.
        seats (Sequence[float]): Seat-share values in ``[0, 1]`` or raw seat counts.
        total_seats (int | None, optional): Total seats used to normalize raw seat counts.
            Defaults to None.

    Returns:
        tuple[list[float], list[float]]: Normalized vote-share and seat-share vectors.

    Raises:
        ValueError: If lengths mismatch, inputs are empty, ``total_seats`` is not a positive
            integer, or normalized values are out of range.
    """
    if len(voteshares) != len(seats):
        raise ValueError("vote_share_data and seats_data must have the same length.")
    if len(voteshares) == 0:
        raise ValueError("vote_share_data and seats_data must have at least one element.")

    normalized_voteshares = [float(vote_share) for vote_share in voteshares]
    if not all(0.0 <= vote_share <= 1.0 for vote_share in normalized_voteshares):
        raise ValueError("All vote-share values must be in [0, 1].")

    if total_seats is None:
        normalized_seatshares = [float(seat_value) for seat_value in seats]
    else:
        if (
            isinstance(total_seats, bool)
            or not isinstance(total_seats, Integral)
            or total_seats <= 0
        ):
            raise ValueError("total_seats must be a positive integer when provided.")
        normalized_seatshares = [float(seat_value) / total_seats for seat_value in seats]

    if not all(0.0 <= seat_share <= 1.0 for seat_share in normalized_seatshares):
        raise ValueError("All seat-share values must be in [0, 1].")

    return normalized_voteshares, normalized_seatshares


def paintball_coordinates(
    voteshares: Sequence[float], seatshares: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Return direct ``(vote share, seat share)`` paintball coordinates.

    Args:
        voteshares (Sequence[float]): Vote-share values in ``[0, 1]``.
        seatshares (Sequence[float]): Seat-share values in ``[0, 1]``.

    Returns:
        tuple[list[float], list[float]]: Vote-share x-coordinates and seat-share y-coordinates.
    """
    return list(voteshares), list(seatshares)


def horizontal_hull_vertices(
    points: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Compute the horizontal hull polygon vertices for a set of points.

    For each distinct y value the hull keeps the minimum and maximum x. Vertices are returned as
    the left side (minimum x, ascending y) followed by the right side (maximum x, descending y),
    ready to close into a polygon.

    Args:
        points (Iterable[tuple[float, float]]): The ``(x, y)`` points to hull.

    Returns:
        list[tuple[float, float]]: Hull vertices in drawing order.
    """
    y_to_minmax_x: dict[float, tuple[float, float]] = {}
    for x_coord, y_coord in points:
        if y_coord not in y_to_minmax_x:
            y_to_minmax_x[y_coord] = (x_coord, x_coord)
            continue

        min_x, max_x = y_to_minmax_x[y_coord]
        y_to_minmax_x[y_coord] = (min(min_x, x_coord), max(max_x, x_coord))

    sorted_y = sorted(y_to_minmax_x)
    left_side = [(y_to_minmax_x[y_val][0], y_val) for y_val in sorted_y]
    right_side = [(y_to_minmax_x[y_val][1], y_val) for y_val in reversed(sorted_y)]
    return left_side + right_side
