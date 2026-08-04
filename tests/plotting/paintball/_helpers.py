"""Shared test helpers for the PaintballPlot test suite."""

from __future__ import annotations

from collections.abc import Iterable

from gerrytools.plotting.data.paintball import PaintballPlot


def simple_paintball(
    vote_share_data: Iterable[float] | None = None,
    seats_data: Iterable[float] | None = None,
    *,
    add_efficiency_gap_line: bool = True,
    add_proportionality_line: bool = True,
    **constructor_kwargs,
) -> PaintballPlot:
    """Build a ``PaintballPlot`` with sensible defaults for tests.

    Args:
        vote_share_data: Override the default vote-share values
            (``[0.4, 0.5, 0.6]`` if omitted).
        seats_data: Override the default seat-share values
            (``[0.3, 0.5, 0.7]`` if omitted).
        add_efficiency_gap_line: Whether to add the efficiency-gap guide
            line after construction. Defaults to True.
        add_proportionality_line: Whether to add the proportionality guide
            line after construction. Defaults to True.
        **constructor_kwargs: Forwarded to ``PaintballPlot(...)``.

    Returns:
        A configured ``PaintballPlot`` instance ready for further test setup.
    """
    if vote_share_data is None:
        vote_share_data = [0.4, 0.5, 0.6]
    if seats_data is None:
        seats_data = [0.3, 0.5, 0.7]
    plot = PaintballPlot(**constructor_kwargs)
    plot.add_seats_votes_data(vote_share_data, seats_data)
    if add_efficiency_gap_line:
        plot.add_efficiency_gap_line()
    if add_proportionality_line:
        plot.add_proportionality_line()
    return plot
