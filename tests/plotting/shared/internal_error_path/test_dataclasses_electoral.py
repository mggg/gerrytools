import math
from typing import Any

import numpy as np
import pytest

from gerrytools.plotting.data._unit_square_base import _SlopeLine
from gerrytools.plotting.data.options import (
    SeatsVotesLineOptions,
    SeatsVotesMarkerOptions,
)
from gerrytools.plotting.data.seatsvotes import _SeatsVotesData
from tests.plotting._typing_utils import as_any

# Styling kwargs routed into the two options dataclasses by ``_make_basic``.
_LINE_FIELDS = {"linecolor", "linealpha", "linestyle", "linewidth", "zorder"}
_MARKER_FIELDS = {
    "markerfacecolor",
    "markerfacealpha",
    "marker",
    "markersize",
    "markeredgecolor",
    "markeredgealpha",
    "markeredgewidth",
    "marker_zorder",
}


# ==============
# == LINEDATA ==
# ==============
class TestSeatsVotesData:
    def _make_basic(self, **overrides):
        line_kwargs: dict[str, Any] = {"linecolor": "blue"}
        marker_kwargs: dict[str, Any] = {"markerfacecolor": "gold"}
        data_kwargs: dict[str, Any] = dict(
            pov_party_vote_counts=np.array([300, 400, 600]),
            total_vote_counts=np.array([1000, 1000, 1000]),
            name="SEN20",
            marker_label="Result",
        )
        for key, value in overrides.items():
            if key in _LINE_FIELDS:
                line_kwargs[key] = value
            elif key in _MARKER_FIELDS:
                marker_kwargs[key] = value
            else:
                data_kwargs[key] = value
        return _SeatsVotesData(
            line_style=SeatsVotesLineOptions(**line_kwargs),
            marker_style=SeatsVotesMarkerOptions(**marker_kwargs),
            **data_kwargs,
        )

    def test_default_construction(self):
        svd = self._make_basic()
        assert svd.line_style.zorder == 1
        assert svd.marker_style.marker_zorder == 2

    def test_linealpha_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="linealpha"):
            self._make_basic(linealpha=1.5)

    def test_linealpha_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="linealpha"):
            self._make_basic(linealpha=-0.1)

    def test_linewidth_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            self._make_basic(linewidth=-1.0)

    def test_linewidth_infinite_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            self._make_basic(linewidth=float("inf"))

    def test_markerfacealpha_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="markerfacealpha"):
            self._make_basic(markerfacealpha=2.0)

    def test_markersize_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            self._make_basic(markersize=-1.0)

    def test_markersize_infinite_raises_valueerror(self):
        with pytest.raises(ValueError, match="markersize must be finite"):
            self._make_basic(markersize=float("inf"))

    def test_markeredgealpha_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="markeredgealpha"):
            self._make_basic(markeredgealpha=-0.5)

    def test_markeredgewidth_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            self._make_basic(markeredgewidth=-0.5)

    def test_markeredgewidth_infinite_raises_valueerror(self):
        with pytest.raises(ValueError, match="markeredgewidth must be finite"):
            self._make_basic(markeredgewidth=float("inf"))

    def test_resolved_linewidth_uses_override_when_set(self):
        svd = self._make_basic(linewidth=5.0)
        assert svd.resolved_linewidth(default_linewidth=2.0) == 5.0

    def test_resolved_linewidth_falls_back_to_default(self):
        svd = self._make_basic()
        assert svd.resolved_linewidth(default_linewidth=2.5) == 2.5

    def test_resolved_markersize_uses_override_when_set(self):
        svd = self._make_basic(markersize=10.0)
        assert svd.resolved_markersize(default_markersize=5.0) == 10.0

    def test_resolved_markersize_falls_back_to_default(self):
        svd = self._make_basic()
        assert svd.resolved_markersize(default_markersize=5.0) == 5.0

    def test_resolved_markeredgecolor_defaults_to_markercolor(self):
        svd = self._make_basic()
        assert svd.resolved_markeredgecolor() == svd.marker_style.markerfacecolor

    def test_resolved_markeredgecolor_uses_override(self):
        svd = self._make_basic(markeredgecolor="red")
        # SeatsVotesMarkerOptions normalizes named colors to hex.
        assert svd.resolved_markeredgecolor() == "#ff0000"

    def test_resolved_markeredgealpha_defaults_to_markeralpha(self):
        svd = self._make_basic(markerfacealpha=0.7)
        assert svd.resolved_markeredgealpha() == 0.7

    def test_seats_votes_curve_values_positive_total_votes(self):
        svd = self._make_basic()
        vote_shares, seat_shares = svd.seats_votes_curve_values()
        assert vote_shares[0] == 0.0
        assert vote_shares[-1] == 1.0
        assert seat_shares[0] == 0.0

    def test_seats_votes_curve_rejects_zero_total_votes(self):
        svd = self._make_basic(total_vote_counts=np.array([0, 1000, 1000]))
        with pytest.raises(ValueError, match="positive"):
            svd.seats_votes_curve_values()

    def test_seats_votes_curve_shape_mismatch_raises_valueerror(self):
        svd = self._make_basic(
            pov_party_vote_counts=np.array([300, 400]),
            total_vote_counts=np.array([1000, 1000, 1000]),
            name="bad",
        )
        with pytest.raises(ValueError, match="same shape"):
            svd.seats_votes_curve_values()

    def test_none_linealpha_resolves_from_color(self):
        # SeatsVotesLineOptions resolves an omitted alpha from the (opaque) line color.
        svd = self._make_basic(linealpha=None)
        assert svd.line_style.linealpha == 1.0

    def test_zero_linewidth_is_valid(self):
        svd = self._make_basic(linewidth=0.0)
        assert svd.line_style.linewidth == 0.0


# ===============
# == SLOPELINE ==
# ===============


class TestSlopeLine:
    def test_default_construction(self):
        line = _SlopeLine(slope=1.0, linecolor="grey", linewidth=2.0, linestyle="--")
        assert line.slope == 1.0
        assert line.zorder == -1

    def test_nan_slope_raises_valueerror(self):
        with pytest.raises(ValueError, match="NaN"):
            _SlopeLine(slope=float("nan"), linecolor="grey", linewidth=1.0, linestyle="-")

    def test_infinite_slope_is_valid(self):
        line = _SlopeLine(slope=float("inf"), linecolor="grey", linewidth=1.0, linestyle="-")
        assert math.isinf(line.slope)

    def test_negative_infinite_slope_is_valid(self):
        line = _SlopeLine(slope=float("-inf"), linecolor="grey", linewidth=1.0, linestyle="-")
        assert line.slope == float("-inf")

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            _SlopeLine(slope=1.0, linecolor="grey", linewidth=-1.0, linestyle="-")

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            _SlopeLine(slope=1.0, linecolor="grey", linewidth=float("inf"), linestyle="-")

    def test_linealpha_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="linealpha"):
            _SlopeLine(slope=1.0, linecolor="grey", linewidth=1.0, linestyle="-", linealpha=2.0)

    def test_zero_slope_is_valid(self):
        line = _SlopeLine(slope=0.0, linecolor="grey", linewidth=1.0, linestyle="-")
        assert line.slope == 0.0

    def test_zorder_coerced_to_int(self):
        line = _SlopeLine(
            slope=1.0,
            linecolor="gray",
            linewidth=1.0,
            linestyle="-",
            zorder=as_any(-2.5),
        )
        assert isinstance(line.zorder, int)
