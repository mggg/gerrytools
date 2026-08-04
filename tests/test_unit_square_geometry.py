"""Tests for the shared unit-square geometry helpers."""

import math

import pytest

from gerrytools._geometry import line_segment_through_unit_square


# ===================
# == LINE SEGMENTS ==
# ===================
class TestLineSegmentThroughUnitSquare:
    def test_horizontal_line_has_expected_endpoints(self):
        assert line_segment_through_unit_square(0.0) == (0.0, 0.5, 1.0, 0.5)

    def test_vertical_line_has_expected_endpoints(self):
        assert line_segment_through_unit_square(math.inf) == (0.5, 0.0, 0.5, 1.0)

    def test_positive_steep_line_has_expected_endpoints(self):
        assert line_segment_through_unit_square(2.0) == (0.25, 0.0, 0.75, 1.0)

    def test_shallow_line_has_expected_endpoints(self):
        assert line_segment_through_unit_square(0.5) == (0.0, 0.25, 1.0, 0.75)

    def test_negative_steep_line_has_expected_endpoints(self):
        assert line_segment_through_unit_square(-2.0) == (0.75, 0.0, 0.25, 1.0)

    def test_round_to_controls_precision(self):
        assert line_segment_through_unit_square(3.0, round_to=2) == (0.33, 0.0, 0.67, 1.0)

    def test_nan_slope_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            line_segment_through_unit_square(float("nan"))
