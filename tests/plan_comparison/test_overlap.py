from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from shapely import Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from gerrytools.plan_comparison import areal_overlap, population_overlap


def test_population_overlap_groups_pairs_without_mutating_input() -> None:
    units = pd.DataFrame(
        {
            "proposed": ["new-b", "new-a", "new-b", "new-a"],
            "enacted": [20, 10, 10, 20],
            "population": [6, 5, 2, 1],
        }
    )
    original = units.copy(deep=True)

    result = population_overlap(units, source="proposed", target="enacted", population="population")

    expected = pd.DataFrame(
        [[6.0, 2.0], [1.0, 5.0]],
        index=pd.Index(["new-b", "new-a"], name="proposed"),
        columns=pd.Index([20, 10], name="enacted"),
    )
    assert_frame_equal(result, expected)
    assert_frame_equal(units, original)


@pytest.mark.parametrize("bad_population", [[1, -1], [1, np.inf], [1, np.nan], [1, "x"]])
def test_population_overlap_rejects_invalid_population(bad_population) -> None:
    units = pd.DataFrame({"source": [1, 2], "target": [1, 2], "population": bad_population})

    with pytest.raises(ValueError):
        population_overlap(units, source="source", target="target", population="population")


def test_population_overlap_rejects_missing_assignment_labels() -> None:
    units = pd.DataFrame({"source": [1, None], "target": [1, 2], "population": [1, 1]})

    with pytest.raises(ValueError, match="missing labels"):
        population_overlap(units, source="source", target="target", population="population")


def test_population_overlap_rejects_empty_units() -> None:
    units = pd.DataFrame({"source": [], "target": [], "population": []})

    with pytest.raises(ValueError, match="units cannot be empty"):
        population_overlap(units, source="source", target="target", population="population")


def test_population_overlap_rejects_duplicate_column_labels() -> None:
    # A duplicated label used to surface pandas' opaque "truth value of a Series is ambiguous".
    units = pd.DataFrame(
        [[1, 1, 2, 3]], columns=pd.Index(["source", "source", "target", "population"])
    )

    with pytest.raises(ValueError, match="duplicate column labels.*source"):
        population_overlap(units, source="source", target="target", population="population")


def test_population_overlap_rejects_identical_column_names() -> None:
    units = pd.DataFrame({"source": [1], "population": [1]})

    with pytest.raises(ValueError, match="must name different columns"):
        population_overlap(units, source="source", target="source", population="population")


def test_population_overlap_preserves_tuple_valued_district_labels() -> None:
    units = pd.DataFrame(
        {
            "source": [("new", 1), ("new", 2)],
            "target": [("old", "A"), ("old", "B")],
            "population": [3, 4],
        }
    )

    result = population_overlap(units, source="source", target="target", population="population")

    assert result.at[("new", 1), ("old", "A")] == 3
    assert result.at[("new", 2), ("old", "B")] == 4


def test_areal_overlap_returns_complete_matrix_in_projected_units() -> None:
    source = gpd.GeoDataFrame(
        {"district": ["A", "B"], "geometry": [box(0, 0, 2, 1), box(2, 0, 4, 1)]},
        crs="EPSG:3857",
    )
    target = gpd.GeoDataFrame(
        {"district": ["X", "Y"], "geometry": [box(0, 0, 1, 1), box(1, 0, 4, 1)]},
        crs="EPSG:3857",
    )
    source_original = source.copy(deep=True)
    target_original = target.copy(deep=True)

    result = areal_overlap(
        source,
        target,
        source_label="district",
        target_label="district",
    )

    expected = pd.DataFrame(
        [[1.0, 1.0], [0.0, 2.0]],
        index=pd.Index(["A", "B"], name="district"),
        columns=pd.Index(["X", "Y"], name="district"),
    )
    assert_frame_equal(result, expected)
    assert_frame_equal(source, source_original)
    assert_frame_equal(target, target_original)


def test_areal_overlap_includes_zero_for_disjoint_pairs() -> None:
    source = gpd.GeoDataFrame({"source": ["A"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:3857")
    target = gpd.GeoDataFrame({"target": ["B"], "geometry": [box(2, 0, 3, 1)]}, crs="EPSG:3857")

    result = areal_overlap(source, target, source_label="source", target_label="target")

    assert result.at["A", "B"] == 0


def test_areal_overlap_reprojects_both_frames_when_crs_is_given() -> None:
    # Web-Mercator gridlines follow lat/lon gridlines, so axis-aligned geographic boxes stay
    # boxes after reprojection and the zero/nonzero overlap pattern is preserved exactly.
    source = gpd.GeoDataFrame(
        {"source": ["A", "B"], "geometry": [box(0, 0, 1, 1), box(2, 0, 3, 1)]},
        crs="EPSG:4326",
    )
    target = gpd.GeoDataFrame(
        {"target": ["X"], "geometry": [box(0, 0, 1, 1)]},
        crs="EPSG:4326",
    )

    result = areal_overlap(
        source, target, source_label="source", target_label="target", crs="EPSG:3857"
    )

    assert result.at["A", "X"] > 0
    assert result.at["B", "X"] == 0


def test_areal_overlap_requires_projected_crs() -> None:
    source = gpd.GeoDataFrame(
        {"source": ["A"], "geometry": [box(-105, 39, -104, 40)]}, crs="EPSG:4326"
    )
    target = gpd.GeoDataFrame(
        {"target": ["B"], "geometry": [box(-105, 39, -104, 40)]}, crs="EPSG:4326"
    )

    with pytest.raises(ValueError, match="projected CRS"):
        areal_overlap(source, target, source_label="source", target_label="target")


def test_areal_overlap_rejects_duplicate_labels_and_invalid_geometry() -> None:
    duplicate = gpd.GeoDataFrame(
        {"source": ["A", "A"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
        crs="EPSG:3857",
    )
    target = gpd.GeoDataFrame({"target": ["B"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:3857")
    with pytest.raises(ValueError, match="source labels must be unique"):
        areal_overlap(duplicate, target, source_label="source", target_label="target")

    invalid = gpd.GeoDataFrame(
        {
            "source": ["A"],
            "geometry": [Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])],
        },
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="topologically valid"):
        areal_overlap(invalid, target, source_label="source", target_label="target")


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("not_geodataframe", "must be GeoDataFrames"),
        ("empty", "cannot be empty"),
        ("target_duplicates", "target labels must be unique"),
        ("missing_crs", "must have a CRS"),
        ("missing_geometry", "cannot contain missing geometries"),
        ("empty_geometry", "cannot contain empty geometries"),
        ("point", "only Polygon or MultiPolygon"),
    ],
)
def test_areal_overlap_rejects_invalid_input_contracts(
    corruption: str,
    message: str,
) -> None:
    source_frame = gpd.GeoDataFrame(
        {"source": ["A"], "geometry": [box(0, 0, 1, 1)]},
        crs="EPSG:3857",
    )
    target_frame = gpd.GeoDataFrame(
        {"target": ["B"], "geometry": [box(0, 0, 1, 1)]},
        crs="EPSG:3857",
    )
    source: object = source_frame
    target: object = target_frame
    if corruption == "not_geodataframe":
        source = pd.DataFrame(source_frame)
    elif corruption == "empty":
        source = source_frame.iloc[:0]
    elif corruption == "target_duplicates":
        target = pd.concat([target_frame, target_frame], ignore_index=True)
    elif corruption == "missing_crs":
        source = gpd.GeoDataFrame({"source": ["A"], "geometry": [box(0, 0, 1, 1)]})
    elif corruption == "missing_geometry":
        source = gpd.GeoDataFrame(
            {"source": ["A"]},
            geometry=cast(list[BaseGeometry], [None]),
            crs="EPSG:3857",
        )
    elif corruption == "empty_geometry":
        source = gpd.GeoDataFrame(
            {"source": ["A"]},
            geometry=[Polygon()],
            crs="EPSG:3857",
        )
    else:
        source = gpd.GeoDataFrame(
            {"source": ["A"]},
            geometry=[Point(0, 0)],
            crs="EPSG:3857",
        )

    with pytest.raises((TypeError, ValueError), match=message):
        areal_overlap(
            cast(gpd.GeoDataFrame, source),
            cast(gpd.GeoDataFrame, target),
            source_label="source",
            target_label="target",
        )


def test_areal_overlap_rejects_geometry_corrupted_by_reprojection(monkeypatch) -> None:
    source = gpd.GeoDataFrame(
        {"source": ["A"], "geometry": [box(0, 0, 1, 1)]},
        crs="EPSG:3857",
    )
    target = gpd.GeoDataFrame(
        {"target": ["B"], "geometry": [box(0, 0, 1, 1)]},
        crs="EPSG:3857",
    )
    original_to_crs = gpd.GeoDataFrame.to_crs

    def corrupt_geometry(frame, *args, **kwargs):
        projected = original_to_crs(frame, *args, **kwargs)
        projected.geometry = [Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])]
        return projected

    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", corrupt_geometry)

    with pytest.raises(ValueError, match="after reprojection.*topologically valid"):
        areal_overlap(source, target, source_label="source", target_label="target")
