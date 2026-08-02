import geopandas as gpd
import pandas as pd
import pytest
from shapely import Polygon, box

from gerrytools._geodataframe import aligned_geodataframe


def test_aligned_geodataframe_uses_explicit_node_identifiers() -> None:
    geometries = gpd.GeoDataFrame(
        {
            "node_id": ["b", "a"],
            "geometry": [box(10, 0, 11, 1), box(0, 0, 1, 1)],
        },
        crs="EPSG:3857",
    )

    aligned = aligned_geodataframe(["a", "b"], geometries, node_column="node_id")

    assert [geometry.bounds for geometry in aligned.geometry] == [
        (0.0, 0.0, 1.0, 1.0),
        (10.0, 0.0, 11.0, 1.0),
    ]


def test_aligned_geodataframe_uses_index_identifiers() -> None:
    geometries = gpd.GeoDataFrame(
        {"geometry": [box(10, 0, 11, 1), box(0, 0, 1, 1)]},
        index=pd.Index([20, 10]),
        crs="EPSG:3857",
    )

    aligned = aligned_geodataframe([10, 20], geometries)

    assert [geometry.bounds for geometry in aligned.geometry] == [
        (0.0, 0.0, 1.0, 1.0),
        (10.0, 0.0, 11.0, 1.0),
    ]


def test_aligned_geodataframe_preserves_tuple_valued_node_identifiers() -> None:
    geometries = gpd.GeoDataFrame(
        {
            "node_id": [(2, "b"), (1, "a")],
            "geometry": [box(10, 0, 11, 1), box(0, 0, 1, 1)],
        },
        crs="EPSG:3857",
    )

    aligned = aligned_geodataframe([(1, "a"), (2, "b")], geometries, node_column="node_id")

    assert [geometry.bounds for geometry in aligned.geometry] == [
        (0.0, 0.0, 1.0, 1.0),
        (10.0, 0.0, 11.0, 1.0),
    ]


def test_aligned_geodataframe_transforms_to_requested_projected_crs() -> None:
    geometries = gpd.GeoDataFrame(
        {"geometry": [box(-105.01, 39.99, -105.00, 40.00)]},
        index=pd.Index(["a"]),
        crs="EPSG:4326",
    )

    aligned = aligned_geodataframe(["a"], geometries, crs="EPSG:26913")

    transformed = aligned.geometry.iloc[0]
    assert str(aligned.crs) == "EPSG:26913"
    assert transformed.area > 0
    assert transformed.bounds[0] > 400_000


@pytest.mark.parametrize(
    ("nodes", "frame", "message"),
    [
        (["a", "a"], gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]), "duplicate"),
        (
            ["a", "b"],
            gpd.GeoDataFrame(
                {"node": ["a", "c"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
                crs="EPSG:3857",
            ),
            "exactly match",
        ),
        (
            ["a"],
            gpd.GeoDataFrame(
                {"node": ["a"], "geometry": [Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])]},
                crs="EPSG:3857",
            ),
            "topologically valid",
        ),
    ],
)
def test_aligned_geodataframe_rejects_unsafe_alignment_or_geometry(nodes, frame, message) -> None:
    node_column = "node" if "node" in frame else None
    with pytest.raises(ValueError, match=message):
        aligned_geodataframe(nodes, frame, node_column=node_column)


def test_aligned_geodataframe_requires_a_projected_crs() -> None:
    geometries = gpd.GeoDataFrame(
        {"geometry": [box(-105.01, 39.99, -105.00, 40.00)]},
        index=pd.Index(["a"]),
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="projected CRS"):
        aligned_geodataframe(["a"], geometries)


def test_aligned_geodataframe_requires_a_geodataframe() -> None:
    with pytest.raises(TypeError, match="GeoDataFrame"):
        aligned_geodataframe(
            ["a"],
            pd.DataFrame({"geometry": [box(0, 0, 1, 1)]}),  # type: ignore
        )


@pytest.mark.parametrize(
    ("nodes", "frame", "node_column", "message"),
    [
        (
            [],
            gpd.GeoDataFrame(geometry=[], crs="EPSG:3857"),
            None,
            "node_order cannot be empty",
        ),
        (
            ["a", None],
            gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:3857"),
            None,
            "missing identifiers",
        ),
        (
            ["a"],
            gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:3857"),
            "node",
            "does not contain node column",
        ),
        (
            ["a", "b"],
            gpd.GeoDataFrame(
                {"node": ["a", "a"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
                crs="EPSG:3857",
            ),
            "node",
            "must be unique",
        ),
        (
            ["a", "b"],
            gpd.GeoDataFrame(
                {"node": ["a", None], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
                crs="EPSG:3857",
            ),
            "node",
            "cannot be missing",
        ),
        (
            ["a"],
            gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], index=pd.Index(["a"])),
            None,
            "must have a CRS",
        ),
    ],
)
def test_aligned_geodataframe_rejects_invalid_identifiers_and_metadata(
    nodes, frame, node_column, message
) -> None:
    with pytest.raises(ValueError, match=message):
        aligned_geodataframe(nodes, frame, node_column=node_column)
