
from evaltools.geography import dissolve, dualgraph, unitmap, invert, reock
from gerrychain.grid import Grid
from gerrychain.graph import Graph
from shapely.geometry import box
import geopandas as gpd
from pathlib import Path
import os
import math

root = Path(os.getcwd()) / Path("tests/test-resources/")

def test_dissolve():
    # Read in geometric data.
    districts = gpd.read_file(root / "test-districts")

    # Assign half the units to district 0, the other half to district 1.
    districts = districts[["geometry", "G20PREDBID"]]

    keep = "G20PREDBID"
    districts["DISTRICT"] = [0]*(len(districts)//2) + [1]*(len(districts)-len(districts)//2)
    dissolved = dissolve(districts, by="DISTRICT", keep=[keep])

    # Assert that we have two distinct geometries.
    assert len(dissolved) == 2

    # Assert that the sum of the kept column is the same as the sum of the original.
    assert dissolved[keep].sum() == districts[keep].sum()


def test_dualgraph():
    # Read in geometric data and get centroids.
    districts = gpd.read_file(root / "test-districts")
    districts["x"] = districts["geometry"].apply(lambda c: c.centroid.coords[0][0])
    districts["y"] = districts["geometry"].apply(lambda c: c.centroid.coords[0][1])
    districts = districts[["district", "geometry", "NAME20", "G20PREDBID", "x", "y"]]

    # Create default dual graph.
    default = dualgraph(districts, index="district")

    # Create an adjusted dual graph.
    adjusted = dualgraph(
        districts, index="district", edges_to_add=[(0, 4)], edges_to_cut=[(0, 6)]
    )
    
    # Create another adjusted dual graph, this time with more things mucked up.
    nameadjusted = dualgraph(
        districts, index="NAME20", colmap={ "G20PREDBID": "BIDEN" }
    )

    # Check that there are different edges in `default` and `adjusted`, and that
    # edges were added and cut.
    assert set(default.edges()) != set(adjusted.edges())
    assert (0, 4) in set(adjusted.edges())
    assert (0, 6) not in set(adjusted.edges())

    # Assert that reindexing and renaming happened.
    assert set(default.nodes()) != set(nameadjusted.nodes())
    for _, data in nameadjusted.nodes(data=True): assert data.get("BIDEN", False)


def test_unitmap():
    # Read in some test dataframes.
    vtds = gpd.read_file(root / "test-vtds")
    counties = gpd.read_file(root / "test-counties")

    # Make an assignment!
    umap = unitmap((vtds, "GEOID20"), (counties, "COUNTYFP"))

    # Assert that the type is a dict and that it has the right number of keys.
    assert type(umap) is dict
    assert len(umap) == len(vtds)

    # Invert it!
    inverse = invert(umap)

    # Assert that we have a dict and that it has as many keys as counties.
    assert type(inverse) is dict
    assert len(inverse) == len(counties)


def test_reock_score_squares_geodataframe():
    grid = Grid((10, 10))
    gdf = gpd.GeoDataFrame([
        {'node': (x, y), 'geometry': box(x, y, x + 1, y + 1)}
        for (x, y) in grid.graph
    ]).set_index('node')
    score_fn = reock(gdf)

    # The Reock score of a square inscribed in a circle is
    # area(square) / area(circle) = 2/π.
    expected_dist_score = 2/ math.pi
    scored = score_fn(grid)

    assert scored.keys() == grid.parts.keys()
    for dist_score in scored.values():
        assert abs(dist_score - expected_dist_score) < 1e-4


def test_reock_score_squares_graph():
    grid = Grid((10, 10))
    for (x, y), data in grid.graph.nodes(data=True):
        data['geometry'] = box(x, y, x + 1, y + 1)
    score_fn = reock(Graph(grid.graph))

    # The Reock score of a square inscribed in a circle is
    # area(square) / area(circle) = 2/π.
    expected_dist_score = 2 / math.pi
    scored = score_fn(grid)

    assert scored.keys() == grid.parts.keys()
    for dist_score in scored.values():
        assert abs(dist_score - expected_dist_score) < 1e-4


def test_reock_score_disconnected():
    grid = Grid((10, 10))
    for (x, y), data in grid.graph.nodes(data=True):
        data['geometry'] = box(x, y, x + 1, y + 1)
    score_fn = reock(Graph(grid.graph))

    # Break district 0 into two disconnected pieces 
    # (while preserving convex hull perimeter),
    # adding a disconnected component to district 3
    # (quadrupling convex hull perimeter).
    grid_disconnected = grid.flip({(0, 1): 3, (1, 0): 3})

    scored = score_fn(grid_disconnected)
    assert scored.keys() == grid_disconnected.parts.keys()
    assert abs(scored[0] - ((23 / 25) * (2 / math.pi))) < 1e-4
    assert abs(scored[1] - (2 / math.pi)) < 1e-4
    assert abs(scored[2] - (2 / math.pi)) < 1e-4
    assert abs(scored[3] == (27 / 25) * (math.pi / 2)) < 1e-4

if __name__ == "__main__":
    root = Path(os.getcwd()) / Path("test-resources/")
