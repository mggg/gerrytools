from gerrytools.scoring import (
    deviations, splits, pieces, unassigned_units, contiguous, reock, 
    polsby_popper, schwartzberg, convex_hull, pop_polygon, summarize
)
from gerrychain import Graph, Partition
import geopandas as gpd
from gerrychain.grid import Grid
from shapely.geometry import box
from math import pi, sqrt
from pathlib import Path
import pytest

from .utils import remotegraphresource

@pytest.fixture(scope="module")
def ia_dataframe():
    """`GeoDataFrame` of Iowa counties."""
    shp_path = Path(__file__).resolve().parent / "fixtures" / "ia_county_with_enacted_2020.zip"
    return gpd.read_file(shp_path)


@pytest.fixture(scope="module")
def ia_graph():
    """NetworkX `Graph` of Iowa counties."""
    return Graph.from_json(
        Path(__file__).resolve().parent
        / "fixtures"
        / "ia_county_with_enacted_2020.json"
    )
   
    
@pytest.fixture(scope="module") 
def ia_enacted(ia_graph):
    return Partition(graph=ia_graph, assignment="DISTRICT") 


def test_splits_pandas():
    # Read in an existing dual graph.
    dg = remotegraphresource("test-graph.json")
    P = Partition(dg, "CONGRESS")

    geometricsplits = splits("COUNTYFP20", popcol="TOTPOP").apply(P)
    geometricsplitsnames = splits("COUNTYFP20", popcol="TOTPOP", names=True).apply(P)

    split = {'005', '135', '085', '067', '091', '045', '097', '017'}

    pairs = [
        (geometricsplits, geometricsplitsnames),
    ]

    for unnamed, named in pairs:
        # Assert that we have a dictionary and that we have the right keys in it.
        assert type(unnamed) is int
        assert unnamed == 8

        # Make sure that we're counting the number of splits correctly – Indiana's
        # enacted Congressional plan should split counties 8 times.
        assert len(named) == 8
        assert set(split) == set(named)


@pytest.mark.xfail(reason="The provided Partitions are not GeometricPartitions, and should fail.")
def test_splits_gerrychain():
    # Read in an existing dual graph.
    dg = remotegraphresource("test-graph.json")
    P = Partition(dg, "CONGRESS")

    geometricsplitsgc = splits("COUNTYFP20", popcol="TOTPOP", how="gerrychain").apply(P)
    geometricsplitsnamesgc = splits("COUNTYFP20", popcol="TOTPOP", how="gerrychain", names=True).apply(P)

    split = {'005', '135', '085', '067', '091', '045', '097', '017'}

    pairs = [
        (geometricsplitsgc, geometricsplitsnamesgc)
    ]

    for unnamed, named in pairs:
        # Assert that we have a dictionary and that we have the right keys in it.
        assert type(unnamed) is int
        assert unnamed == 8

        # Make sure that we're counting the number of splits correctly – Indiana's
        # enacted Congressional plan should split counties 8 times.
        assert len(named) == 8
        assert set(split) == set(named)


def test_pieces_pandas():
    # Read in an existing dual graph.
    dg = remotegraphresource("test-graph.json")
    P = Partition(dg, "CONGRESS")

    geometricpieces = pieces("COUNTYFP20", popcol="TOTPOP").apply(P)
    geometricpiecesnames = pieces("COUNTYFP20", popcol="TOTPOP", names=True).apply(P)

    split = {'005', '135', '085', '067', '091', '045', '097', '017'}

    pairs = [
        (geometricpieces, geometricpiecesnames),
    ]

    for unnamed, named in pairs:
        # Assert that we have a dictionary and that we have the right keys in it.
        assert type(unnamed) is int
        assert unnamed == 16

        # Make sure that we're counting the number of splits correctly – Indiana's
        # enacted Congressional plan should split counties 8 times.
        assert len(named) == 8
        assert set(split) == set(named)


@pytest.mark.xfail(reason="The provided Partitions are not GeometricPartitions, and should fail.")
def test_pieces_gerrychain():
    # Read in an existing dual graph.
    dg = remotegraphresource("test-graph.json")
    P = Partition(dg, "CONGRESS")

    geometricpiecesgc = pieces("COUNTYFP20", popcol="TOTPOP", how="gerrychain").apply(P)
    geometricpiecesnamesgc = pieces("COUNTYFP20", popcol="TOTPOP", how="gerrychain", names=True).apply(P)

    split = {'005', '135', '085', '067', '091', '045', '097', '017'}

    pairs = [
        (geometricpiecesgc, geometricpiecesnamesgc)
    ]

    for unnamed, named in pairs:
        # Assert that we have a dictionary and that we have the right keys in it.
        assert type(unnamed) is int
        assert unnamed == 16

        # Make sure that we're counting the number of splits correctly – Indiana's
        # enacted Congressional plan should split counties 8 times.
        assert len(named) == 8
        assert set(split) == set(named)


def test_deviations():
    dg = remotegraphresource("test-graph.json")
    P = Partition(dg, "CONGRESS")
    devs = deviations(P, "TOTPOP")

    assert type(devs) is dict
    assert set(data["CONGRESS"] for _, data in dg.nodes(data=True)) == set(devs.keys())


def test_contiguity():
    dg = remotegraphresource("test-graph.json")
    P = Partition(dg, "CONGRESS")
    contiguity = contiguous(P)

    # This plan should *not* be contiguous, as some VTDs are discontiguous
    # themselves.
    assert not contiguity


def test_unassigned_units():
    dg = remotegraphresource("test-graph.json")
    P = Partition(dg, "CONGRESS")
    bads = unassigned_units(P)
    wholebads = unassigned_units(P, raw=True)

    # This plan shouldn't have any unassigned units.
    assert bads == 0

    # These two things should report the same number, since none are unassigned.
    assert bads == wholebads
    

def test_polsby_popper__iowa_counties(ia_enacted, ia_dataframe):
    scores = summarize(ia_enacted, [polsby_popper()], gdf=ia_dataframe, join_on="GEOID20")
    avg_polsby = sum(scores["polsby_popper"].values()) / len(scores["polsby_popper"])
    # Outside source: the value returned by Dave's for the enacted Iowa plan in 0.3116,
    # which is sufficiently close to our value (any differences are likely attributable
    # to resolution).
    # https://davesredistricting.org/maps#ratings::628d5e9a-bd35-4248-aa8c-73af095e0135
    assert abs(avg_polsby - 0.31076) < 1e-4
    

def test_schwartzberg__iowa_counties(ia_enacted, ia_dataframe):
    scores = summarize(ia_enacted, [schwartzberg()], gdf=ia_dataframe, join_on="GEOID20")
    avg_schwartzberg = sum(scores["schwartzberg"].values()) / len(scores["schwartzberg"])
    # Ground truth computed from known-good Polsby-Popper scores; recall that
    # for a single district, Schwartzberg = 1/sqrt(Polsby-Popper).
    assert abs(avg_schwartzberg - 1.811256) < 1e-4
    

def test_convex_hull__iowa_counties(ia_enacted, ia_dataframe):
    scores = summarize(ia_enacted, [convex_hull()], gdf=ia_dataframe, join_on = "GEOID20")
    avg_ch = sum(scores["convex_hull"].values())/len(scores["convex_hull"])
    # TODO: find an exact source of comparison for this metric.
    # Without boundary clipping, the score should be 0.74379 or so.
    assert abs(avg_ch - 0.77202) < 1e-4
    

def test_polsby_popper__squares():
    grid = Grid((10, 10))
    gdf = gpd.GeoDataFrame([
        {'node': (x, y), 'geometry': box(x, y, x+1, y+1)}
        for (x, y) in grid.graph
    ]).set_index('node')

    gdf.crs = 26918  # UTM zone 18N

    expected_dist_score = pi/4

    scored = polsby_popper().apply(gdf)
    for dist_score in scored.values():
        assert abs(dist_score - expected_dist_score) < 1e-4


def test_schwartzberg__squares():
    grid = Grid((10, 10))
    gdf = gpd.GeoDataFrame([
        {'node': (x, y), 'geometry': box(x, y, x+1, y+1)}
        for (x, y) in grid.graph
    ]).set_index('node')

    gdf.crs = 26918 # UTM zone 18N

    expected_dist_score = sqrt(4/pi)

    scored = schwartzberg().apply(gdf)
    for dist_score in scored.values():
        assert abs(dist_score - expected_dist_score) < 1e-4

def test_reock__iowa_counties(ia_enacted, ia_dataframe):
    scores = summarize(ia_enacted, [reock()], gdf=ia_dataframe, join_on = "GEOID20")
    avg_reock = sum(scores["reock"].values())/len(scores["reock"])

     # Outside source: the value returned by Dave's for the enacted Iowa plan in 0.3825,
    # which is sufficiently close to our value (any differences are likely attributable
    # to resolution).
    # https://davesredistricting.org/maps#ratings::628d5e9a-bd35-4248-aa8c-73af095e0135
    assert abs(avg_reock - 0.38247) < 1e-4

@pytest.mark.skip(reason="Tests should use real-world data.")
def test_reock_score_squares_geodataframe():
    grid = Grid((10, 10))
    gdf = gpd.GeoDataFrame([
        {'node': (x, y), 'geometry': box(x, y, x + 1, y + 1)}
        for (x, y) in grid.graph
    ]).set_index('node')
    score_fn = reock(gdf)

    # The Reock score of a square inscribed in a circle is
    # area(square) / area(circle) = 2/π.
    expected_dist_score = 2 / pi
    scored = score_fn(grid)

    assert scored.keys() == grid.parts.keys()
    for dist_score in scored.values():
        assert abs(dist_score - expected_dist_score) < 1e-4


@pytest.mark.skip(reason="Tests should use real-world data.")
def test_reock_score_squares_graph():
    grid = Grid((10, 10))
    for (x, y), data in grid.graph.nodes(data=True):
        data['geometry'] = box(x, y, x + 1, y + 1)
    score_fn = reock(grid.graph)

    # The Reock score of a square inscribed in a circle is
    # area(square) / area(circle) = 2/π.
    expected_dist_score = 2 /pi
    scored = score_fn(grid)

    assert scored.keys() == grid.parts.keys()
    for dist_score in scored.values():
        assert abs(dist_score - expected_dist_score) < 1e-4


@pytest.mark.skip(reason="Tests should use real-world data.")
def test_reock_score_disconnected():
    grid = Grid((10, 10))
    for (x, y), data in grid.graph.nodes(data=True):
        data['geometry'] = box(x, y, x + 1, y + 1)
    score_fn = reock(grid.graph)

    # Break district 0 into two disconnected pieces
    # (while preserving convex hull perimeter),
    # adding a disconnected component to district 3
    # (quadrupling convex hull perimeter).
    grid_disconnected = grid.flip({(0, 1): 3, (1, 0): 3})

    scored = score_fn(grid_disconnected)
    assert scored.keys() == grid_disconnected.parts.keys()
    assert abs(scored[0] - ((23 / 25) * (2 / pi))) < 1e-4
    assert abs(scored[1] - (2 / pi)) < 1e-4
    assert abs(scored[2] - (2 / pi)) < 1e-4
    assert abs(scored[3] == (27 / 25) * (pi / 2)) < 1e-4


  

if __name__ == "__main__":
    pass
