
from evaltools.geography import dissolve, dualgraph, unitmap, invert
import geopandas as gpd
from pathlib import Path
import os

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

if __name__ == "__main__":
    root = Path(os.getcwd()) / Path("test-resources/")
