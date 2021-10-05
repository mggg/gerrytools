
from evaltools.mapping import drawgraph, drawplan
from gerrychain.graph import Graph
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import os

root = Path(os.getcwd()) / Path("tests/test-resources/")

def test_drawgraph():
    # Read in the graph and the districts.
    graph = Graph.from_json(root / "test-graph.json")
    districts = gpd.read_file(root / "test-districts")

    # Plot districts on the same axes.
    ax = districts.plot(column="district")

    # Draw it twice!
    single_axes = drawgraph(graph, ax=ax)
    multiple_axes = drawgraph(graph, components=True)

    # Assert that single_axes is a single object, and that multiple_axes is a
    # list of objects.
    assert type(single_axes) is not list
    assert type(multiple_axes) is list


def test_drawplan():
    # Read in a districting plan.
    districts = gpd.read_file(root / "test-districts")
    districts["district"] = districts["district"] + 1

    # Draw the plan and make sure we get axes back.
    ax = drawplan(districts, assignment="district", numbers=True)
    assert ax is not None


if __name__ == "__main__":
    root = Path(os.getcwd()) / Path("test-resources/")
    test_drawplan()
