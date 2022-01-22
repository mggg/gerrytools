
from evaltools.plotting import (
    drawgraph, drawplan, PlotSpecification, districtr, boxplot
)
from gerrychain.graph import Graph
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import os

root = Path(os.getcwd()) / Path("tests/test-resources/")

def test_PlotSpecification():
    # Read in test stuff.
    districts = gpd.read_file(root / "test-districts")
    counties = gpd.read_file(root / "test-counties")

    # Create a plot specification.
    marion = PlotSpecification()
    marion.context = True
    bbox = marion.computebbox(counties, ["097"], idcolumn="COUNTYFP")
    bbox_adj = marion.computebbox(counties, ["097"], idcolumn="COUNTYFP", margin=2)

    # Assert that the bbox computed is the correct one!
    assert bbox == (-9615282.470860107, 4807354.328610508, -9561225.27038856, 4860676.685822271)

    # Assert that the adjusted bbox actually changes the margins correctly.
    offset = 5280
    assert bbox_adj == (
        -9615282.470860107-offset, 4807354.328610508-offset,
        -9561225.27038856+offset, 4860676.685822271+offset
    )

    # Now get the right districts!
    locs = marion.computelabels(districts, "district")


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

def test_boxplot():
    means = [int(x) for x in np.random.normal(24, 6, size=10)]
    labels = [f"E{i+1}" for i in range(len(means))]
    num_proposed = 3

    scores = {
        "ensemble": [[int(x) for x in np.random.normal(m, 4, size=1000)] for m in means],
        "proposed": [[int(x) for x in np.random.normal(m, 8, size=num_proposed)] for m in means],
    }

    proposed_info = {
        "colors": districtr(num_proposed),
        "names": [f"Plan {i+1}" for i in range(num_proposed)],
    }

    fig, ax = plt.subplots(figsize=(12,6))
    ax = boxplot(ax, scores, labels, proposed_info, percentiles=(25,75))
    plt.show()


if __name__ == "__main__":
    root = Path(os.getcwd()) / Path("test-resources/")
    test_boxplot()

