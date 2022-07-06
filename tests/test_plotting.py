
from evaltools.plotting import (
    drawgraph, drawplan, districtr, boxplot
)
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

from .utils import remotegraphresource, remoteresource


def test_drawgraph():
    # Read in the graph and the districts.
    graph = remotegraphresource("test-graph.json")
    districts = gpd.read_file(remoteresource("test-districts.geojson"))

    # Plot districts on the same axes.
    ax = districts.plot(column="DISTRICT")

    # Draw it twice!
    single_axes = drawgraph(graph, ax=ax)
    multiple_axes = drawgraph(graph, components=True)

    # Assert that single_axes is a single object, and that multiple_axes is a
    # list of objects.
    assert type(single_axes) is not list
    assert type(multiple_axes) is list

    plt.close()


def test_drawplan():
    # Read in a districting plan.
    districts = gpd.read_file(remoteresource("test-districts.geojson"))
    districts["DISTRICT"] = districts["DISTRICT"] + 1

    # Draw the plan and make sure we get axes back.
    ax = drawplan(districts, assignment="DISTRICT", numbers=True)
    assert ax is not None

    plt.close()


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
    ax = boxplot(ax, scores, labels, proposed_info=proposed_info, percentiles=(25,75))
    plt.close()


if __name__ == "__main__":
    test_boxplot()
