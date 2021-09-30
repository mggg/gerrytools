
from plan_evaluation.mapping import drawgraph
from gerrychain.graph import Graph
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import os

root = Path(os.getcwd()) / Path("tests/test_resources/")

def test_drawgraph():
    # Read in the graph and the districts.
    graph = Graph.from_json(root / "test-dualgraph.json")
    districts = gpd.read_file(root / "test-districts")

    # Draw it!
    figures, axes = drawgraph(graph, x="x", y="y")

    # Plot districts on the same axes.
    ax = districts.plot(ax=axes[0], column="district")

    plt.show()


if __name__ == "__main__":
    root = Path(os.getcwd()) / Path("test_resources/")
    test_drawgraph()
