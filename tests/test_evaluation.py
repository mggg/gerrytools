
from evaltools.evaluation import deviations, splits
from gerrychain.graph import Graph
from pathlib import Path
import os

root = Path(os.getcwd()) / Path("tests/test-resources/")

def test_splits():
    # Read in an existing dual graph.
    dg = Graph.from_json(root / "test-graph.json")
    units = ["COUNTYFP20"]
    geometricsplits = splits(dg, "CONGRESS", ["COUNTYFP20"])

    # Assert that we have a dictionary and that we have the right keys in it.
    assert type(geometricsplits) is dict
    assert set(units) == set(geometricsplits.keys())

    # Make sure that we're counting the number of splits correctly – Indiana's
    # enacted Congressional plan should split counties 8 times.
    assert geometricsplits["COUNTYFP20"] == 8


def test_deviations():
    dg = Graph.from_json(root / "test-graph.json")
    devs = deviations(dg, "CONGRESS", "TOTPOP")

    assert type(devs) is dict
    assert set(data["CONGRESS"] for _, data in dg.nodes(data=True)) == set(devs.keys())

if __name__ == "__main__":
    root = Path(os.getcwd()) / Path("test-resources/")
    test_deviations()
