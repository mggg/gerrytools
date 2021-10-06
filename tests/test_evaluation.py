
from evaltools.evaluation import (
    deviations, splits, pieces, contiguous, unassigned
)
from evaltools.geography import Partition
from gerrychain.graph import Graph
from pathlib import Path
import os

root = Path(os.getcwd()) / Path("tests/test-resources/")

def test_splits():
    # Read in an existing dual graph.
    dg = Graph.from_json(root / "test-graph.json")
    P = Partition(dg, "CONGRESS")
    units = ["COUNTYFP20"]
    geometricsplits = splits(P, ["COUNTYFP20"])

    # Assert that we have a dictionary and that we have the right keys in it.
    assert type(geometricsplits) is dict
    assert set(units) == set(geometricsplits.keys())

    # Make sure that we're counting the number of splits correctly – Indiana's
    # enacted Congressional plan should split counties 8 times.
    assert geometricsplits["COUNTYFP20"] == 8


def test_pieces():
    # Read in an existing dual graph.
    dg = Graph.from_json(root / "test-graph.json")
    P = Partition(dg, "CONGRESS")
    units = ["COUNTYFP20"]
    geometricpieces = pieces(P, ["COUNTYFP20"])

    # Assert that we have a dictionary and that we have the right keys in it.
    assert type(geometricpieces) is dict
    assert set(units) == set(geometricpieces.keys())
    
    # Make sure that we're counting the number of splits correctly – Indiana's
    # enacted Congressional plan should split counties 8 times.
    assert geometricpieces["COUNTYFP20"] == 16


def test_deviations():
    dg = Graph.from_json(root / "test-graph.json")
    P = Partition(dg, "CONGRESS")
    devs = deviations(P, "TOTPOP")

    assert type(devs) is dict
    assert set(data["CONGRESS"] for _, data in dg.nodes(data=True)) == set(devs.keys())


def test_contiguity():
    dg = Graph.from_json(root / "test-graph.json")
    P = Partition(dg, "CONGRESS")
    contiguity = contiguous(P)

    # This plan should *not* be contiguous, as some VTDs are discontiguous
    # themselves.
    assert not contiguity


def test_unassigned():
    dg = Graph.from_json(root / "test-graph.json")
    P = Partition(dg, "CONGRESS")
    bads = unassigned(P)
    wholebads = unassigned(P, raw=True)

    # This plan shouldn't have any unassigned units.
    assert bads == 0

    # These two things should report the same number, since none are unassigned.
    assert bads == wholebads

if __name__ == "__main__":
    root = Path(os.getcwd()) / Path("test-resources/")
    test_unassigned()
