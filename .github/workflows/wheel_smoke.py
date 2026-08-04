"""Smoke test a freshly built GerryTools wheel in a clean environment."""

import pathlib
import tempfile

import geopandas as gpd
import networkx as nx
import pyogrio
from shapely.geometry import Point

from gerrytools.scoring import PlanEvaluator, Tally


def main() -> None:
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, dict(enumerate([1, 2, 3, 4])), "population")
    result = PlanEvaluator(graph).add_metric(Tally("population")).evaluate([0, 0, 1, 1])
    assert result["population"].tolist() == [3.0, 7.0]

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "tiny.gpkg"
        frame = gpd.GeoDataFrame({"value": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        pyogrio.write_dataframe(frame, path)
        assert pyogrio.read_dataframe(path)["value"].tolist() == [1]

    print("wheel smoke test passed")


if __name__ == "__main__":
    main()
