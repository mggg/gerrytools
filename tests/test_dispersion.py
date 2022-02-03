from gerrychain import (
    GeographicPartition,
    Graph,
    MarkovChain,
    proposals,
    updaters,
    constraints,
    accept,
)
from gerrychain.proposals import recom
from functools import partial

from networkx.algorithms.centrality.dispersion import dispersion
import evaltools.geometry
import geopandas as gpd

def test_dispersion_calc():
    gdf = gpd.read_file("tests/test-resources/test-vtds/test-vtds.shp")
    graph = Graph.from_geodataframe(gdf)

    my_updaters = {
        "population": updaters.Tally("TOTPOP", alias="population"), 
        "dispersion": evaltools.geometry.dispersion_updater_closure(gdf, "CONGRESS", "TOTPOP")
    }

    initial_partition = GeographicPartition(
        graph, assignment="CONGRESS", updaters=my_updaters
    )

    ideal_population = sum(initial_partition["population"].values()) / len(
        initial_partition
    )

    proposal = partial(
        recom, pop_col="TOTPOP", pop_target=ideal_population, epsilon=0.10, node_repeats=2
    )

    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)

    chain = MarkovChain(
        proposal=proposal,
        constraints=[pop_constraint],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=100,
    )

    total_dispersion_over_run = 0
    for count, partition in enumerate(chain):
        if count == 0:
            assert partition["dispersion"] == 0

        total_dispersion_over_run += partition["dispersion"]

    assert total_dispersion_over_run
