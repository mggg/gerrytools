import errno
import gc
import os
import stat
import subprocess
import sys
import warnings
from io import BytesIO
from pathlib import Path
from typing import cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from binary_ensemble import BenDecoder
from geopandas.testing import assert_geodataframe_equal
from gerrychain import Graph, MarkovChain, Partition
from gerrychain.partition import GeographicPartition
from gerrychain.partition.assignment import Assignment

import gerrytools.ben as ben
from gerrytools.ben import (
    BendlDecoder,
    BendlEncoder,
    RecordedChain,
    RecordedRun,
    read_geoparquet_asset,
    read_parquet_asset,
)


def test_public_type_names_identify_their_recording_scope() -> None:
    assert "RecordedChainIterator" in ben.__all__
    assert "EncodingVariant" in ben.__all__
    assert not hasattr(ben, "RunIterator")
    assert not hasattr(ben, "Variant")


def configured_chain(
    tmp_path: Path,
    *,
    total_steps: int = 3,
    graph_order=None,
    output_name: str = "ensemble.bendl",
    partition_type=Partition,
    use_default_updaters: bool = True,
    **kwargs,
) -> RecordedChain:
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, {0: 0, 1: 0, 2: 1, 3: 1}, "district")
    chain = RecordedChain(
        graph,
        output_path=tmp_path / output_name,
        graph_order=graph_order,
        total_steps=total_steps,
        rng=1,
        **kwargs,
    )
    chain.initial_partition = partition_type(
        chain.graph,
        "district",
        use_default_updaters=use_default_updaters,
    )
    chain.proposal_fn = lambda partition, *, rng: partition.flip({0: 1})
    return chain


def test_parquet_asset_helpers_read_dataframes(tmp_path):
    table = pd.DataFrame({"GEOID": ["001", "002"], "population": [10, 20]})
    geodata = gpd.GeoDataFrame(
        {"GEOID": ["001", "002"]},
        geometry=gpd.points_from_xy([-105.0, -104.5], [39.5, 40.0]),
        crs="EPSG:4326",
    )
    path = tmp_path / "assets.bendl"
    table_buffer = BytesIO()
    geodata_buffer = BytesIO()
    table.to_parquet(table_buffer, index=False)
    geodata.to_parquet(geodata_buffer, index=False)

    with BendlEncoder(path) as encoder:
        encoder.add_asset("table.parquet", table_buffer.getvalue(), "binary")
        encoder.add_asset("units.parquet", geodata_buffer.getvalue(), "binary")

    decoder = BendlDecoder(path)
    decoder.verify()
    table_result = read_parquet_asset(decoder, "table.parquet")
    geodata_result = read_geoparquet_asset(decoder, "units.parquet")

    pd.testing.assert_frame_equal(table_result, table)
    assert_geodataframe_equal(geodata_result, geodata)


@pytest.mark.parametrize("order", ["mlc", "rcm", "key", None])
def test_graph_order_modes_round_trip(tmp_path, order):
    graph = nx.path_graph([10, 20, 30, 40])
    nx.set_node_attributes(graph, {node: node for node in graph}, "order")
    nx.set_node_attributes(graph, {10: 0, 20: 0, 30: 1, 40: 1}, "district")
    chain = RecordedChain(
        graph,
        output_path=tmp_path / f"{order}.bendl",
        graph_order=order,
        graph_order_key="order" if order == "key" else None,
        total_steps=1,
    )
    chain.initial_partition = Partition(chain.graph, "district")
    chain.proposal_fn = lambda partition, *, rng: partition

    assert list(chain) == [chain.initial_partition]
    decoded_graph = chain.recording.decoder.read_graph()
    assert decoded_graph is not None
    assert list(decoded_graph.nodes) == list(chain.graph.nodes)
    assert (chain.recording.decoder.read_node_permutation_map() is not None) == (order is not None)


@pytest.mark.parametrize("order", ["mlc", "key"])
def test_partition_reconstruction_matches_live_run_under_reordering(tmp_path, order):
    # The graph's insertion order deliberately differs from its path structure and from the
    # sort key, so the requested reordering actually permutes the nodes before encoding.
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 2), (2, 3), (3, 1)])  # path 0-2-3-1
    nx.set_node_attributes(graph, {0: 3, 1: 2, 2: 1, 3: 0}, "sortkey")
    nx.set_node_attributes(graph, {0: 0, 2: 0, 3: 1, 1: 1}, "district")
    chain = RecordedChain(
        graph,
        output_path=tmp_path / f"reordered_{order}.bendl",
        graph_order=order,
        graph_order_key="sortkey" if order == "key" else None,
        total_steps=3,
        rng=1,
    )
    chain.initial_partition = Partition(chain.graph, "district")
    chain.proposal_fn = lambda partition, *, rng: partition.flip({2: 1})

    live_vectors = [partition.assignment_vector.tolist() for partition in chain]

    if order == "key":
        # Reordering relabels nodes 0..n-1 sorted by the key; insertion order had the
        # sortkeys reversed, so a real permutation separates recorded from source order.
        assert [chain.graph.nodes[node]["sortkey"] for node in chain.graph.nodes] == [0, 1, 2, 3]
    reconstructed = [
        chain.recording.partition_at(index).assignment_vector.tolist()
        for index in range(len(live_vectors))
    ]
    assert reconstructed == live_vectors
    decoded_graph = chain.recording.decoder.read_graph()
    assert decoded_graph is not None
    assert list(decoded_graph.nodes) == list(chain.graph.nodes)


def test_no_order_uses_json_canonical_values(tmp_path):
    graph = nx.path_graph(2)
    graph.graph["nested"] = {1: (2, 3)}
    graph.nodes[0]["district"] = 0
    graph.nodes[1]["district"] = 1

    chain = RecordedChain(
        graph,
        output_path=tmp_path / "canonical.bendl",
        graph_order=None,
        total_steps=1,
    )

    assert chain.graph.graph["nested"] == {"1": [2, 3]}


def test_numpy_scalars_are_owned_and_normalized(tmp_path):
    graph = nx.path_graph(2)
    graph.graph["value"] = {"integer": np.int64(2)}
    graph.nodes[0].update(district=np.int64(0), nested=[np.float64(1.5)])
    graph.nodes[1].update(district=np.int64(1), flag=np.bool_(True))

    chain = RecordedChain(
        graph,
        output_path=tmp_path / "numpy.bendl",
        graph_order=None,
        total_steps=1,
    )

    assert isinstance(graph.graph["value"]["integer"], np.integer)
    assert chain.graph.graph["value"] == {"integer": 2}
    assert chain.graph.nodes[0]["nested"] == [1.5]
    assert chain.graph.nodes[1]["flag"] is True


def test_numpy_array_fails_during_preparation(tmp_path):
    graph = nx.path_graph(2)
    graph.nodes[0]["array"] = np.array([1, 2])

    with pytest.raises(TypeError, match="JSON serializable") as error:
        RecordedChain(graph, output_path=tmp_path / "bad.bendl", graph_order=None)

    assert any("no-order canonicalization" in note for note in error.value.__notes__)


@pytest.mark.parametrize(
    ("attribute_owner", "attribute_name"),
    [("node", "__networkx_node__"), ("node", "id"), ("edge", "id")],
)
def test_rejects_reserved_graph_attributes(tmp_path, attribute_owner, attribute_name):
    graph = nx.path_graph(2)
    attributes = graph.nodes[0] if attribute_owner == "node" else graph.edges[0, 1]
    attributes[attribute_name] = 1 if attribute_name == "__networkx_node__" else 0

    with pytest.raises(ValueError, match="reserved"):
        RecordedChain(graph, output_path=tmp_path / "reserved.bendl")


def test_reuses_graph_after_partition_conversion(tmp_path):
    first = configured_chain(tmp_path, output_name="first.bendl", total_steps=1)
    Partition(first.graph, "district")

    second = RecordedChain(
        first.graph,
        output_path=tmp_path / "second.bendl",
        graph_order=None,
        total_steps=1,
    )

    assert all(
        "__networkx_node__" not in attributes for _, attributes in second.graph.nodes(data=True)
    )


def test_accepts_networkx_backed_gerrychain_graph(tmp_path):
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, {0: 0, 1: 0, 2: 1, 3: 1}, "district")
    chain = RecordedChain(
        Graph.from_networkx(graph),
        output_path=tmp_path / "gerrychain-graph.bendl",
        graph_order=None,
        total_steps=1,
    )
    chain.initial_partition = Partition(chain.graph, "district")
    chain.proposal_fn = lambda partition, *, rng: partition

    assert list(chain) == [chain.initial_partition]
    assert chain.recording.decoder.read_graph() is not None


@pytest.mark.parametrize("graph", [nx.DiGraph([(0, 1)]), nx.MultiGraph([(0, 1)])])
def test_rejects_directed_and_multigraphs(tmp_path, graph):
    with pytest.raises(TypeError, match="simple undirected"):
        RecordedChain(graph, output_path=tmp_path / "unsupported.bendl")


def test_rejects_rustworkx_graphs(tmp_path):
    rx_graph = Graph.from_networkx(nx.path_graph(2)).convert_from_nx_to_rx()
    with pytest.raises(TypeError, match="NetworkX-backed"):
        RecordedChain(rx_graph, output_path=tmp_path / "rx.bendl")


def test_rejects_frozen_gerrychain_graphs(tmp_path):
    graph = nx.path_graph(2)
    nx.set_node_attributes(graph, {0: 0, 1: 1}, "district")
    frozen_graph = cast(Graph, Partition(graph, "district").graph)
    with pytest.raises(TypeError, match="frozen"):
        RecordedChain(frozen_graph, output_path=tmp_path / "frozen.bendl")


def test_recording_options_are_owned_and_read_only(tmp_path):
    metadata = {"nested": [1]}
    chain = configured_chain(tmp_path, metadata=metadata)
    metadata["nested"].append(2)
    retrieved = cast(dict, chain.metadata)
    retrieved["nested"].append(3)

    assert chain.metadata == {"nested": [1]}
    assert nx.is_frozen(chain.graph)
    with pytest.raises(AttributeError):
        setattr(chain, "output_path", tmp_path / "other.bendl")
    with pytest.raises(nx.NetworkXError):
        chain.graph.add_node(10)


def test_records_live_partitions_metadata_and_vectors(tmp_path):
    chain = configured_chain(tmp_path, metadata={"seed": 1, "notes": ["original"]})

    yielded = list(chain)
    expected = [partition.assignment_vector.tolist() for partition in yielded]
    recording = chain.recording
    metadata = cast(dict, recording.metadata)
    metadata["notes"].append("changed")

    assert yielded[0] is chain.initial_partition
    assert recording.count_samples() == 3
    assert len(recording) == 3
    assert recording.metadata == {"seed": 1, "notes": ["original"]}
    assert list(recording) == expected
    assert list(recording) == expected
    assert recording.verify() is None


def test_numpy_scalar_metadata_is_normalized_and_records(tmp_path):
    # The docstring recommends putting epsilon/seed in metadata, and those are often NumPy
    # scalars; they must be absorbed at construction, not explode inside the run.
    chain = configured_chain(
        tmp_path,
        metadata={"seed": np.int64(1), "epsilon": np.float64(0.01), "vintages": (2010, 2020)},
    )

    list(chain)

    expected = {"seed": 1, "epsilon": 0.01, "vintages": [2010, 2020]}
    assert chain.metadata == expected
    assert chain.recording.decoder.read_metadata() == expected


def test_unserializable_metadata_fails_at_construction(tmp_path):
    with pytest.raises(TypeError, match="JSON-serializable"):
        configured_chain(tmp_path, metadata={"nodes": {0, 1}})
    assert list(tmp_path.iterdir()) == []


def test_list_metadata_round_trips(tmp_path):
    chain = configured_chain(tmp_path, metadata=["recom", {"epsilon": 0.01}])

    list(chain)

    assert chain.recording.decoder.read_metadata() == ["recom", {"epsilon": 0.01}]


def test_none_metadata_omits_asset(tmp_path):
    chain = configured_chain(tmp_path, total_steps=1)

    list(chain)

    assert chain.recording.decoder.read_metadata() is None
    assert "metadata.json" not in chain.recording.decoder.asset_names()


@pytest.mark.parametrize("variant", ["standard", "mkv_chain", "twodelta"])
def test_public_variants(tmp_path, variant):
    chain = configured_chain(tmp_path, variant=variant)

    live_vectors = [partition.assignment_vector.tolist() for partition in chain]

    assert len(chain.recording.decoder) == 3
    # Every variant must decode back to the exact recorded assignments, not just the count.
    assert list(chain.recording.decoder) == live_vectors


def test_invalid_variant_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="variant"):
        configured_chain(tmp_path, variant="alias", output_name="invalid.bendl")
    assert list(tmp_path.iterdir()) == []


def test_invalid_graph_order_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="graph_order"):
        configured_chain(tmp_path, graph_order="zigzag", output_name="invalid.bendl")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("graph_order", "graph_order_key"),
    [(None, "order"), ("key", None)],
)
def test_graph_order_key_is_present_exactly_for_key_order(tmp_path, graph_order, graph_order_key):
    with pytest.raises(ValueError, match="required only"):
        configured_chain(
            tmp_path,
            graph_order=graph_order,
            graph_order_key=graph_order_key,
            output_name="invalid.bendl",
        )

    assert list(tmp_path.iterdir()) == []


def test_missing_parent_creates_nothing(tmp_path):
    missing = configured_chain(tmp_path, output_name="missing/ensemble.bendl")
    with pytest.raises(FileNotFoundError):
        next(iter(missing))
    assert not (tmp_path / "missing").exists()


def test_initial_partition_must_use_prepared_graph(tmp_path):
    original = nx.path_graph(4)
    nx.set_node_attributes(original, {0: 0, 1: 0, 2: 1, 3: 1}, "district")
    chain = RecordedChain(
        original,
        output_path=tmp_path / "wrong.bendl",
        graph_order=None,
        total_steps=1,
    )
    chain.initial_partition = Partition(original, "district")
    chain.proposal_fn = lambda partition, *, rng: partition

    with pytest.raises(RuntimeError, match="not built from RecordedChain.graph"):
        next(iter(chain))
    assert not chain.output_path.exists()


def test_initial_partition_rejects_foreign_graph_after_prepared_graph_was_converted(tmp_path):
    original = nx.path_graph(4)
    nx.set_node_attributes(original, {0: 0, 1: 0, 2: 1, 3: 1}, "district")
    chain = RecordedChain(
        original,
        output_path=tmp_path / "wrong-after-conversion.bendl",
        graph_order=None,
        total_steps=1,
    )
    Partition(chain.graph, "district")
    chain.initial_partition = Partition(original, "district")
    chain.proposal_fn = lambda partition, *, rng: partition

    with pytest.raises(RuntimeError, match="not built from RecordedChain.graph"):
        next(iter(chain))

    assert not chain.output_path.exists()


def test_initial_partition_rejects_prepared_nodes_in_a_different_order(tmp_path):
    chain = configured_chain(tmp_path, total_steps=1)
    reordered = nx.Graph()
    reordered.add_nodes_from(
        (node, dict(chain.graph.nodes[node])) for node in reversed(list(chain.graph.nodes))
    )
    reordered.add_edges_from(chain.graph.edges)
    chain.initial_partition = Partition(reordered, "district")
    chain.proposal_fn = lambda partition, *, rng: partition

    with pytest.raises(RuntimeError, match="node order does not match RecordedChain.graph"):
        next(iter(chain))

    assert not chain.output_path.exists()


def test_graph_mutation_during_run_prevents_publication(tmp_path):
    chain = configured_chain(tmp_path, total_steps=2)
    run = iter(chain)
    next(run)
    chain.graph.nodes[0]["changed"] = True
    next(run)

    with pytest.raises(RuntimeError, match="final partition execution graph") as error:
        next(run)

    assert not chain.output_path.exists()
    assert "finalized BENDL" in error.value.__notes__[0]


def test_proposal_edge_scratch_attributes_do_not_prevent_publication(tmp_path):
    # GerryChain's ReCom writes random spanning-tree weights into edge dicts shared with
    # chain.graph while the chain runs; edge-attribute scratch must not fail the run or a rerun.
    chain = configured_chain(tmp_path, total_steps=2)

    def proposal(partition, *, rng):
        chain.graph.edges[0, 1]["random_weight"] = 0.5
        return partition.flip({0: 1})

    chain.proposal_fn = proposal

    assert len(list(chain)) == 2
    assert chain.output_path.exists()
    assert len(list(chain.allow_overwrite())) == 2


def test_proposal_cannot_mutate_original_edge_attributes(tmp_path):
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, {0: 0, 1: 0, 2: 1, 3: 1}, "district")
    nx.set_edge_attributes(graph, {edge: 1.0 for edge in graph.edges}, "weight")
    chain = RecordedChain(
        graph,
        output_path=tmp_path / "ensemble.bendl",
        graph_order=None,
        total_steps=2,
        rng=1,
    )
    chain.initial_partition = Partition(chain.graph, "district")

    def proposal(partition, *, rng):
        chain.graph.edges[0, 1]["weight"] = 999.0
        return partition.flip({0: 1})

    chain.proposal_fn = proposal

    with pytest.raises(RuntimeError, match="final partition execution graph"):
        list(chain)

    assert not chain.output_path.exists()


def test_direct_rustworkx_mutation_during_run_prevents_publication(tmp_path):
    chain = configured_chain(tmp_path, total_steps=1)
    run = iter(chain)
    next(run)
    assert chain.initial_partition is not None
    chain.initial_partition.graph.graph.get_rx_graph().add_node({})

    with pytest.raises(RuntimeError, match="contiguous internal node ids"):
        next(run)

    assert not chain.output_path.exists()


def test_assignment_error_names_sample_and_preserves_recording(tmp_path):
    chain = configured_chain(tmp_path, total_steps=1)
    chain.initial_partition = Partition(chain.graph, {0: 70_000, 1: 0, 2: 1, 3: 1})
    run = iter(chain)

    with pytest.raises((TypeError, ValueError, OverflowError)) as error:
        next(run)

    assert any("sample 0" in note for note in error.value.__notes__)
    assert any("unfinalized BENDL" in note for note in error.value.__notes__)


def test_midstream_error_preserves_recoverable_prefix(tmp_path):
    chain = configured_chain(tmp_path, total_steps=2)

    def fail(_partition, *, rng):
        raise RuntimeError("injected midstream failure")

    chain.proposal_fn = fail

    with pytest.raises(RuntimeError, match="injected midstream failure") as error:
        list(chain)

    preservation_note = next(note for note in error.value.__notes__ if "unfinalized BENDL" in note)
    preserved = Path(preservation_note.rsplit(" at ", 1)[1])
    recovered = tmp_path / "recovered-prefix.ben"
    BendlDecoder(preserved).extract_stream(recovered, allow_unfinalized=True)

    assert chain.initial_partition is not None
    assert list(BenDecoder(recovered)) == [chain.initial_partition.assignment_vector.tolist()]


def test_unstarted_iterators_do_not_reserve_chain(tmp_path):
    chain = configured_chain(tmp_path, total_steps=1)
    unused = iter(chain)
    unused.close()
    first = iter(chain)
    second = iter(chain)

    next(first)
    with pytest.raises(RuntimeError, match="already active"):
        next(second)
    with pytest.warns(RuntimeWarning, match="Incomplete RecordedChain"):
        first.close()

    list(chain)


def test_early_close_preserves_recoverable_stream_without_assets(tmp_path):
    chain = configured_chain(tmp_path, total_steps=2)
    run = iter(chain)
    first = next(run)

    with pytest.warns(RuntimeWarning) as warning:
        run.close()

    preserved = Path(str(warning[0].message).split(" at ", 1)[1].split(";", 1)[0])
    decoder = BendlDecoder(preserved)
    recovered = tmp_path / "recovered.ben"
    decoder.extract_stream(recovered, allow_unfinalized=True)

    assert not decoder.is_complete()
    assert decoder.asset_names() == []
    assert list(BenDecoder(recovered)) == [first.assignment_vector.tolist()]


def test_parent_close_failure_does_not_annotate_foreign_exception(tmp_path, monkeypatch):
    chain = configured_chain(tmp_path, total_steps=1)
    original_iter = MarkovChain.__iter__

    class CloseFails:
        def __init__(self, recorded_chain):
            self._iterator = original_iter(recorded_chain)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._iterator)

        def close(self):
            raise RuntimeError("parent close failed")

    monkeypatch.setattr(MarkovChain, "__iter__", lambda recorded_chain: CloseFails(recorded_chain))

    foreign = RuntimeError("foreign")
    try:
        raise foreign
    except RuntimeError:
        with pytest.warns(RuntimeWarning, match="published successfully"):
            assert len(list(chain)) == 1

    assert getattr(foreign, "__notes__", []) == []


def test_assignment_and_partition_access_are_zero_based_and_preserve_updaters(tmp_path):
    updater = lambda partition: len(partition.assignment)
    chain = configured_chain(tmp_path)
    assert chain.initial_partition is not None
    chain.initial_partition.updaters["size"] = updater
    list(chain)
    sequential = iter(chain.recording.decoder)
    first_from_cursor = next(sequential)

    first_vector = chain.recording.assignment_at(0)
    first = chain.recording.partition_at(0)
    middle = chain.recording.partition_at(1)

    assert first_vector == first_from_cursor
    assert chain.recording.lookup(0) == first_vector
    assert first.assignment_vector.tolist() == first_from_cursor
    assert first["size"] == 4
    assert first.graph is middle.graph
    assert next(sequential) == chain.recording.decoder.lookup(1)
    with pytest.raises(IndexError):
        chain.recording.assignment_at(3)


@pytest.mark.parametrize(
    ("method", "args", "indices"),
    [
        ("subsample_every", (2,), [0, 2]),
        ("subsample_indices", ([0, 2],), [0, 2]),
        ("subsample_range", (0, 2), [0, 1]),
    ],
)
def test_subsampling_vectors_compose_with_partitions(tmp_path, method, args, indices):
    chain = configured_chain(tmp_path)
    recorded = [partition.assignment_vector.tolist() for partition in chain]
    sequential = iter(chain.recording.decoder)
    assert next(sequential) == recorded[0]

    vectors = list(getattr(chain.recording, method)(*args))
    partitions = list(chain.recording.partitions(getattr(chain.recording, method)(*args)))

    assert vectors == [recorded[index] for index in indices]
    assert next(sequential) == recorded[1]
    assert [partition.assignment_vector.tolist() for partition in partitions] == [
        recorded[index] for index in indices
    ]


@pytest.mark.parametrize("indices", [[3, 0], [2, 2, 0]])
def test_subsample_indices_rejects_unsorted_or_duplicate_indices(tmp_path, indices):
    chain = configured_chain(tmp_path, total_steps=4)
    list(chain)

    with pytest.raises(ValueError, match="sorted and unique"):
        chain.recording.subsample_indices(indices)


def test_identical_chain_configs_replay_deterministically(tmp_path):
    # Two runs of the same configuration (same graph, seed, and proposal) must decode to
    # identical assignment vectors, so a recording is a faithful stand-in for a rerun.
    first = configured_chain(tmp_path, output_name="first.bendl")
    second = configured_chain(tmp_path, output_name="second.bendl")

    first_vectors = [partition.assignment_vector.tolist() for partition in first]
    list(second)

    assert list(first.recording.decoder) == first_vectors
    assert list(second.recording.decoder) == first_vectors


def test_wrong_length_vector_raises_the_same_error_on_both_cache_paths(tmp_path):
    # Before the explicit check, the cold path raised a strict-zip ValueError while the warm
    # (cached-graph) path surfaced a gerrychain KeyError for the same mistake.
    chain = configured_chain(tmp_path, total_steps=1)
    list(chain)
    recording = chain.recording

    with pytest.raises(ValueError, match="3 entries.*4 nodes"):
        next(recording.partitions([[0, 0, 1]]))

    recording.partition_at(0)  # prime the cached frozen graph
    with pytest.raises(ValueError, match="3 entries.*4 nodes"):
        next(recording.partitions([[0, 0, 1]]))


def test_recorded_run_without_embedded_graph_raises_for_partitions(tmp_path):
    # Vectors in a graphless bendl stay readable, but partitions cannot be rebuilt.
    path = tmp_path / "graphless.bendl"
    encoder = BendlEncoder(path, overwrite=False)
    with encoder.ben_stream(variant="standard") as stream:
        stream.write([0, 0, 1, 1])

    recorded_run = RecordedRun(path, Partition, {})

    assert recorded_run.lookup(0) == [0, 0, 1, 1]
    with pytest.raises(RuntimeError, match="no embedded graph"):
        recorded_run.partition_at(0)


def test_recorded_run_from_bendl_reopens_published_recording(tmp_path):
    chain = configured_chain(tmp_path, total_steps=2)
    live_vectors = [partition.assignment_vector.tolist() for partition in chain]

    recording = RecordedRun.from_bendl(str(chain.output_path))

    assert list(recording.decoder) == live_vectors
    assert recording.partition_at(1).assignment_vector.tolist() == live_vectors[1]


def test_truncated_finalized_file_raises_on_read(tmp_path):
    chain = configured_chain(tmp_path, total_steps=2)
    list(chain)
    recorded_bytes = chain.output_path.read_bytes()
    truncated_path = tmp_path / "truncated.bendl"
    truncated_path.write_bytes(recorded_bytes[: len(recorded_bytes) // 2])

    recorded_run = RecordedRun(truncated_path, Partition, {})

    with pytest.raises(RuntimeError, match="Could not open recorded BENDL"):
        recorded_run.lookup(0)


def test_partition_at_restores_geographic_class_without_defaults(tmp_path):
    chain = configured_chain(
        tmp_path,
        partition_type=GeographicPartition,
        use_default_updaters=False,
        total_steps=1,
    )
    list(chain)

    restored = chain.recording.partition_at(0)

    assert type(restored) is GeographicPartition
    assert restored.updaters == {}


def test_recording_guards(tmp_path):
    # The read-back API lives on chain.recording, so the property is the one guard: it raises
    # before any successful run and while a run is active.
    chain = configured_chain(tmp_path, total_steps=1)
    with pytest.raises(RuntimeError, match="before"):
        _ = chain.recording

    run = iter(chain)
    next(run)
    with pytest.raises(RuntimeError, match="while"):
        _ = chain.recording
    with pytest.warns(RuntimeWarning):
        run.close()


def test_recording_handle_survives_active_and_failed_reruns(tmp_path):
    chain = configured_chain(tmp_path, total_steps=2)
    list(chain)
    recording = chain.recording

    rerun = chain.allow_overwrite()
    next(rerun)
    # The chain refuses to hand out a reader mid-run, but an active or failed rerun streams
    # only to its temporary bundle: the published file is untouched until a clean publish, so
    # a handle from the previous successful run keeps reading it. Only a rerun that completes
    # and replaces the file invalidates the handle (see the invalidation tests below).
    with pytest.raises(RuntimeError, match="while"):
        _ = chain.recording
    assert len(recording.lookup(0)) == 4
    with pytest.warns(RuntimeWarning):
        rerun.close()

    assert chain.recording is recording
    assert len(recording.lookup(0)) == 4


def test_successful_same_path_rerun_invalidates_old_handle(tmp_path):
    chain = configured_chain(tmp_path, total_steps=3)
    original = [partition.assignment_vector.tolist() for partition in chain]
    stale = chain.recording
    iterator = iter(stale)
    assert next(iterator) == original[0]
    assert len(stale.assignment_at(0)) == 4  # prime the cached decoder before the rerun

    chain.total_steps = 2
    list(chain.allow_overwrite())

    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        stale.lookup(0)
    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        stale.assignment_at(0)
    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        stale.count_samples()
    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        len(stale)
    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        _ = stale.metadata
    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        stale.verify()
    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        iter(stale)
    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        next(iterator)
    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        stale.subsample_every(1)
    assert stale.path == chain.output_path.resolve()  # path itself stays readable
    assert chain.recording is not stale
    assert len(chain.recording.decoder) == 2


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("subsample_every", (1,)),
        ("subsample_indices", ([0, 1],)),
        ("subsample_range", (0, 2)),
    ],
)
def test_active_subsample_iterator_stops_after_overwrite(tmp_path, method, args):
    chain = configured_chain(tmp_path, total_steps=3)
    original = [partition.assignment_vector.tolist() for partition in chain]
    stale = chain.recording
    iterator = getattr(stale, method)(*args)
    assert next(iterator) == original[0]

    chain.total_steps = 2
    list(chain.allow_overwrite())

    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        next(iterator)


def test_active_partition_iterator_stops_after_overwrite(tmp_path):
    chain = configured_chain(tmp_path, total_steps=3)
    assignments = [partition.assignment_vector.tolist() for partition in chain]
    stale = chain.recording
    iterator = stale.partitions(assignments)
    assert next(iterator).assignment_vector.tolist() == assignments[0]

    chain.total_steps = 2
    list(chain.allow_overwrite())

    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        next(iterator)


def test_rerun_to_different_resolved_path_keeps_old_handle(tmp_path):
    # The output path routes through a symlinked directory; retargeting the symlink between
    # runs makes the authorized rerun publish to a different file, so the handle from the
    # first run keeps reading its original data.
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    (tmp_path / "current").symlink_to(tmp_path / "first")
    chain = configured_chain(tmp_path, total_steps=3, output_name="current/ensemble.bendl")
    original = [partition.assignment_vector.tolist() for partition in chain]
    old_handle = chain.recording

    (tmp_path / "current").unlink()
    (tmp_path / "current").symlink_to(tmp_path / "second")
    chain.total_steps = 2
    list(chain.allow_overwrite())

    assert old_handle.path == (tmp_path / "first" / "ensemble.bendl").resolve()
    assert old_handle.lookup(2) == original[2]
    assert list(old_handle.subsample_every(1)) == original
    assert len(chain.recording.decoder) == 2
    assert chain.recording.path == (tmp_path / "second" / "ensemble.bendl").resolve()


def test_explicit_overwrite_is_one_use_and_preserves_mode(tmp_path):
    chain = configured_chain(tmp_path, total_steps=1)
    list(chain)
    old_decoder = chain.recording.decoder
    os.chmod(chain.output_path, 0o640)

    with pytest.raises(RuntimeError, match="allow_overwrite"):
        next(iter(chain))
    list(chain.allow_overwrite())

    assert stat.S_IMODE(chain.output_path.stat().st_mode) == 0o640
    assert chain.recording.decoder is not old_decoder
    with pytest.raises(RuntimeError, match="allow_overwrite"):
        next(iter(chain))


def test_authorized_first_run_replaces_existing_regular_file(tmp_path):
    chain = configured_chain(tmp_path, total_steps=1)
    chain.output_path.write_text("external")
    os.chmod(chain.output_path, 0o600)

    with pytest.raises(FileExistsError, match="allow_overwrite"):
        next(iter(chain))
    list(chain.allow_overwrite())

    assert stat.S_IMODE(chain.output_path.stat().st_mode) == 0o600
    assert len(chain.recording.decoder) == 1


def test_authorized_overwrite_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("external")
    chain = configured_chain(tmp_path, total_steps=1)
    chain.output_path.symlink_to(target)

    with pytest.raises(OSError, match="regular file") as error:
        list(chain.allow_overwrite())

    assert chain.output_path.is_symlink()
    assert target.read_text() == "external"
    assert any("finalized BENDL" in note for note in error.value.__notes__)


def test_authorized_overwrite_rejects_directory(tmp_path):
    chain = configured_chain(tmp_path, total_steps=1)
    chain.output_path.mkdir()

    with pytest.raises(OSError, match="regular file") as error:
        list(chain.allow_overwrite())

    assert chain.output_path.is_dir()
    assert any("finalized BENDL" in note for note in error.value.__notes__)


def test_first_publication_does_not_clobber_racing_destination(tmp_path, monkeypatch):
    chain = configured_chain(tmp_path, total_steps=1)
    real_link = os.link

    def racing_link(source, destination):
        Path(destination).write_text("racing writer")
        return real_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)

    with pytest.raises(FileExistsError) as error:
        list(chain)

    assert chain.output_path.read_text() == "racing writer"
    assert any("finalized BENDL" in note for note in error.value.__notes__)


def test_publication_fails_safely_when_hard_links_are_unsupported(tmp_path, monkeypatch):
    chain = configured_chain(tmp_path, total_steps=2)

    def unsupported_link(source, destination, **kwargs):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", unsupported_link)

    with pytest.raises(OSError, match="requires hard-link support") as error:
        list(chain)

    assert not chain.output_path.exists()
    assert any("finalized BENDL" in note for note in error.value.__notes__)


def test_unsupported_hard_link_preserves_racing_destination(tmp_path, monkeypatch):
    chain = configured_chain(tmp_path, total_steps=1)

    def racing_unsupported_link(source, destination, **kwargs):
        Path(destination).write_text("racing writer")
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", racing_unsupported_link)

    with pytest.raises(OSError, match="requires hard-link support") as error:
        list(chain)

    assert chain.output_path.read_text() == "racing writer"
    assert any("finalized BENDL" in note for note in error.value.__notes__)


def test_unsupported_hard_link_never_attempts_nonatomic_replace(tmp_path, monkeypatch):
    chain = configured_chain(tmp_path, total_steps=1)
    replace_called = False

    def unsupported_link(source, destination, **kwargs):
        raise OSError(errno.EPERM, "Operation not permitted")

    def racing_replace(source, destination):
        nonlocal replace_called
        replace_called = True

    monkeypatch.setattr(os, "link", unsupported_link)
    monkeypatch.setattr(os, "replace", racing_replace)

    with pytest.raises(OSError, match="requires hard-link support"):
        list(chain)

    assert not replace_called
    assert not chain.output_path.exists()


def test_cleanup_warning_cannot_undo_committed_run(tmp_path, monkeypatch):
    chain = configured_chain(tmp_path, total_steps=1)
    real_rmdir = Path.rmdir

    def fail_private_rmdir(path):
        if path.name.startswith(".ensemble.bendl."):
            raise OSError("cleanup failed")
        return real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", fail_private_rmdir)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        list(chain)

    assert len(chain.recording.decoder) == 1


def test_hardlink_alias_cleanup_failure_keeps_run_committed(tmp_path, monkeypatch):
    chain = configured_chain(tmp_path, total_steps=1)
    real_unlink = Path.unlink

    def fail_recording_unlink(path, *args, **kwargs):
        if path.name == "recording.bendl":
            raise OSError("unlink failed")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_recording_unlink)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(chain)

    assert any("writable alias" in str(item.message) for item in caught)
    assert len(chain.recording.decoder) == 1


def test_failed_rerun_keeps_previous_recording_and_schema(tmp_path):
    chain = configured_chain(tmp_path, total_steps=1)
    list(chain)
    previous_bytes = chain.output_path.read_bytes()
    previous_decoder = chain.recording.decoder
    previous_type = type(chain.recording.partition_at(0))
    chain.proposal_fn = lambda partition, *, rng: (_ for _ in ()).throw(RuntimeError("boom"))
    chain.total_steps = 2

    run = chain.allow_overwrite()
    next(run)
    with pytest.raises(RuntimeError, match="boom"):
        next(run)

    assert chain.output_path.read_bytes() == previous_bytes
    assert chain.recording.decoder is previous_decoder
    assert type(chain.recording.partition_at(0)) is previous_type


def test_failed_publish_revokes_stale_handle_invalidation(tmp_path, monkeypatch):
    # Regression: the old handle was invalidated just before publish and stayed invalidated
    # when publish itself failed, even though its file survived untouched.
    from gerrytools.ben.recorded_chain import _BendlTransaction

    chain = configured_chain(tmp_path, total_steps=2)
    original = [partition.assignment_vector.tolist() for partition in chain]
    handle = chain.recording

    def failing_publish(self, overwrite):
        raise OSError("disk full")

    monkeypatch.setattr(_BendlTransaction, "publish", failing_publish)
    with pytest.raises(OSError, match="disk full"):
        list(chain.allow_overwrite())

    # A failed publish never touches the destination, so the handle reads its original data.
    assert handle.lookup(0) == original[0]
    assert list(handle.subsample_every(1)) == original
    assert chain.recording is handle

    monkeypatch.undo()
    list(chain.allow_overwrite())
    with pytest.raises(RuntimeError, match="later rerun overwrote"):
        handle.lookup(0)
    assert chain.recording is not handle


def test_transient_foreign_graph_partition_fails_at_its_step(tmp_path):
    # Regression: only the initial and final partitions were verified, so a proposal emitting
    # a transient partition on a same-nodes, different-order copy of the graph published a
    # permuted assignment vector for that step. The per-step identity check fails at the step.
    chain = configured_chain(tmp_path, total_steps=3)
    foreign = nx.Graph()
    foreign.add_nodes_from(reversed(list(chain.graph.nodes)))
    foreign.add_edges_from(chain.graph.edges)
    for node in foreign.nodes:
        foreign.nodes[node]["district"] = chain.graph.nodes[node]["district"]
    foreign_partition = Partition(foreign, "district")
    assert chain.initial_partition is not None
    proposals = iter([foreign_partition, chain.initial_partition.flip({0: 1})])
    chain.proposal_fn = lambda partition, *, rng: next(proposals)

    run = iter(chain)
    next(run)
    with pytest.raises(RuntimeError, match="sample 1.*initial partition's graph"):
        next(run)

    assert not chain.output_path.exists()


def test_transient_partition_subclass_fails_at_its_step(tmp_path):
    class AlternatePartition(Partition):
        pass

    chain = configured_chain(tmp_path, total_steps=2)
    assert chain.initial_partition is not None
    alternate = AlternatePartition(chain.initial_partition.graph, "district")
    chain.proposal_fn = lambda partition, *, rng: alternate

    run = iter(chain)
    next(run)
    with pytest.raises(RuntimeError, match="sample 1 partition class AlternatePartition"):
        next(run)

    assert not chain.output_path.exists()


def test_completed_recording_requires_explicit_rerun_after_external_delete(tmp_path):
    chain = configured_chain(tmp_path, total_steps=1)
    list(chain)
    chain.output_path.unlink()

    with pytest.raises(RuntimeError, match="allow_overwrite"):
        next(iter(chain))


def test_partition_at_rejects_incompatible_partition_subclass(tmp_path):
    class NeedsArgument(Partition):
        def __init__(self, graph=None, assignment=None, *, required, **kwargs):
            super().__init__(graph, assignment, **kwargs)
            self.required = required

    graph = nx.path_graph(2)
    chain = RecordedChain(
        graph,
        output_path=tmp_path / "custom.bendl",
        graph_order=None,
        total_steps=1,
    )
    chain.initial_partition = NeedsArgument(chain.graph, {0: 0, 1: 1}, required=True)
    chain.proposal_fn = lambda partition, *, rng: partition
    list(chain)

    with pytest.raises(TypeError, match="Cannot reconstruct"):
        chain.recording.partition_at(0)


def test_upstream_nonempty_flip_child_copies_cached_vector(monkeypatch):
    root = Partition(nx.path_graph(2), {0: 0, 1: 1})
    assert root.assignment_vector.tolist() == [0, 1]
    child = root.flip({0: 1})

    monkeypatch.setattr(
        Assignment,
        "to_vector",
        lambda self: (_ for _ in ()).throw(AssertionError("rebuilt vector")),
    )

    assert child.assignment_vector.tolist() == [1, 1]


def test_abandoned_iterator_releases_lock_when_collected(tmp_path):
    chain = configured_chain(tmp_path, total_steps=2)
    run = iter(chain)
    next(run)
    with pytest.warns(RuntimeWarning):
        del run
        gc.collect()

    list(chain)


def test_recorded_chain_import_does_not_require_docker():
    code = """
import builtins
import sys

real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "docker" or name.startswith("docker."):
        raise ModuleNotFoundError("No module named 'docker'", name="docker")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked
from gerrytools.ben import RecordedChain
assert RecordedChain.__name__ == "RecordedChain"
"""
    subprocess.run([sys.executable, "-c", code], check=True)
