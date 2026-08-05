"""Resource-owning interface for engine-backed plan scoring."""

import json
import math
import numbers
import os
import tempfile
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast
from weakref import WeakSet

import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from gerrychain import Graph as GerryGraph
from gerrychain import Partition
from pyproj import CRS
from tqdm.auto import tqdm

from gerrytools import _scoring_engine
from gerrytools._geodataframe import (
    _alignment_positions,
    _object_index,
    _validated_geometry_frame,
)

from ._manifest import find_manifest_path, preferred_manifest_path
from ._types import (
    EvaluationSummary,
    _MetricResult,
    _ResultShape,
    _StreamRunOptions,
    is_valid_metric_name,
)
from .metrics import Metric, _merged_keys, _MetricBase, _OutputSpec, _ResourceSpec
from .result import (
    EnsembleEvalResult,
    ManyPlanEvalResult,
    PlanEvalResult,
)

_MAX_DISTRICTS = _scoring_engine.MAX_DISTRICTS
Assignment: TypeAlias = Mapping[Any, Any] | Iterable[Any]
"""District assignment accepted by scoring: a node-to-district mapping, or district labels in
graph-node order. A pandas Series is treated as a labeled mapping keyed by its index."""


def _assignment_mapping(assignment: Assignment) -> Mapping[Any, Any] | None:
    """Return labeled assignments as mappings, preserving a Series index."""
    if isinstance(assignment, pd.Series):
        if not assignment.index.is_unique:
            raise ValueError("assignment Series index must be unique")
        return assignment.to_dict()
    return assignment if isinstance(assignment, Mapping) else None


def _batch_summary(count: int, uniqueness: tuple[int, int] | None) -> EvaluationSummary:
    if uniqueness is None:
        return EvaluationSummary(count, count)
    unique_plans, unique_districts = uniqueness
    return EvaluationSummary(count, count, unique_plans, unique_districts)


@dataclass(frozen=True, slots=True)
class _LogicalOutput:
    source: int
    columns: tuple[int, ...]
    spec: _OutputSpec


@dataclass(frozen=True, slots=True)
class _GeometrySource:
    frame: GeoDataFrame
    node_id_column: str | None
    target_crs: CRS | None


@dataclass(frozen=True, slots=True)
class _UnitAlignment:
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PreparedGeometry:
    """Transformed, validated, and aligned geometry snapshot."""

    frame: GeoDataFrame
    wkb: tuple[bytes, ...]
    crs: CRS
    native: _scoring_engine.PreparedGeometry


@dataclass(frozen=True, slots=True)
class _PreparedRegion:
    raw: tuple[Hashable | None, ...]
    dense: tuple[int | None, ...]
    labels: tuple[Hashable, ...]


@dataclass(frozen=True, slots=True)
class _PreparedResources:
    """Published immutable resources used by metric validation and engine registration."""

    spec: _ResourceSpec
    topology: tuple[tuple[int, int], ...] | None
    node_columns: Mapping[tuple[str, str], tuple[float, ...]]
    edge_columns: Mapping[str, tuple[float, ...]]
    region_columns: Mapping[tuple[str, str], _PreparedRegion]
    alignment: _UnitAlignment | None
    geometry: _PreparedGeometry | None
    fixed_values: Mapping[Hashable, float]


@dataclass(slots=True)
class _ResourceCandidate:
    spec: _ResourceSpec
    topology: tuple[tuple[int, int], ...] | None
    node_columns: dict[tuple[str, str], tuple[float, ...]]
    edge_columns: dict[str, tuple[float, ...]]
    region_columns: dict[tuple[str, str], _PreparedRegion]
    alignment: _UnitAlignment | None
    geometry: _PreparedGeometry | None
    fixed_values: dict[Hashable, float]


class PlanEvaluator:
    """Prepare graph and geometry resources once, then evaluate plans with engine kernels.

    Graph structure is captured lazily, while metric columns are snapshotted when first required.
    Assignments must follow construction-time node order when supplied as sequences; mappings are
    reordered by node identifier. Arbitrary hashable district labels are mapped densely for Rust
    and restored in the result.

    Args:
        graph (nx.Graph): An undirected, simple NetworkX-compatible graph containing metric
            attributes.
        geometry (GeoDataFrame | None, optional): GeoDataFrame with exactly one row per graph node.
            The whole aligned frame is authoritative for ordinary node and region columns, except
            that its active geometry column is reserved for geometry-backed metrics. Defaults to
            None.
        node_id_column (str | None, optional): Geometry-frame column containing graph node
            identifiers. Defaults to None, which uses the GeoDataFrame index.
        target_crs (Any | None, optional): Projected CRS used by geometry-backed metrics. Defaults
            to None, which uses the geometry's CRS.

    Raises:
        ValueError: If the graph is empty, directed, a multigraph, or otherwise incompatible with
            the supplied geometry options.
    """

    def __init__(
        self,
        graph: nx.Graph,
        *,
        geometry: GeoDataFrame | None = None,
        node_id_column: str | None = None,
        target_crs: Any | None = None,
    ) -> None:
        if graph.is_directed():
            raise ValueError("graph must be undirected")
        if graph.is_multigraph():
            raise ValueError("graph must be simple, not a multigraph")
        if geometry is None and (node_id_column is not None or target_crs is not None):
            raise ValueError("node_id_column and target_crs require geometry")

        self._node_order = tuple(graph.nodes)
        if not self._node_order:
            raise ValueError("graph must contain at least one node")
        if len(set(self._node_order)) != len(self._node_order):
            raise ValueError("graph node identifiers must be unique")
        self._node_set = set(self._node_order)
        self._node_index = {node: index for index, node in enumerate(self._node_order)}
        for left, right in graph.edges:
            if left == right:
                raise ValueError(f"graph contains a self-loop at node {left!r}")
        self._graph = graph
        self._geometry_source: _GeometrySource | None = None
        self._edge_labels: frozenset[frozenset[Hashable]] | None = None
        self._verified_partition_graphs: WeakSet[GerryGraph] = WeakSet()
        self._metrics: list[tuple[str, _MetricBase]] = []
        self._resources: _PreparedResources | None = None
        self._preparing: _ResourceCandidate | None = None
        self._preparing_tally_indices: dict[str, int] | None = None
        self._engine: _scoring_engine.ScoringEngine | None = None
        self._tally_indices: dict[str, int] = {}
        self._engine_prepared: tuple[_OutputSpec, ...] = ()
        self._outputs: tuple[_LogicalOutput, ...] = ()
        if geometry is not None:
            self.add_geometry(
                geometry,
                node_id_column=node_id_column,
                target_crs=target_crs,
            )

    @property
    def _node_count(self) -> int:
        return len(self._node_order)

    @property
    def metrics(self) -> tuple[str, ...]:
        """Registered logical metric names in evaluation order."""
        return tuple(name for name, _ in self._metrics)

    @property
    def _has_geometry(self) -> bool:
        return self._geometry_source is not None

    def add_geometry(
        self,
        geometry: GeoDataFrame,
        *,
        node_id_column: str | None = None,
        target_crs: Any | None = None,
    ) -> "PlanEvaluator":
        """Record an authoritative GeoDataFrame for lazy metric preparation.

        Args:
            geometry (GeoDataFrame): Frame containing one row per graph node.
            node_id_column (str | None, optional): Column containing graph node identifiers.
                Defaults to None, which uses the frame index.
            target_crs (Any | None, optional): Projected CRS for geometry metrics. Defaults to None.

        Returns:
            PlanEvaluator: This evaluator, for chained registration.

        Raises:
            RuntimeError: If geometry is already registered or a metric was added first.
            TypeError: If ``geometry`` is not a GeoDataFrame.
            ValueError: If ``node_id_column`` is not a nonempty string or ``target_crs`` is invalid.
        """
        if self._metrics:
            raise RuntimeError("geometry must be added before the first metric")
        if self._has_geometry:
            raise RuntimeError("geometry has already been added")
        if not isinstance(geometry, GeoDataFrame):
            raise TypeError("geometry must be a GeoDataFrame")
        if node_id_column is not None and (
            not isinstance(node_id_column, str) or not node_id_column
        ):
            raise ValueError("node_id_column must be a nonempty string or None")
        resolved_crs = None if target_crs is None else CRS.from_user_input(target_crs)
        self._geometry_source = _GeometrySource(geometry, node_id_column, resolved_crs)
        self._invalidate()
        return self

    def add_metric(self, metric: Metric) -> "PlanEvaluator":
        """Register one logical metric.

        Args:
            metric (Metric): Metric description to register.

        Returns:
            PlanEvaluator: This evaluator, for chained registration.

        Raises:
            TypeError: If ``metric`` is not a supported metric description.
            ValueError: If its result name is already registered or its options are invalid.
        """
        if not isinstance(metric, _MetricBase):
            raise TypeError("metric must be a supported GerryTools metric description")
        instance = _result_name(metric)
        if any(existing == instance for existing, _ in self._metrics):
            raise ValueError(f"metric name {instance!r} is already registered")
        self._validate_metric_configuration(metric)
        self._metrics.append((instance, metric))
        self._invalidate()
        return self

    def add_metrics(self, *metrics: Metric) -> "PlanEvaluator":
        """Register metrics after atomically validating the whole batch.

        Result names and metric requirements are all checked before any metric is registered, so
        a failure anywhere in the batch registers nothing.

        Args:
            *metrics (Metric): Metric descriptions to register.

        Returns:
            PlanEvaluator: This evaluator, for chained registration.

        Raises:
            TypeError: If any value is not a supported metric description.
            ValueError: If a result name is duplicated or metric options are invalid.
        """
        names = {name for name, _ in self._metrics}
        batch: list[tuple[str, _MetricBase]] = []
        for metric in metrics:
            if not isinstance(metric, _MetricBase):
                raise TypeError("metric must be a supported GerryTools metric description")
            instance = _result_name(metric)
            if instance in names:
                raise ValueError(f"metric name {instance!r} is already registered")
            names.add(instance)
            batch.append((instance, metric))
        for _, metric in batch:
            self._validate_metric_configuration(metric)
        if batch:
            self._metrics.extend(batch)
            self._invalidate()
        return self

    def to_updaters(self) -> dict[str, Callable[[Partition], Any]]:
        """Return GerryChain updaters for the currently registered metrics.

        The first metric requested from a partition evaluates every registered metric. GerryChain
        caches that combined result, so requesting additional metrics from the same partition does
        not evaluate the plan again. The returned mapping reflects the metrics registered when this
        method is called.

        Returns:
            dict[str, Callable[[Partition], Any]]: Updaters keyed by registered metric name, plus
            one internal updater that shares the combined evaluation.
        """
        names = self.metrics
        if not names:
            return {}

        cache_name = f"_gerrytools:evaluation:{id(self):x}"
        updaters: dict[str, Callable[[Partition], Any]] = {cache_name: self.evaluate}
        for name in names:

            def metric_updater(partition: Partition, metric: str = name) -> Any:
                return partition[cache_name][metric]

            updaters[name] = metric_updater
        return updaters

    def evaluate(self, plan: Assignment | Partition) -> PlanEvalResult:
        """Evaluate one assignment or GerryChain partition.

        Args:
            plan (Assignment | Partition): District assignment or compatible partition.

        Returns:
            PlanEvalResult: Values keyed by registered metric name.

        Raises:
            TypeError: If the plan or required graph attributes have incompatible types.
            ValueError: If the plan, graph resources, or metric inputs are invalid.
        """
        self._prepare()
        results, _ = self._score_rows([self._normalize_plan(plan)])
        return PlanEvalResult(results)

    def evaluate_many(
        self,
        plans: Iterable[Assignment | Partition],
        *,
        sample_ids: Iterable[Hashable] | None = None,
        track_uniqueness: bool = False,
        progress: bool = False,
    ) -> ManyPlanEvalResult:
        """Evaluate a nonempty batch of assignments or partitions with stable district labels.

        Args:
            plans (Iterable[Assignment | Partition]): Assignments or partitions to evaluate.
            sample_ids (Iterable[Hashable] | None, optional): Unique result-index values in plan
                order. Defaults to None, which uses a range index.
            track_uniqueness (bool, optional): Whether to count label-invariant unique plans and
                districts. Defaults to False.
            progress (bool, optional): Whether to display a progress bar. Defaults to False.

        Returns:
            ManyPlanEvalResult: Metric values and optional uniqueness summary for every plan.

        Raises:
            TypeError: If a Boolean option or plan has an incompatible type.
            ValueError: If ``plans`` is empty or plan data and sample identifiers are invalid.
        """
        if not isinstance(track_uniqueness, bool):
            raise TypeError("track_uniqueness must be a boolean")
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean")
        iterator = iter(plans)
        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError("plans must contain at least one plan")
        self._prepare()
        rows = [self._normalize_plan(plan) for plan in chain((first,), iterator)]
        if not progress:
            results, uniqueness = self._score_rows(rows, track_uniqueness=track_uniqueness)
            return ManyPlanEvalResult(
                results,
                sample_ids,
                summary=_batch_summary(len(rows), uniqueness),
            )
        with tqdm(total=len(rows), desc="Evaluating plans", unit="plan") as bar:
            results, uniqueness = self._score_rows(
                rows,
                track_uniqueness=track_uniqueness,
                progress=bar.update,
            )
            return ManyPlanEvalResult(
                results,
                sample_ids,
                summary=_batch_summary(len(rows), uniqueness),
            )

    def evaluate_stream(
        self,
        source: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        *,
        max_samples: int | np.integer | None = None,
        batch_size: int | np.integer = 256,
        track_uniqueness: bool = False,
        progress: bool = False,
        update: bool = False,
    ) -> EnsembleEvalResult:
        """Evaluate a BEN, XBEN, or finalized BENDL stream into a Parquet run directory.

        Assignment positions must follow this evaluator's graph-node order. The output path must
        contain scores for the same assignment stream when it already exists. New score names are
        added to an existing run. Existing names require ``update=True`` and are replaced. A BENDL
        graph, when present, must use exactly this evaluator's node order. BEN, XBEN, and graph-free
        BENDL inputs leave ordering to the caller.

        Args:
            source (str | os.PathLike[str]): Input BEN, XBEN, or finalized BENDL file.
            output_dir (str | os.PathLike[str]): Directory containing the manifest and score tables.
            max_samples (int | np.integer | None, optional): Limit after expanding frame
                repetitions. Zero writes an empty run. Defaults to None, which reads the full run.
            batch_size (int | np.integer, optional): Maximum full assignment frames scored in one
                engine batch. Defaults to 256.
            track_uniqueness (bool, optional): Whether to count label-invariant unique plans and
                districts. Defaults to False.
            progress (bool, optional): Whether to display a progress bar. Defaults to False.
            update (bool, optional): Whether to replace registered score names already present in
                ``output_dir``. New names are always added. Defaults to False.

        Returns:
            EnsembleEvalResult: The completed ensemble evaluation.

        Raises:
            FileExistsError: If a registered score name is already present and ``update`` is
                false.
            RuntimeError: If no metrics are registered.
            TypeError: If a Boolean option has an incompatible type.
            ValueError: If a numeric option, input stream, assignment, or output path is invalid.
        """
        if not self._metrics:
            raise RuntimeError("at least one metric must be registered before scoring")
        if max_samples is not None and (
            not isinstance(max_samples, numbers.Integral)
            or isinstance(max_samples, (bool, np.bool_))
            or max_samples < 0
        ):
            raise ValueError("max_samples must be a nonnegative integer or None")
        if (
            not isinstance(batch_size, numbers.Integral)
            or isinstance(batch_size, (bool, np.bool_))
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        max_samples = None if max_samples is None else int(max_samples)
        batch_size = int(batch_size)
        if not isinstance(track_uniqueness, bool):
            raise TypeError("track_uniqueness must be a boolean")
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean")
        if not isinstance(update, bool):
            raise TypeError("update must be a boolean")

        source_path = Path(source)
        output_path = Path(output_dir)
        bendl_node_order_json = _node_order_json(self._node_order)
        engine = self._prepare()
        metrics = []
        for (instance, metric), logical_output in zip(self._metrics, self._outputs, strict=True):
            spec = logical_output.spec
            description = {
                "kind": metric._kind,
                "instance": instance,
                "options": metric._stream_options(self),
                "shape": spec.shape,
                "subkeys": _stream_subkeys(instance, spec),
                "dtypes": list(spec.dtypes),
                "axes": {"metric": [str(column) for column in spec.columns]},
            }
            if spec.shape == _ResultShape.REGION:
                assert spec.region_name is not None
                labels = self._stream_region_labels(instance, spec.region_name)
                if len(labels) != len(spec.regions):
                    raise RuntimeError("streamed region labels do not match prepared region order")
                description["axes"] = _stream_region_axes(spec, labels)
            metrics.append(description)
        metadata = json.dumps(
            {"source": str(source_path), "metrics": metrics},
            allow_nan=False,
            separators=(",", ":"),
        )
        projections = [
            (logical_output.source, list(logical_output.columns))
            for logical_output in self._outputs
        ]
        stream_options: _StreamRunOptions = {
            "bendl_node_order_json": bendl_node_order_json,
            "max_samples": max_samples,
            "batch_size": batch_size,
            "track_uniqueness": track_uniqueness,
            "progress": None,
        }
        existing = EnsembleEvalResult.open(output_path) if output_path.exists() else None
        if existing is not None:
            duplicates = tuple(name for name in self.metrics if name in existing.metrics)
            if duplicates and not update:
                names = ", ".join(repr(name) for name in duplicates)
                raise FileExistsError(
                    f"evaluation run already contains {names}; pass update=True to replace them"
                )

        def score(target: Path) -> None:
            if progress:
                total = max_samples
                with tqdm(total=total, desc="Evaluating ensemble", unit="sample") as bar:

                    def report_progress(amount: int, discovered_total: int | None) -> None:
                        if discovered_total is None:
                            bar.update(amount)
                        else:
                            bar.reset(total=discovered_total)

                    stream_options["progress"] = report_progress
                    engine.score_run(
                        source_path,
                        target,
                        metadata,
                        stream_options,
                        projections,
                    )
            else:
                engine.score_run(
                    source_path,
                    target,
                    metadata,
                    stream_options,
                    projections,
                )

        if existing is None:
            score(output_path)
            return EnsembleEvalResult.open(output_path)

        output_parent = output_path.parent if output_path.parent != Path("") else Path(".")
        with tempfile.TemporaryDirectory(
            dir=output_parent,
            prefix=f".{output_path.name}.update-",
        ) as temporary:
            staged = Path(temporary) / output_path.name
            score(staged)
            _merge_evaluation_run(output_path, existing, staged)
        return EnsembleEvalResult.open(output_path)

    def _score_rows(
        self,
        rows: list[list[Hashable]],
        *,
        track_uniqueness: bool = False,
        progress: Callable[[int], object] | None = None,
    ) -> tuple[dict[str, _MetricResult], tuple[int, int] | None]:
        assert self._engine is not None
        encoded, labels = _encode_districts(rows)
        engine_districts, engine_rows, uniqueness = self._engine.score_many(
            encoded, track_uniqueness, progress
        )
        expected_districts = list(range(len(labels)))
        if engine_districts != expected_districts:
            raise RuntimeError("scoring engine returned an unexpected district order")

        results = {}
        plan_count = len(rows)
        for (instance, _), output in zip(self._metrics, self._outputs, strict=True):
            engine_spec = self._engine_prepared[output.source]
            spec = output.spec
            values = np.asarray(
                [row[output.source] for row in engine_rows],
                dtype=np.float64,
            )
            if spec.shape == _ResultShape.REGION:
                values = values.reshape(
                    plan_count,
                    engine_spec.value_count,
                    len(labels),
                )[:, output.columns, :]
                values = values.reshape(
                    plan_count,
                    len(spec.columns),
                    len(spec.regions),
                    len(labels),
                )
                districts = labels
            elif spec.shape == _ResultShape.DISTRICT:
                values = values.reshape(
                    plan_count,
                    engine_spec.value_count,
                    len(labels),
                )[:, output.columns, :]
                districts = labels
            else:
                values = values.reshape(plan_count, engine_spec.value_count)[:, output.columns]
                districts = ()
            results[instance] = _MetricResult(
                values,
                spec.shape,
                spec.columns,
                districts,
                spec.dtypes,
                spec.regions,
                spec.region_name,
            )
        return results, uniqueness

    def _prepare(self) -> _scoring_engine.ScoringEngine:
        if self._engine is not None:
            return self._engine
        if not self._metrics:
            raise RuntimeError("at least one metric must be registered before scoring")

        required = _ResourceSpec()
        for _, metric in self._metrics:
            required |= metric._resources(self)
        prepared = _ResourceSpec() if self._resources is None else self._resources.spec
        resources_changed = not prepared.contains(required)
        candidate = self._build_resources(required) if resources_changed else None
        if candidate is not None:
            self._preparing = candidate

        try:
            for _, metric in self._metrics:
                metric._validate(self)

            engine_metrics: list[_MetricBase] = []
            sources = []
            for _, metric in self._metrics:
                for source, existing in enumerate(engine_metrics):
                    merged = existing._merge(metric)
                    if merged is not None:
                        engine_metrics[source] = merged
                        sources.append(source)
                        break
                else:
                    sources.append(len(engine_metrics))
                    engine_metrics.append(metric)

            engine = _scoring_engine.ScoringEngine()
            tally_keys: tuple[str, ...] = ()
            for metric in engine_metrics:
                tally_keys = _merged_keys(tally_keys, metric._tally_keys())
            tally_indices = {key: index for index, key in enumerate(tally_keys)}
            self._preparing_tally_indices = tally_indices
            if tally_keys:
                engine.set_tally_bank([self._numeric_node_column(key) for key in tally_keys])
            engine_prepared = tuple(metric._prepare(engine, self) for metric in engine_metrics)
            outputs = []
            for (_, metric), source in zip(self._metrics, sources, strict=True):
                prepared = engine_prepared[source]
                if prepared.shape == _ResultShape.REGION:
                    columns = tuple(range(prepared.value_count))
                    spec = prepared
                else:
                    columns = metric._column_indices(prepared.columns)
                    spec = _OutputSpec(
                        prepared.shape,
                        metric._result_columns(prepared.columns, columns),
                        tuple(prepared.dtypes[index] for index in columns),
                    )
                outputs.append(_LogicalOutput(source, columns, spec))
            if candidate is not None:
                if not candidate.spec.fixed_values.issubset(candidate.fixed_values):
                    raise RuntimeError("metric preparation did not build every fixed resource")
                resources = _freeze_resources(candidate)
            else:
                resources = self._resources
        finally:
            self._preparing = None
            self._preparing_tally_indices = None

        if resources_changed:
            assert resources is not None
            self._resources = resources
        self._engine = engine
        self._tally_indices = tally_indices
        self._engine_prepared = engine_prepared
        self._outputs = tuple(outputs)
        return engine

    def _build_resources(self, required: _ResourceSpec) -> _ResourceCandidate:
        current = self._resources
        candidate = _ResourceCandidate(
            spec=required | (_ResourceSpec() if current is None else current.spec),
            topology=None if current is None else current.topology,
            node_columns={} if current is None else dict(current.node_columns),
            edge_columns={} if current is None else dict(current.edge_columns),
            region_columns={} if current is None else dict(current.region_columns),
            alignment=None if current is None else current.alignment,
            geometry=None if current is None else current.geometry,
            fixed_values={} if current is None else dict(current.fixed_values),
        )
        if required.alignment and candidate.alignment is None:
            candidate.alignment = self._build_alignment()
        if required.topology and candidate.topology is None:
            candidate.topology = self._build_topology()
        for resource in sorted(required.node_columns - candidate.node_columns.keys()):
            candidate.node_columns[resource] = self._build_numeric_node_column(*resource, candidate)
        for key in sorted(required.edge_columns - candidate.edge_columns.keys()):
            candidate.edge_columns[key] = self._build_numeric_edge_column(key, candidate)
        for resource in sorted(required.region_columns - candidate.region_columns.keys()):
            candidate.region_columns[resource] = self._build_region_column(*resource, candidate)
        if required.geometry and candidate.geometry is None:
            candidate.geometry = self._build_geometry(candidate)
        return candidate

    def _invalidate(self) -> None:
        self._engine = None
        self._tally_indices = {}
        self._engine_prepared = ()
        self._outputs = ()

    def _normalize_plan(self, plan: Assignment | Partition) -> list[Any]:
        if not isinstance(plan, Partition):
            return self._normalize_assignment(plan)
        self._verify_partition_graph(plan)
        return self._normalize_assignment(_partition_assignment(plan))

    def _verify_partition_graph(self, partition: Partition) -> None:
        frozen = partition.graph
        graph = frozen.graph
        if graph in self._verified_partition_graphs:
            return

        edge_labels = self._edge_labels
        if edge_labels is None:
            edge_labels = frozenset(frozenset((left, right)) for left, right in self._graph.edges)
        nodes = {
            graph.original_nx_node_id_for_internal_node_id(node) for node in graph.node_indices
        }
        edges = frozenset(
            frozenset(
                (
                    graph.original_nx_node_id_for_internal_node_id(left),
                    graph.original_nx_node_id_for_internal_node_id(right),
                )
            )
            for edge in graph.edge_indices
            for left, right in (graph.get_edge_from_edge_id(edge),)
        )
        if nodes != self._node_set or edges != edge_labels:
            raise ValueError(
                "partition graph must have the same original node and edge sets as the evaluator"
            )
        self._edge_labels = edge_labels
        self._verified_partition_graphs.add(graph)

    def _normalize_assignment(self, assignment: Assignment) -> list[Any]:
        mapping = _assignment_mapping(assignment)
        if mapping is not None:
            missing = [node for node in self._node_order if node not in mapping]
            unexpected = [node for node in mapping if node not in self._node_set]
            if missing or unexpected:
                raise ValueError(
                    "assignment keys must exactly match graph nodes; "
                    f"missing={missing!r}, unexpected={unexpected!r}"
                )
            values = [mapping[node] for node in self._node_order]
        else:
            values = list(assignment)
            if len(values) != self._node_count:
                raise ValueError(
                    f"assignment has {len(values)} values; expected {self._node_count}"
                )
        if any(_is_missing(value) for value in values):
            raise ValueError("assignment cannot contain missing district labels")
        return values

    def _ordinary_column_resource(self, key: str) -> tuple[Literal["graph", "geometry"], str]:
        return ("geometry" if self._has_geometry else "graph", key)

    def _validate_metric_configuration(self, metric: _MetricBase) -> None:
        if metric._resources(self).geometry and not self._has_geometry:
            label = getattr(metric, "_geometry_label", type(metric).__name__)
            raise RuntimeError(f"{label} requires geometry")

    def _resource_view(self) -> _PreparedResources | _ResourceCandidate:
        resources = self._preparing if self._preparing is not None else self._resources
        if resources is None:
            raise RuntimeError("evaluator resources have not been prepared")
        return resources

    @property
    def _edges(self) -> list[tuple[int, int]]:
        topology = self._resource_view().topology
        if topology is None:
            raise RuntimeError("graph topology was not prepared")
        return list(topology)

    def _numeric_node_column(self, key: str) -> list[float]:
        resource = self._ordinary_column_resource(key)
        try:
            return list(self._resource_view().node_columns[resource])
        except KeyError:
            raise RuntimeError(f"node column {key!r} was not prepared") from None

    def _nonnegative_node_column(self, key: str, metric: str) -> list[float]:
        values = self._numeric_node_column(key)
        if any(value < 0 for value in values):
            raise ValueError(f"{metric} column {key!r} cannot contain negative values")
        return values

    def _tally_column_indices(self, keys: tuple[str, ...]) -> list[int]:
        indices = (
            self._preparing_tally_indices
            if self._preparing_tally_indices is not None
            else self._tally_indices
        )
        try:
            return [indices[key] for key in keys]
        except KeyError as error:
            raise RuntimeError(f"tally column {error.args[0]!r} was not prepared") from None

    def _fixed_value(self, key: Hashable, prepare: Callable[[], float]) -> float:
        resources = self._resource_view()
        if key in resources.fixed_values:
            return resources.fixed_values[key]
        if not isinstance(resources, _ResourceCandidate):
            raise RuntimeError(f"fixed metric value {key!r} was not prepared")
        resources.fixed_values[key] = prepare()
        return resources.fixed_values[key]

    def _numeric_graph_node_column(self, key: str) -> list[float]:
        try:
            return list(self._resource_view().node_columns[("graph", key)])
        except KeyError:
            raise RuntimeError(f"graph node column {key!r} was not prepared") from None

    def _numeric_edge_column(self, key: str) -> list[float]:
        try:
            return list(self._resource_view().edge_columns[key])
        except KeyError:
            raise RuntimeError(f"graph edge column {key!r} was not prepared") from None

    def _region_column(self, key: str) -> tuple[list[int | None], tuple[Hashable, ...]]:
        resource = self._ordinary_column_resource(key)
        try:
            region = self._resource_view().region_columns[resource]
        except KeyError:
            raise RuntimeError(f"region column {key!r} was not prepared") from None
        return list(region.dense), region.labels

    def _stream_region_labels(self, instance: str, key: str) -> list[dict[str, object]]:
        resource = self._ordinary_column_resource(key)
        region = self._resource_view().region_columns[resource]
        labels: list[dict[str, object]] = []
        seen: set[tuple[str, str | int]] = set()
        for node, value in zip(self._node_order, region.raw, strict=True):
            if value is None:
                continue
            if isinstance(value, str):
                label: tuple[str, str | int] = ("str", value)
            elif isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)):
                integer = int(value)
                if not -(2**63) <= integer < 2**63:
                    raise ValueError(
                        f"metric {instance!r} region label {value!r} at graph node {node!r} "
                        "cannot be represented in a streamed run"
                    )
                label = ("int", integer)
            else:
                raise ValueError(
                    f"metric {instance!r} region label {value!r} at graph node {node!r} "
                    "cannot be represented in a streamed run"
                )
            if label not in seen:
                seen.add(label)
                labels.append({"kind": label[0], "value": label[1]})
        return labels

    def _require_geometry(self, metric: str) -> _PreparedGeometry:
        if not self._has_geometry:
            raise RuntimeError(f"{metric} requires geometry")
        geometry = self._resource_view().geometry
        if geometry is None:
            raise RuntimeError(f"{metric} geometry was not prepared")
        return geometry

    def _build_alignment(self) -> _UnitAlignment:
        source = self._geometry_source
        assert source is not None
        units = source.frame
        if source.node_id_column is None:
            unit_nodes = units.index
        else:
            if source.node_id_column not in units.columns:
                raise ValueError(
                    f"geometry does not contain node ID column {source.node_id_column!r}"
                )
            unit_nodes = units[source.node_id_column]
        return _UnitAlignment(
            _alignment_positions(self._node_order, unit_nodes, target_name="graph nodes")
        )

    def _build_topology(self) -> tuple[tuple[int, int], ...]:
        edges = []
        for left, right in self._graph.edges:
            u = self._node_index[left]
            v = self._node_index[right]
            edges.append((min(u, v), max(u, v)))
        return tuple(sorted(edges))

    def _source_values(
        self,
        source: Literal["graph", "geometry"],
        key: str,
        candidate: _ResourceCandidate,
    ) -> tuple[tuple[Any, ...], str]:
        if source == "graph":
            values = []
            for node in self._node_order:
                data = self._graph.nodes[node]
                if key not in data:
                    raise ValueError(f"graph node {node!r} has no {key!r} attribute")
                values.append(data[key])
            return tuple(values), "graph node"

        geometry = self._geometry_source
        assert geometry is not None and candidate.alignment is not None
        if key == geometry.frame.geometry.name:
            raise ValueError(
                f"active geometry column {key!r} cannot be used as a graph attribute; "
                "omit geometry to read that column from the graph"
            )
        if key not in geometry.frame.columns:
            raise ValueError(f"geometry row {self._node_order[0]!r} has no {key!r} attribute")
        values = geometry.frame.iloc[list(candidate.alignment.positions)][key]
        return tuple(values), "geometry row"

    def _build_numeric_node_column(
        self,
        source: Literal["graph", "geometry"],
        key: str,
        candidate: _ResourceCandidate,
    ) -> tuple[float, ...]:
        values, label = self._source_values(source, key, candidate)
        return _checked_numeric_values(values, self._node_order, key, label)

    def _build_numeric_edge_column(
        self, key: str, candidate: _ResourceCandidate
    ) -> tuple[float, ...]:
        assert candidate.topology is not None
        values = []
        identifiers = []
        for left, right in candidate.topology:
            u = self._node_order[left]
            v = self._node_order[right]
            data = self._graph.edges[u, v]
            if key not in data:
                raise ValueError(f"graph edge {(left, right)!r} has no {key!r} attribute")
            values.append(data[key])
            identifiers.append((left, right))
        return _checked_numeric_values(values, identifiers, key, "graph edge")

    def _build_region_column(
        self,
        source: Literal["graph", "geometry"],
        key: str,
        candidate: _ResourceCandidate,
    ) -> _PreparedRegion:
        raw_values, label = self._source_values(source, key, candidate)
        dense: dict[Hashable, int] = {}
        labels: list[Hashable] = []
        raw: list[Hashable | None] = []
        values = []
        for node, value in zip(self._node_order, raw_values, strict=True):
            if _is_missing(value):
                raw.append(None)
                values.append(None)
                continue
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{label} {node!r} attribute {key!r} cannot be boolean")
            try:
                if value not in dense:
                    dense[value] = len(labels)
                    labels.append(value)
                raw.append(value)
                values.append(dense[value])
            except TypeError as error:
                raise ValueError(f"{label} {node!r} attribute {key!r} must be hashable") from error
        return _PreparedRegion(tuple(raw), tuple(values), tuple(labels))

    def _build_geometry(self, candidate: _ResourceCandidate) -> _PreparedGeometry:
        source = self._geometry_source
        assert source is not None and candidate.alignment is not None
        geometry_column = source.frame.geometry.name
        frame = cast(
            GeoDataFrame,
            source.frame.iloc[list(candidate.alignment.positions)][[geometry_column]].copy(),
        )
        frame.index = _object_index(self._node_order, "node")
        validated = _validated_geometry_frame(frame, crs=source.target_crs)
        wkb = tuple(bytes(value) for value in validated.geometry.to_wkb(hex=False))
        return _PreparedGeometry(
            validated,
            wkb,
            CRS.from_user_input(validated.crs),
            _scoring_engine._prepare_geometry(wkb),
        )


def _checked_numeric_values(
    values: Iterable[Any],
    identifiers: Iterable[Any],
    key: str,
    label: str,
) -> tuple[float, ...]:
    checked = []
    for identifier, value in zip(identifiers, values, strict=True):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{label} {identifier!r} attribute {key!r} must be finite numeric")
        checked.append(float(value))
    return tuple(checked)


def _merge_evaluation_run(
    output: Path,
    existing: EnsembleEvalResult,
    staged: Path,
) -> None:
    staged_run = EnsembleEvalResult.open(staged)
    if (
        existing.summary.samples != staged_run.summary.samples
        or existing.summary.accepted != staged_run.summary.accepted
        or existing._districts != staged_run._districts
    ):
        raise ValueError("new scores do not match the existing run dimensions")

    existing_manifest = _version_2_manifest(_read_run_manifest(output))
    staged_manifest = _version_2_manifest(_read_run_manifest(staged))
    existing_entries = cast("list[dict[str, object]]", existing_manifest["metrics"])
    staged_entries = cast("list[dict[str, object]]", staged_manifest["metrics"])
    staged_by_name = {cast(str, entry["instance"]): entry for entry in staged_entries}

    merged = []
    for entry in existing_entries:
        name = cast(str, entry["instance"])
        merged.append(staged_by_name.pop(name, entry))
    merged.extend(entry for entry in staged_entries if entry["instance"] in staged_by_name)

    replaced = set(existing.metrics) & set(staged_run.metrics)
    stale = {table.path for name in replaced for table in existing._metric_metadata[name].tables}
    staged_tables = [
        table for metric in staged_run._metric_metadata.values() for table in metric.tables
    ]
    destinations = {output / table.path.relative_to(staged): table.path for table in staged_tables}
    unrelated = {
        table.path
        for name, metric in existing._metric_metadata.items()
        if name not in replaced
        for table in metric.tables
    }
    for destination in destinations:
        if any(_paths_overlap(destination, path) for path in unrelated):
            raise ValueError(f"score output path conflicts with an existing metric: {destination}")
        if os.path.lexists(destination) and destination not in stale:
            raise FileExistsError(f"new score would overwrite an untracked path: {destination}")
        parent = destination.parent
        while parent != output:
            if parent.is_symlink() or (os.path.lexists(parent) and not parent.is_dir()):
                raise FileExistsError(f"new score has an obstructed output path: {parent}")
            parent = parent.parent

    rollback_root = staged.parent / ".rollback"
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for destination, source in destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(destination):
                backup = rollback_root / destination.relative_to(output)
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backups[destination] = backup
            os.replace(source, destination)
            installed.append(destination)

        existing_manifest["metrics"] = merged
        _write_run_manifest(output, existing_manifest)
    except BaseException:
        for destination in reversed(installed):
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(backup, destination)
        raise

    for path in stale - set(destinations):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    for parent in sorted(
        {path.parent for path in stale}, key=lambda path: len(path.parts), reverse=True
    ):
        if parent != output:
            try:
                parent.rmdir()
            except OSError:
                pass


def _read_run_manifest(path: Path) -> dict[str, object]:
    with find_manifest_path(path).open() as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("evaluation manifest must be an object")
    return cast("dict[str, object]", manifest)


def _version_2_manifest(manifest: dict[str, object]) -> dict[str, object]:
    version = manifest.get("format_version")
    if version == 2:
        return manifest
    if version != 1:
        raise ValueError("evaluation manifest must use format version 1 or 2")
    metrics = cast("list[dict[str, object]]", manifest["metrics"])
    for metric in metrics:
        metric["tables"] = [
            {
                "path": metric.pop("table"),
                "subkeys": metric["subkeys"],
                "size": metric.pop("table_size"),
                "sha256": metric.pop("table_sha256"),
            }
        ]
    manifest["format_version"] = 2
    return manifest


def _write_run_manifest(path: Path, manifest: dict[str, object]) -> None:
    previous = find_manifest_path(path)
    published = preferred_manifest_path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path,
            prefix=".manifest-",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(manifest, file, allow_nan=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, published)
        temporary = None
        if previous != published:
            try:
                previous.unlink(missing_ok=True)
            except OSError:
                pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _paths_overlap(left: Path, right: Path) -> bool:
    left_parts = left.parts
    right_parts = right.parts
    common = min(len(left_parts), len(right_parts))
    return left_parts[:common] == right_parts[:common]


def _freeze_resources(candidate: _ResourceCandidate) -> _PreparedResources:
    return _PreparedResources(
        spec=candidate.spec,
        topology=candidate.topology,
        node_columns=MappingProxyType(candidate.node_columns),
        edge_columns=MappingProxyType(candidate.edge_columns),
        region_columns=MappingProxyType(candidate.region_columns),
        alignment=candidate.alignment,
        geometry=candidate.geometry,
        fixed_values=MappingProxyType(candidate.fixed_values),
    )


def _partition_assignment(partition: Partition) -> dict[Hashable, Hashable]:
    graph = partition.graph.graph
    vector = partition.assignment_vector
    assignment = {}
    for internal in graph.node_indices:
        if not isinstance(internal, numbers.Integral) or not 0 <= int(internal) < len(vector):
            raise ValueError("partition graph must use assignment-vector node identifiers")
        original = graph.original_nx_node_id_for_internal_node_id(internal)
        if original in assignment:
            raise ValueError("partition graph original node identifiers must be unique")
        assignment[original] = vector[int(internal)]
    return assignment


def _node_order_json(node_order: tuple[Hashable, ...]) -> str | None:
    try:
        return json.dumps(node_order, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        # Raw BEN/XBEN streams carry no graph and can score arbitrary hashable labels.
        # Rust rejects this missing comparison only when the source is actually BENDL.
        return None


def _partition_graph(partition: Partition) -> nx.Graph:
    """Reconstruct a NetworkX graph under a partition's original node identifiers."""
    source = partition.graph.graph
    graph = nx.Graph()
    internal_nodes = []
    for internal in source.node_indices:
        if not isinstance(internal, numbers.Integral):
            raise ValueError("partition graph must use assignment-vector node identifiers")
        internal_nodes.append(int(internal))
    for internal in sorted(internal_nodes):
        original = source.original_nx_node_id_for_internal_node_id(internal)
        data = dict(source.node_data(internal))
        data.pop("__networkx_node__", None)
        graph.add_node(original, **data)
    for edge in source.edge_indices:
        left, right = source.get_edge_from_edge_id(edge)
        graph.add_edge(
            source.original_nx_node_id_for_internal_node_id(left),
            source.original_nx_node_id_for_internal_node_id(right),
            **dict(source.edge_data(edge)),
        )
    return graph


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _result_name(metric: _MetricBase) -> str:
    result = metric._default_name() if metric.result_name is None else metric.result_name
    if not isinstance(result, str):
        raise TypeError("metric name must be a string")
    if not is_valid_metric_name(result):
        raise ValueError(
            "metric names must contain only ASCII letters, digits, '_', '-', and '.' "
            "and cannot equal '.', '..', or 'manifest.json'; choose another name"
        )
    return result


def _stream_subkeys(instance: str, spec: _OutputSpec) -> list[str]:
    if spec.shape == _ResultShape.REGION:
        return [
            f"{metric}__region_{region}"
            for metric in spec.columns
            for region in range(len(spec.regions))
        ]

    subkeys = [str(column) for column in spec.columns]
    if any(not subkey for subkey in subkeys):
        raise ValueError(f"metric {instance!r} has an empty streaming output column")
    if len(set(subkeys)) != len(subkeys):
        raise ValueError(f"metric {instance!r} has duplicate streaming output column names")
    return subkeys


def _stream_region_axes(
    spec: _OutputSpec,
    labels: list[dict[str, object]],
) -> dict[str, object]:
    assert spec.shape == _ResultShape.REGION and spec.region_name is not None
    return {
        "metric": list(spec.columns),
        "region": {"name": spec.region_name, "labels": labels},
    }


def _encode_districts(rows: list[list[Hashable]]) -> tuple[list[list[int]], tuple[Hashable, ...]]:
    label_to_id: dict[Hashable, int] = {}
    labels: list[Hashable] = []
    first = rows[0]
    for label in first:
        try:
            if label not in label_to_id:
                label_to_id[label] = len(labels)
                labels.append(label)
        except TypeError as error:
            raise ValueError("district labels must be hashable") from error

    if len(labels) > _MAX_DISTRICTS:
        raise ValueError(f"assignments cannot contain more than {_MAX_DISTRICTS} districts")

    encoded = []
    expected = set(range(len(labels)))
    for row in rows:
        try:
            dense = [label_to_id[label] for label in row]
        except (KeyError, TypeError) as error:
            raise ValueError("district labels must be the same in every assignment") from error
        if set(dense) != expected:
            raise ValueError("district labels must be the same in every assignment")
        encoded.append(dense)
    return encoded, tuple(labels)
