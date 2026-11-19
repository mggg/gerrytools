"""Orchestrate reproducible travel-time table construction.

Scoring an existing table has no OSM dependency. This module coordinates the
optional road preparation, snapping, and routing steps used to build one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Hashable, Literal

import geopandas as gpd
import networkx as nx
import numpy as np

from .od_table import TravelTimeTable, UnitId
from .provenance import default_snapping_path, save_snapping_artifact
from .road_graph import (
    DEFAULT_FALLBACK_SPEED_KPH,
    DEFAULT_FERRY_SPEED_KPH,
    FerryBuildSummary,
    SpeedRepairSummary,
    _weak_component_count,
    build_osm_drive_graph,
)
from .routing import (
    _directional_statistics,
    _save_directed_audit,
    compute_directed_travel_times,
)
from .snapping import snap_units_to_graph
from .table_io import save_travel_time_table

BuildStatus = Literal["applicable", "not_applicable"]


@dataclass(frozen=True, slots=True)
class TravelTimeBuildReport:
    """Validation and provenance summary for one table build."""

    status: BuildStatus
    unit_count: int
    unique_snap_count: int
    graph_node_count: int
    graph_edge_count: int
    graph_component_count: int
    connectivity_component_sizes: tuple[int, ...]
    asymmetric_pair_count: int
    max_directional_difference_seconds: float
    mean_snap_distance_m: float
    max_snap_distance_m: float
    ferry: FerryBuildSummary = field(default_factory=FerryBuildSummary)
    speed_repair: SpeedRepairSummary = field(default_factory=SpeedRepairSummary)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ("applicable", "not_applicable"):
            raise ValueError(f"Unknown travel-time build status: {self.status!r}")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "status": self.status,
            "unit_count": self.unit_count,
            "unique_snap_count": self.unique_snap_count,
            "graph_node_count": self.graph_node_count,
            "graph_edge_count": self.graph_edge_count,
            "graph_component_count": self.graph_component_count,
            "connectivity_component_sizes": list(self.connectivity_component_sizes),
            "asymmetric_pair_count": self.asymmetric_pair_count,
            "max_directional_difference_seconds": self.max_directional_difference_seconds,
            "mean_snap_distance_m": self.mean_snap_distance_m,
            "max_snap_distance_m": self.max_snap_distance_m,
            "ferry": self.ferry.to_dict(),
            "speed_repair": self.speed_repair.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TravelTimeBuildResult:
    """Result of a build, including an optional table when the universe is connected."""

    table: TravelTimeTable | None
    report: TravelTimeBuildReport
    unit_to_node: Mapping[UnitId, Hashable]
    snap_distances_m: Mapping[UnitId, float] = field(default_factory=dict)
    directed_times: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        table_exists = self.table is not None
        report_is_applicable = self.report.status == "applicable"
        if table_exists != report_is_applicable:
            raise ValueError(
                "TravelTimeBuildResult table and report status disagree: "
                f"table_exists={table_exists}, status={self.report.status!r}"
            )
        object.__setattr__(self, "unit_to_node", MappingProxyType(dict(self.unit_to_node)))
        object.__setattr__(self, "snap_distances_m", MappingProxyType(dict(self.snap_distances_m)))
        if self.directed_times is not None:
            directed = np.array(self.directed_times, dtype=np.float32, copy=True)
            directed.setflags(write=False)
            object.__setattr__(self, "directed_times", directed)

    @property
    def is_applicable(self) -> bool:
        """Whether a connected scoring table was produced.

        Applicability means the selected snapped road nodes are mutually
        reachable. It does not prove that every representative point was
        geographically near its snapped node; inspect ``snap_distances_m``.
        """
        return self.table is not None


def _write_json(path: str | Path, document: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _speed_repair_from_graph(graph: nx.Graph) -> SpeedRepairSummary:
    payload = graph.graph.get("speed_repair")
    if not isinstance(payload, Mapping):
        return SpeedRepairSummary()
    return SpeedRepairSummary(
        inspected=int(payload.get("inspected", 0)),
        repaired_speed=int(payload.get("repaired_speed", 0)),
        repaired_travel_time=int(payload.get("repaired_travel_time", 0)),
        fallback_speed_kph=float(payload.get("fallback_speed_kph", DEFAULT_FALLBACK_SPEED_KPH)),
    )


def build_travel_time_table(
    graph: nx.Graph,
    unit_to_node: Mapping[UnitId, Hashable],
    *,
    snap_distances_m: Mapping[UnitId, float] | None = None,
    ferry: FerryBuildSummary | None = None,
    speed_repair: SpeedRepairSummary | None = None,
    metadata: Mapping[str, Any] | None = None,
    batch_size: int = 32,
    keep_directed: bool = False,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    directed_audit_path: str | Path | None = None,
) -> TravelTimeBuildResult:
    """Route, symmetrize, validate, and optionally publish a scoring table.

    A genuinely disconnected universe returns ``table=None`` and a report with
    status ``not_applicable``. It never publishes a partial scoring table.

    Args:
        graph (nx.Graph): Prepared road graph with nonnegative ``travel_time`` edges.
        unit_to_node (Mapping[UnitId, Hashable]): Selected road node for each unit.
        snap_distances_m (Mapping[UnitId, float] | None): Optional point-to-node distances.
        ferry (FerryBuildSummary | None): Ferry-processing summary for the report.
        speed_repair (SpeedRepairSummary | None): Speed-repair summary for the report.
        metadata (Mapping[str, Any] | None): JSON-compatible build metadata.
        batch_size (int): Number of source nodes routed per SciPy call.
        keep_directed (bool): Retain the directed matrix in the returned result.
        output_path (str | Path | None): Optional connected scoring-table destination.
        report_path (str | Path | None): Optional JSON build-report destination.
        directed_audit_path (str | Path | None): Optional directed-matrix audit destination.

    Returns:
        TravelTimeBuildResult: Validated table when applicable, report, and snapping details.

    Raises:
        KeyError: If a selected road node is absent from the graph.
        ValueError: If graph weights, batch size, or snapping distances are invalid.
    """
    directed = compute_directed_travel_times(graph, unit_to_node, batch_size=batch_size)
    unit_ids = tuple(unit_to_node)
    symmetric = TravelTimeTable.from_directed_matrix(unit_ids, directed)
    components = symmetric.connectivity_components()
    applicable = symmetric.is_fully_reachable
    asymmetry_count, max_asymmetry = _directional_statistics(directed)

    snap_values = np.array(list((snap_distances_m or {}).values()), dtype=float)
    if snap_values.size and not np.isfinite(snap_values).all():
        raise ValueError("snap_distances_m must contain only finite values")
    build_metadata = dict(metadata or {})
    repair_summary = speed_repair or _speed_repair_from_graph(graph)
    if repair_summary.repaired_speed or repair_summary.repaired_travel_time:
        build_metadata["speed_repair"] = {
            **repair_summary.to_dict(),
            "policy": "impute_fallback_speed_keep_edge",
        }
    report = TravelTimeBuildReport(
        status="applicable" if applicable else "not_applicable",
        unit_count=len(unit_ids),
        unique_snap_count=len(set(unit_to_node.values())),
        graph_node_count=graph.number_of_nodes(),
        graph_edge_count=graph.number_of_edges(),
        graph_component_count=_weak_component_count(graph),
        connectivity_component_sizes=tuple(len(component) for component in components),
        asymmetric_pair_count=asymmetry_count,
        max_directional_difference_seconds=max_asymmetry,
        mean_snap_distance_m=float(np.mean(snap_values)) if snap_values.size else 0.0,
        max_snap_distance_m=float(np.max(snap_values)) if snap_values.size else 0.0,
        ferry=ferry or FerryBuildSummary(),
        speed_repair=repair_summary,
        metadata=build_metadata,
    )

    if report_path is not None:
        _write_json(report_path, report.to_dict())
    if directed_audit_path is not None:
        _save_directed_audit(directed_audit_path, unit_ids, directed)
    if applicable and output_path is not None:
        artifact_metadata = dict(build_metadata)
        artifact_metadata["build_report"] = report.to_dict()
        save_travel_time_table(
            output_path,
            symmetric,
            time_unit="seconds",
            metadata=artifact_metadata,
        )
        if snap_distances_m:
            save_snapping_artifact(
                default_snapping_path(output_path),
                unit_to_node,
                snap_distances_m,
                metadata=artifact_metadata,
            )

    return TravelTimeBuildResult(
        table=symmetric if applicable else None,
        report=report,
        unit_to_node=unit_to_node,
        snap_distances_m=dict(snap_distances_m or {}),
        directed_times=directed if keep_directed else None,
    )


def build_travel_time_table_from_pbf(
    unit_points: gpd.GeoDataFrame,
    pbf_path: str | Path,
    *,
    id_column: str,
    unit_polygons: gpd.GeoDataFrame | None = None,
    ferry_speed_kph: float = DEFAULT_FERRY_SPEED_KPH,
    fallback_speed_kph: float = DEFAULT_FALLBACK_SPEED_KPH,
    batch_size: int = 32,
    keep_directed: bool = False,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    directed_audit_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TravelTimeBuildResult:
    """Build a complete table from representative points and a local OSM PBF.

    ``unit_points`` must be a nonempty GeoDataFrame of Point geometries with a
    declared CRS and unique ``id_column`` values. Optional ``unit_polygons``
    enables inside-first snapping with the nearest exterior-boundary fallback.
    This helper does not calculate centroids or repair geometry.

    Args:
        unit_points (gpd.GeoDataFrame): Representative point for every geographic unit.
        pbf_path (str | Path): Local OpenStreetMap ``.osm.pbf`` extract.
        id_column (str): Unique unit identifier shared by points and optional polygons.
        unit_polygons (gpd.GeoDataFrame | None): Optional boundaries for polygon-aware snapping.
        ferry_speed_kph (float): Fallback speed for ferries without durations.
        fallback_speed_kph (float): Fallback speed for invalid road-edge speeds.
        batch_size (int): Number of source nodes routed per SciPy call.
        keep_directed (bool): Retain the directed matrix in the returned result.
        output_path (str | Path | None): Optional connected scoring-table destination.
        report_path (str | Path | None): Optional JSON build-report destination.
        directed_audit_path (str | Path | None): Optional directed-matrix audit destination.
        metadata (Mapping[str, Any] | None): JSON-compatible user metadata.

    Returns:
        TravelTimeBuildResult: Build report, snapping details, and table when applicable.

    Raises:
        TypeError: If ``unit_points`` is not a GeoDataFrame.
        FileNotFoundError: If the PBF does not exist.
        ImportError: If optional OSM dependencies are unavailable.
        ValueError: If graph construction, snapping, or routing inputs are invalid.
    """
    if not isinstance(unit_points, gpd.GeoDataFrame):
        raise TypeError("unit_points must be a GeoDataFrame of Point geometries")
    graph, ferry = build_osm_drive_graph(
        pbf_path,
        ferry_speed_kph=ferry_speed_kph,
        fallback_speed_kph=fallback_speed_kph,
    )
    unit_to_node, snap_distances = snap_units_to_graph(
        graph,
        unit_points,
        id_column=id_column,
        unit_polygons=unit_polygons,
    )
    provenance = dict(metadata or {})
    provenance.update(
        {
            "source_pbf": str(Path(pbf_path).resolve()),
            "representative_point": {
                "method": "caller_supplied_points",
                "crs": str(unit_points.crs),
            },
            "snapping": {
                "method": (
                    "inside_nearest_anchor_else_nearest_exterior_boundary"
                    if unit_polygons is not None
                    else "nearest_drive_graph_node"
                ),
                "road_network": "principal_strong_component",
                **(
                    {"fallback_boundary": "exterior_rings_only"}
                    if unit_polygons is not None
                    else {}
                ),
            },
            "road_graph_filter": dict(graph.graph.get("connectivity_filter", {})),
            "ferry_missing_duration_speed_kph": float(ferry_speed_kph),
            "fallback_speed_kph": float(fallback_speed_kph),
            "symmetrization": "mean_of_directions",
            "measure": (
                "Travel time between connected road-network nodes selected by the "
                "documented polygon-aware snapping policy."
                if unit_polygons is not None
                else "Travel time between the connected road-network nodes nearest "
                "the supplied representative points."
            ),
        }
    )
    return build_travel_time_table(
        graph,
        unit_to_node,
        snap_distances_m=snap_distances,
        ferry=ferry,
        speed_repair=_speed_repair_from_graph(graph),
        metadata=provenance,
        batch_size=batch_size,
        keep_directed=keep_directed,
        output_path=output_path,
        report_path=report_path,
        directed_audit_path=directed_audit_path,
    )
