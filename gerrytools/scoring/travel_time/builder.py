"""Reproducible construction of travel-time scoring tables from OSM data.

The score itself has no OSM dependency. This module isolates the optional,
expensive build step and imports OSMnx/Pyrosm only when a PBF is read.
"""

from __future__ import annotations

import importlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Hashable, Literal, Mapping, Sequence, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge

from .od_table import TravelTimeTable, UnitId
from .table_io import save_travel_time_table

DEFAULT_FERRY_SPEED_KPH = 20.0
BuildStatus = Literal["applicable", "not_applicable"]

_ACCESS_ALLOWED = frozenset({"yes", "designated", "permissive", "official", "destination"})
_ACCESS_DENIED = frozenset({"no", "private"})
_DRIVABLE_FERRY_CLASSES = frozenset(
    {
        "yes",
        "motorcar",
        "motor_vehicle",
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "track",
        "road",
        "living_street",
    }
)
_ISO_DURATION = re.compile(
    r"^P(?:\d+D)?T(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class FerryBuildSummary:
    """Counts describing which OSM ferry ways were added to the road graph."""

    found: int = 0
    included: int = 0
    skipped_access: int = 0
    skipped_geometry: int = 0
    osm_duration: int = 0
    estimated_duration: int = 0


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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            "ferry": {
                "found": self.ferry.found,
                "included": self.ferry.included,
                "skipped_access": self.ferry.skipped_access,
                "skipped_geometry": self.ferry.skipped_geometry,
                "osm_duration": self.ferry.osm_duration,
                "estimated_duration": self.ferry.estimated_duration,
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TravelTimeBuildResult:
    """Result of a build, including an optional table when the universe is connected."""

    table: TravelTimeTable | None
    report: TravelTimeBuildReport
    unit_to_node: Mapping[UnitId, Hashable]
    directed_times: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_to_node", MappingProxyType(dict(self.unit_to_node)))
        if self.directed_times is not None:
            directed = np.array(self.directed_times, dtype=np.float32, copy=True)
            directed.setflags(write=False)
            object.__setattr__(self, "directed_times", directed)

    @property
    def is_applicable(self) -> bool:
        """Whether a connected scoring table was produced."""
        return self.table is not None


def census_representative_points(
    units: pd.DataFrame,
    *,
    id_column: str,
    latitude_column: str = "INTPTLAT20",
    longitude_column: str = "INTPTLON20",
) -> gpd.GeoDataFrame:
    """Create WGS84 points from Census internal-point columns.

    This works for any Census geography carrying the requested columns; it is
    not restricted to VTDs.
    """
    required = [id_column, latitude_column, longitude_column]
    missing = [column for column in required if column not in units.columns]
    if missing:
        raise KeyError(f"Representative-point columns not found: {missing!r}")
    if len(units) == 0:
        raise ValueError("units must contain at least one row")

    ids = units[id_column].to_numpy()
    if bool(pd.isna(ids).any()) or len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{id_column!r} must contain unique, non-missing unit IDs")
    try:
        latitudes = np.asarray(pd.to_numeric(units[latitude_column], errors="raise"), dtype=float)
        longitudes = np.asarray(pd.to_numeric(units[longitude_column], errors="raise"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Representative-point coordinates must be numeric") from exc
    if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
        raise ValueError("Representative-point coordinates must be finite")
    if (
        not ((-90.0 <= latitudes) & (latitudes <= 90.0)).all()
        or not ((-180.0 <= longitudes) & (longitudes <= 180.0)).all()
    ):
        raise ValueError("Representative-point coordinates fall outside longitude/latitude bounds")

    return gpd.GeoDataFrame(
        {id_column: ids.tolist()},
        geometry=gpd.points_from_xy(longitudes, latitudes),
        crs="EPSG:4326",
    )


def parse_osm_duration_seconds(value: Any) -> float | None:
    """Parse an OSM ``duration`` value, returning seconds when valid.

    Supported forms are ``M``, ``H:MM``, ``H:MM:SS``, and ISO-8601 time
    durations such as ``PT1H30M``. A bare number follows OSM convention and is
    interpreted as minutes.
    """
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.replace(".", "", 1).isdigit():
        minutes = float(text)
        return minutes * 60.0 if minutes >= 0 else None
    iso = _ISO_DURATION.fullmatch(text)
    if iso:
        hours = float(iso.group("hours") or 0.0)
        minutes = float(iso.group("minutes") or 0.0)
        seconds = float(iso.group("seconds") or 0.0)
        total = hours * 3600.0 + minutes * 60.0 + seconds
        return total if total >= 0 else None
    parts = text.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if any(number < 0 for number in numbers) or any(number >= 60 for number in numbers[1:]):
        return None
    if len(numbers) == 2:
        hours, minutes = numbers
        return hours * 3600.0 + minutes * 60.0
    hours, minutes, seconds = numbers
    return hours * 3600.0 + minutes * 60.0 + seconds


def _tag(value: Any) -> str | None:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return None
    return str(value).strip().lower()


def _allows_motor_vehicle(row: Mapping[str, Any]) -> bool:
    motor_values = [_tag(row.get("motorcar")), _tag(row.get("motor_vehicle"))]
    if any(value in _ACCESS_ALLOWED for value in motor_values):
        return True
    if any(value in _ACCESS_DENIED for value in motor_values):
        return False
    if _tag(row.get("access")) in _ACCESS_DENIED or _tag(row.get("vehicle")) in _ACCESS_DENIED:
        return False
    if _tag(row.get("vehicle")) in _ACCESS_ALLOWED:
        return True
    return _tag(row.get("ferry")) in _DRIVABLE_FERRY_CLASSES


def _as_line(geometry: Any) -> LineString | None:
    if isinstance(geometry, LineString):
        return geometry if not geometry.is_empty else None
    if isinstance(geometry, MultiLineString):
        merged = linemerge(geometry)
        return merged if isinstance(merged, LineString) and not merged.is_empty else None
    return None


def _weak_component_count(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    if isinstance(graph, nx.DiGraph):
        return nx.number_weakly_connected_components(graph)
    return nx.number_connected_components(graph)


def add_vehicle_ferries(
    graph: nx.MultiDiGraph,
    ferry_features: gpd.GeoDataFrame | None,
    *,
    ferry_speed_kph: float = DEFAULT_FERRY_SPEED_KPH,
) -> FerryBuildSummary:
    """Add vehicle-permitted ferry ways to a projected road graph.

    OSM ``duration`` wins when present. Otherwise travel time is estimated from
    projected route length at ``ferry_speed_kph``. Ambiguous ferries without an
    affirmative motor-vehicle tag or drivable ``ferry=*`` class are skipped.
    """
    if not np.isfinite(ferry_speed_kph) or ferry_speed_kph <= 0:
        raise ValueError("ferry_speed_kph must be finite and positive")
    if ferry_features is None or len(ferry_features) == 0:
        return FerryBuildSummary()
    if graph.graph.get("crs") is None:
        raise ValueError("road graph must declare a CRS before ferries are added")
    if ferry_features.crs is None:
        raise ValueError("ferry_features must declare a CRS")

    try:
        ox = importlib.import_module("osmnx")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install gerrytools[travel-time] to add OSM ferries") from exc

    projected = ferry_features.to_crs(graph.graph["crs"])
    candidates: list[tuple[pd.Series, LineString]] = []
    skipped_access = 0
    skipped_geometry = 0
    for _, row in projected.iterrows():
        if not _allows_motor_vehicle(cast(Mapping[str, Any], row)):
            skipped_access += 1
            continue
        line = _as_line(row.geometry)
        if line is None or line.length <= 0:
            skipped_geometry += 1
            continue
        candidates.append((row, line))

    if not candidates:
        return FerryBuildSummary(
            found=len(projected),
            skipped_access=skipped_access,
            skipped_geometry=skipped_geometry,
        )

    endpoints = [(line.coords[0], line.coords[-1]) for _, line in candidates]
    xs = [coordinate[0] for pair in endpoints for coordinate in pair]
    ys = [coordinate[1] for pair in endpoints for coordinate in pair]
    nearest = np.atleast_1d(ox.distance.nearest_nodes(graph, X=xs, Y=ys))

    included = 0
    osm_duration = 0
    estimated_duration = 0
    for index, ((row, line), (start, end)) in enumerate(zip(candidates, endpoints)):
        _ = start, end
        u = nearest[2 * index].item()
        v = nearest[2 * index + 1].item()
        if u == v:
            skipped_geometry += 1
            continue
        duration = parse_osm_duration_seconds(row.get("duration"))
        if duration is None or duration <= 0:
            duration = float(line.length) / (ferry_speed_kph * 1000.0 / 3600.0)
            source = "estimated_20_kph" if ferry_speed_kph == 20.0 else "estimated_speed"
            estimated_duration += 1
        else:
            source = "osm_duration"
            osm_duration += 1
        attributes = {
            "route": "ferry",
            "length": float(line.length),
            "speed_kph": float(ferry_speed_kph),
            "travel_time": float(duration),
            "duration_source": source,
            "osm_id": row.get("id"),
        }
        oneway = _tag(row.get("oneway"))
        if oneway == "-1":
            graph.add_edge(v, u, **attributes)
        else:
            graph.add_edge(u, v, **attributes)
            if oneway not in {"yes", "true", "1"}:
                graph.add_edge(v, u, **attributes)
        included += 1

    return FerryBuildSummary(
        found=len(projected),
        included=included,
        skipped_access=skipped_access,
        skipped_geometry=skipped_geometry,
        osm_duration=osm_duration,
        estimated_duration=estimated_duration,
    )


def build_osm_drive_graph(
    pbf_path: str | Path,
    *,
    ferry_speed_kph: float = DEFAULT_FERRY_SPEED_KPH,
) -> tuple[nx.MultiDiGraph, FerryBuildSummary]:
    """Build a projected, directed, retain-all driving graph from an OSM PBF."""
    source = Path(pbf_path)
    if not source.is_file():
        raise FileNotFoundError(f"OSM PBF not found: {source}")

    try:
        ox = importlib.import_module("osmnx")
        OSM = importlib.import_module("pyrosm").OSM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install gerrytools[travel-time] to build tables from OSM PBFs") from exc

    osm = OSM(str(source))
    nodes, edges = osm.get_network(network_type="driving", nodes=True)
    if nodes is None or edges is None or len(nodes) == 0 or len(edges) == 0:
        raise ValueError(f"No drivable roads found in {source}")
    graph = OSM.to_graph(
        nodes,
        edges,
        graph_type="networkx",
        network_type="driving",
        retain_all=True,
        osmnx_compatible=True,
    )
    graph = ox.routing.add_edge_speeds(graph)
    graph = ox.routing.add_edge_travel_times(graph)
    graph = ox.project_graph(graph)

    ferry_tags = [
        "route",
        "ferry",
        "motorcar",
        "motor_vehicle",
        "vehicle",
        "access",
        "duration",
        "oneway",
        "name",
    ]
    ferries = osm.get_data_by_custom_criteria(
        custom_filter={"route": ["ferry"]},
        filter_type="keep",
        tags_as_columns=ferry_tags,
        keep_nodes=False,
        keep_ways=True,
        keep_relations=False,
    )
    summary = add_vehicle_ferries(graph, ferries, ferry_speed_kph=ferry_speed_kph)
    graph.graph["source_pbf"] = str(source.resolve())
    graph.graph["ferry_speed_kph"] = float(ferry_speed_kph)
    return graph, summary


def snap_units_to_graph(
    graph: nx.MultiDiGraph,
    unit_points: gpd.GeoDataFrame,
    *,
    id_column: str,
) -> tuple[dict[UnitId, Hashable], dict[UnitId, float]]:
    """Snap representative points to nearest graph nodes and report distances."""
    if id_column not in unit_points.columns:
        raise KeyError(f"Unit ID column not found: {id_column!r}")
    if unit_points.crs is None:
        raise ValueError("unit_points must declare a CRS")
    if graph.graph.get("crs") is None:
        raise ValueError("road graph must declare a CRS")
    ids = unit_points[id_column].to_list()
    if len(ids) == 0 or len(set(ids)) != len(ids):
        raise ValueError("unit_points must contain unique IDs and at least one row")
    projected = unit_points.to_crs(graph.graph["crs"])
    if not all(
        isinstance(geometry, Point) and not geometry.is_empty for geometry in projected.geometry
    ):
        raise ValueError("unit_points geometry must contain nonempty Point values")

    try:
        ox = importlib.import_module("osmnx")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install gerrytools[travel-time] to snap units to an OSM graph") from exc
    nearest, distances = ox.distance.nearest_nodes(
        graph,
        X=projected.geometry.x.to_numpy(),
        Y=projected.geometry.y.to_numpy(),
        return_dist=True,
    )
    nodes = np.atleast_1d(nearest)
    distance_values = np.atleast_1d(distances).astype(float)
    if not np.isfinite(distance_values).all():
        raise ValueError("one or more units could not be snapped to a finite graph distance")
    return dict(zip(ids, nodes.tolist())), dict(zip(ids, distance_values.tolist()))


def compute_directed_travel_times(
    graph: nx.Graph,
    unit_to_node: Mapping[UnitId, Hashable],
    *,
    weight: str = "travel_time",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute a directed unit-to-unit matrix with batched SciPy Dijkstra."""
    if not unit_to_node:
        raise ValueError("unit_to_node must contain at least one unit")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    missing_nodes = [node for node in unit_to_node.values() if node not in graph]
    if missing_nodes:
        raise KeyError(f"Snap nodes not in graph: {missing_nodes[:5]!r}")

    graph_nodes = list(graph.nodes)
    node_index = {node: index for index, node in enumerate(graph_nodes)}
    edge_weights: dict[tuple[int, int], float] = {}
    for u, v, data in graph.edges(data=True):
        if weight not in data:
            raise ValueError(f"Road edge {(u, v)!r} is missing {weight!r}")
        try:
            value = float(data[weight])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Road edge {(u, v)!r} has nonnumeric {weight!r}") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Road edge {(u, v)!r} has invalid {weight!r}: {value!r}")
        key = (node_index[u], node_index[v])
        edge_weights[key] = min(edge_weights.get(key, math.inf), value)
        if not graph.is_directed():
            reverse = (key[1], key[0])
            edge_weights[reverse] = min(edge_weights.get(reverse, math.inf), value)
    if not edge_weights:
        raise ValueError("road graph has no weighted edges")

    rows = [key[0] for key in edge_weights]
    columns = [key[1] for key in edge_weights]
    values = list(edge_weights.values())
    sparse = csr_matrix((values, (rows, columns)), shape=(len(graph_nodes), len(graph_nodes)))

    snap_nodes = list(dict.fromkeys(unit_to_node.values()))
    snap_graph_indices = np.array([node_index[node] for node in snap_nodes], dtype=np.int64)
    snap_matrix = np.full((len(snap_nodes), len(snap_nodes)), np.nan, dtype=np.float32)
    for start in range(0, len(snap_nodes), batch_size):
        stop = min(start + batch_size, len(snap_nodes))
        distances = dijkstra(
            sparse,
            directed=True,
            indices=snap_graph_indices[start:stop],
            return_predecessors=False,
        )
        distances = np.atleast_2d(distances)[:, snap_graph_indices]
        distances[~np.isfinite(distances)] = np.nan
        snap_matrix[start:stop] = distances.astype(np.float32)
    np.fill_diagonal(snap_matrix, 0.0)

    snap_index = {node: index for index, node in enumerate(snap_nodes)}
    unit_snap_indices = np.array([snap_index[node] for node in unit_to_node.values()])
    directed = snap_matrix[np.ix_(unit_snap_indices, unit_snap_indices)]
    np.fill_diagonal(directed, 0.0)
    return directed


def _directional_statistics(directed: np.ndarray) -> tuple[int, float]:
    upper = np.triu_indices_from(directed, k=1)
    forward = directed[upper]
    reverse = directed.T[upper]
    finite = np.isfinite(forward) & np.isfinite(reverse)
    if not finite.any():
        return 0, 0.0
    differences = np.abs(forward[finite] - reverse[finite])
    return int(np.count_nonzero(differences > 1e-8)), float(np.max(differences, initial=0.0))


def _write_json(path: str | Path, document: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_directed_audit(
    path: str | Path,
    unit_ids: Sequence[UnitId],
    directed: np.ndarray,
) -> None:
    serializable_ids: list[str | int] = []
    for unit_id in unit_ids:
        if isinstance(unit_id, str):
            serializable_ids.append(unit_id)
        elif isinstance(unit_id, (int, np.integer)) and not isinstance(unit_id, bool):
            serializable_ids.append(int(unit_id))
        else:
            raise TypeError("Directed-audit unit IDs must be strings or integers")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        np.savez_compressed(
            file,
            unit_ids_json=np.array(json.dumps(serializable_ids, separators=(",", ":"))),
            directed_times=directed.astype(np.float32),
            time_unit=np.array("seconds"),
        )


def build_travel_time_table(
    graph: nx.Graph,
    unit_to_node: Mapping[UnitId, Hashable],
    *,
    snap_distances_m: Mapping[UnitId, float] | None = None,
    ferry: FerryBuildSummary | None = None,
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

    return TravelTimeBuildResult(
        table=symmetric if applicable else None,
        report=report,
        unit_to_node=unit_to_node,
        directed_times=directed if keep_directed else None,
    )


def build_travel_time_table_from_pbf(
    units: pd.DataFrame,
    pbf_path: str | Path,
    *,
    id_column: str,
    latitude_column: str = "INTPTLAT20",
    longitude_column: str = "INTPTLON20",
    ferry_speed_kph: float = DEFAULT_FERRY_SPEED_KPH,
    batch_size: int = 32,
    keep_directed: bool = False,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    directed_audit_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TravelTimeBuildResult:
    """Build a complete table from Census-style units and a local OSM PBF."""
    points = census_representative_points(
        units,
        id_column=id_column,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
    )
    graph, ferry = build_osm_drive_graph(pbf_path, ferry_speed_kph=ferry_speed_kph)
    unit_to_node, snap_distances = snap_units_to_graph(graph, points, id_column=id_column)
    provenance = dict(metadata or {})
    provenance.update(
        {
            "source_pbf": str(Path(pbf_path).resolve()),
            "representative_point": {
                "latitude_column": latitude_column,
                "longitude_column": longitude_column,
            },
            "snapping": "nearest_drive_graph_node",
            "ferry_missing_duration_speed_kph": float(ferry_speed_kph),
            "symmetrization": "mean_of_directions",
        }
    )
    return build_travel_time_table(
        graph,
        unit_to_node,
        snap_distances_m=snap_distances,
        ferry=ferry,
        metadata=provenance,
        batch_size=batch_size,
        keep_directed=keep_directed,
        output_path=output_path,
        report_path=report_path,
        directed_audit_path=directed_audit_path,
    )
