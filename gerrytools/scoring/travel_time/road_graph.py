"""OSM road-network preparation for travel-time table construction."""

from __future__ import annotations

import importlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, TypedDict, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

DEFAULT_FERRY_SPEED_KPH = 20.0
DEFAULT_FALLBACK_SPEED_KPH = 48.0
_OSM_DRIVE_NETWORK_TYPE = "driving+service"
_ROAD_NETWORK_POLICY = "driving+scc_bridge_service"

_EXCLUDED_SERVICE_CLASSES = frozenset(
    {"alley", "driveway", "emergency_access", "parking", "parking_aisle", "private"}
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

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable summary."""
        return {
            "found": self.found,
            "included": self.included,
            "skipped_access": self.skipped_access,
            "skipped_geometry": self.skipped_geometry,
            "osm_duration": self.osm_duration,
            "estimated_duration": self.estimated_duration,
        }


@dataclass(frozen=True, slots=True)
class SpeedRepairSummary:
    """Counts of edges whose free-flow speed or travel time needed repair.

    Repairs replace non-positive or non-finite ``speed_kph`` values with a
    documented fallback and recompute ``travel_time``. Edges are never dropped.
    """

    inspected: int = 0
    repaired_speed: int = 0
    repaired_travel_time: int = 0
    fallback_speed_kph: float = DEFAULT_FALLBACK_SPEED_KPH

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable summary."""
        return {
            "inspected": self.inspected,
            "repaired_speed": self.repaired_speed,
            "repaired_travel_time": self.repaired_travel_time,
            "fallback_speed_kph": self.fallback_speed_kph,
        }


@dataclass(frozen=True, slots=True)
class FerryAccessDecision:
    """Result of applying OSM's motor-vehicle access hierarchy to a ferry way.

    Args:
        allowed (bool): Whether the ferry admits motor vehicles.
        source (str): OSM tag whose value decided access, or ``"undetermined"``.
        value (str | None): Normalized deciding tag value when present.
    """

    allowed: bool
    source: str
    value: str | None


@dataclass(frozen=True, slots=True)
class FerryAccessPolicy:
    """Structured OSM motor-vehicle access policy for ferry ways.

    Args:
        keys (tuple[str, ...]): Access tags from most to least specific.
        allow (frozenset[str]): Values that explicitly permit access.
        deny (frozenset[str]): Values that explicitly deny access.
        ferry_classes (frozenset[str]): Drivable ``ferry=*`` fallback classes.
    """

    keys: tuple[str, ...]
    allow: frozenset[str]
    deny: frozenset[str]
    ferry_classes: frozenset[str]

    def decide(self, tags: Mapping[str, Any]) -> FerryAccessDecision:
        """Resolve access using the first recognized tag in hierarchy order.

        Args:
            tags (Mapping[str, Any]): OSM attributes for one ferry way.

        Returns:
            FerryAccessDecision: Permission plus the tag and value that decided it.
        """
        for source in self.keys:
            values = _tag_values(tags.get(source))
            denied = sorted(values & self.deny)
            if denied:
                return FerryAccessDecision(False, source, denied[0])
            allowed = sorted(values & self.allow)
            if allowed:
                return FerryAccessDecision(True, source, allowed[0])

        raw_ferry_classes = _tag_values(tags.get("ferry"))
        ferry_classes = sorted(raw_ferry_classes & self.ferry_classes)
        if ferry_classes:
            return FerryAccessDecision(True, "ferry", ferry_classes[0])
        unrecognized = sorted(raw_ferry_classes)
        return FerryAccessDecision(
            False,
            "undetermined",
            unrecognized[0] if unrecognized else None,
        )


DRIVING_FERRY_ACCESS = FerryAccessPolicy(
    keys=("motorcar", "motor_vehicle", "vehicle", "access"),
    allow=frozenset({"yes", "designated", "permissive", "official", "destination"}),
    deny=frozenset({"no", "private"}),
    ferry_classes=frozenset(
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
    ),
)


class ConnectivityFilterSummary(TypedDict):
    """Fixed-shape summary of the principal-SCC filter."""

    method: str
    original_strong_component_count: int
    original_node_count: int
    original_edge_count: int
    retained_node_count: int
    retained_edge_count: int
    removed_node_count: int
    removed_edge_count: int


def _parse_iso_time_duration(text: str) -> float | None:
    """Parse the ISO-8601 time-only subset ``PT[nH][nM][nS]``."""
    normalized = text.upper()
    if not normalized.startswith("PT"):
        return None
    body = normalized[2:]
    if not body:
        return None

    multipliers = {"H": 3600.0, "M": 60.0, "S": 1.0}
    order = {"H": 0, "M": 1, "S": 2}
    total = 0.0
    number = ""
    previous_order = -1
    for character in body:
        if character.isdigit() or character == ".":
            number += character
            continue
        if character not in multipliers or not number:
            return None
        component_order = order[character]
        if component_order <= previous_order:
            return None
        try:
            component = float(number)
        except ValueError:
            return None
        if not math.isfinite(component) or component < 0:
            return None
        total += component * multipliers[character]
        number = ""
        previous_order = component_order

    if number or previous_order < 0:
        return None
    return total


def parse_osm_duration_seconds(value: Any) -> float | None:
    """Parse an OSM ``duration`` value, returning seconds when valid.

    Supported forms are ``M``, ``H:MM``, ``H:MM:SS``, and ISO-8601 time
    durations such as ``PT1H30M``. A bare number follows OSM convention and is
    interpreted as minutes.

    Args:
        value (Any): Raw OSM duration value.

    Returns:
        float | None: Duration in seconds, or ``None`` when the value is missing or invalid.
    """
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.replace(".", "", 1).isdigit():
        minutes = float(text)
        return minutes * 60.0 if minutes >= 0 else None
    iso_seconds = _parse_iso_time_duration(text)
    if iso_seconds is not None:
        return iso_seconds
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


def _tag_values(value: Any) -> set[str]:
    """Return normalized values from scalar, list, or semicolon-delimited OSM tags."""
    if value is None or value is pd.NA:
        return set()
    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        pass
    values = value if isinstance(value, (list, tuple, set, np.ndarray)) else [value]
    normalized: set[str] = set()
    for item in values:
        normalized.update(part.strip().lower() for part in str(item).split(";") if part.strip())
    return normalized


def _directed_road_endpoints(edges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Expand raw OSM segments into directed endpoints for SCC classification."""
    sources = edges["u"].to_numpy(dtype=np.int64, copy=False)
    targets = edges["v"].to_numpy(dtype=np.int64, copy=False)
    if sources.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty
    oneway = (
        edges["oneway"].map(_tag).to_numpy(dtype=object)
        if "oneway" in edges.columns
        else np.full(sources.size, None, dtype=object)
    )
    junction = (
        edges["junction"].map(_tag).to_numpy(dtype=object)
        if "junction" in edges.columns
        else np.full(sources.size, None, dtype=object)
    )
    one_way = np.isin(oneway, ["yes", "true", "1", "-1", "t", "f"]) | (junction == "roundabout")
    against = np.isin(oneway, ["-1", "t"])
    along = one_way & ~against
    reverse = one_way & against
    both = ~one_way
    return (
        np.concatenate([sources[along], targets[reverse], sources[both], targets[both]]),
        np.concatenate([targets[along], sources[reverse], targets[both], sources[both]]),
    )


def _filter_service_edges(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Keep unclassified service components only when they span base-road SCCs."""
    stats = {
        "excluded_subtype_edge_count": 0,
        "excluded_nonbridge_edge_count": 0,
        "retained_scc_bridge_edge_count": 0,
    }
    required = {"u", "v", "highway"}
    if not required.issubset(edges.columns):
        return nodes, edges, stats

    if "service" in edges.columns:
        excluded_subtype = (
            edges["service"]
            .map(lambda value: bool(_tag_values(value) & _EXCLUDED_SERVICE_CLASSES))
            .to_numpy(dtype=bool)
        )
    else:
        excluded_subtype = np.zeros(len(edges), dtype=bool)
    stats["excluded_subtype_edge_count"] = int(excluded_subtype.sum())
    permitted = edges.loc[~excluded_subtype].copy()

    service_mask = (
        permitted["highway"].map(lambda value: "service" in _tag_values(value)).to_numpy(dtype=bool)
    )
    service_positions = np.flatnonzero(service_mask)
    keep = ~service_mask
    if service_positions.size:
        base = permitted.loc[~service_mask]
        base_sources, base_targets = _directed_road_endpoints(base)
        if base_sources.size:
            base_nodes, base_inverse = np.unique(
                np.concatenate([base_sources, base_targets]), return_inverse=True
            )
            base_matrix = csr_matrix(
                (
                    np.ones(base_sources.size, dtype=np.uint8),
                    (base_inverse[: base_sources.size], base_inverse[base_sources.size :]),
                ),
                shape=(base_nodes.size, base_nodes.size),
            )
            _base_count, base_labels = connected_components(
                base_matrix, directed=True, connection="strong", return_labels=True
            )

            service = permitted.iloc[service_positions]
            service_nodes, service_inverse = np.unique(
                np.concatenate([service["u"].to_numpy(), service["v"].to_numpy()]),
                return_inverse=True,
            )
            edge_count = len(service)
            service_matrix = csr_matrix(
                (
                    np.ones(edge_count * 2, dtype=np.uint8),
                    (
                        np.concatenate(
                            [service_inverse[:edge_count], service_inverse[edge_count:]]
                        ),
                        np.concatenate(
                            [service_inverse[edge_count:], service_inverse[:edge_count]]
                        ),
                    ),
                ),
                shape=(service_nodes.size, service_nodes.size),
            )
            service_count, service_labels = connected_components(
                service_matrix, directed=False, return_labels=True
            )
            attachment_mask = np.isin(service_nodes, base_nodes)
            attachment_nodes = service_nodes[attachment_mask]
            attachment_base_labels = base_labels[np.searchsorted(base_nodes, attachment_nodes)]
            component_pairs = np.unique(
                np.column_stack([service_labels[attachment_mask], attachment_base_labels]),
                axis=0,
            )
            spanned_scc_count = np.bincount(
                component_pairs[:, 0] if component_pairs.size else np.array([], dtype=int),
                minlength=service_count,
            )
            keep_service = spanned_scc_count[service_labels[service_inverse[:edge_count]]] >= 2
        else:
            keep_service = np.zeros(service_positions.size, dtype=bool)
        keep[service_positions] = keep_service
        stats["retained_scc_bridge_edge_count"] = int(keep_service.sum())
        stats["excluded_nonbridge_edge_count"] = int((~keep_service).sum())

    retained = permitted.iloc[np.flatnonzero(keep)].copy()
    if "id" in nodes.columns:
        used = np.unique(np.concatenate([retained["u"].to_numpy(), retained["v"].to_numpy()]))
        nodes = nodes.loc[nodes["id"].isin(used.tolist())].copy()
    return nodes, retained, stats


def _service_only_nodes(graph: nx.MultiDiGraph) -> frozenset[Hashable]:
    """Return retained nodes whose incident road edges are all service roads."""
    candidates: set[Hashable] = set()
    for u, v, data in graph.edges(data=True):
        if "service" in _tag_values(data.get("highway")):
            candidates.update((u, v))
    if not candidates:
        return frozenset()
    for u, v, data in graph.edges(data=True):
        if "service" not in _tag_values(data.get("highway")):
            candidates.discard(u)
            candidates.discard(v)
    return frozenset(candidates)


def _ferry_access_decision(row: Mapping[str, Any]) -> FerryAccessDecision:
    """Resolve access from the most specific applicable OSM tag."""
    return DRIVING_FERRY_ACCESS.decide(row)


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


def _edge_geometry(
    graph: nx.MultiDiGraph, u: Hashable, v: Hashable, data: Mapping[str, Any]
) -> LineString:
    """Return an edge geometry, constructing one from endpoint coordinates."""
    geometry = data.get("geometry")
    if isinstance(geometry, LineString) and not geometry.is_empty:
        return geometry
    if isinstance(geometry, MultiLineString):
        merged = linemerge(geometry)
        if isinstance(merged, LineString) and not merged.is_empty:
            return merged
    return LineString(
        [
            (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
            (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
        ]
    )


def _retain_principal_strong_component(
    graph: nx.MultiDiGraph,
    *,
    include_component_audit: bool = False,
) -> tuple[ConnectivityFilterSummary, gpd.GeoDataFrame | None]:
    """Retain the largest SCC and optionally capture removed edge evidence."""
    original_node_count = graph.number_of_nodes()
    original_edge_count = graph.number_of_edges()
    principal_nodes: set[Hashable] = set()
    strong_component_count = 0
    for component in nx.strongly_connected_components(graph):
        strong_component_count += 1
        if len(component) > len(principal_nodes):
            principal_nodes = component
    if not principal_nodes:
        raise ValueError("No strongly connected driving component found")

    removed_edges: gpd.GeoDataFrame | None = None
    if include_component_audit:
        records: list[dict[str, Any]] = []
        for u, v, key, data in graph.edges(keys=True, data=True):
            if u in principal_nodes and v in principal_nodes:
                continue
            records.append(
                {
                    "u": u,
                    "v": v,
                    "key": key,
                    "highway": data.get("highway"),
                    "name": data.get("name"),
                    "route": data.get("route"),
                    "access": data.get("access"),
                    "motor_vehicle": data.get("motor_vehicle"),
                    "speed_kph": data.get("speed_kph"),
                    "speed_imputed": bool(data.get("speed_imputed", False)),
                    "travel_time": data.get("travel_time"),
                    "length": data.get("length"),
                    "geometry": _edge_geometry(graph, u, v, data),
                }
            )
        columns = [
            "u",
            "v",
            "key",
            "highway",
            "name",
            "route",
            "access",
            "motor_vehicle",
            "speed_kph",
            "speed_imputed",
            "travel_time",
            "length",
            "geometry",
        ]
        removed_edges = gpd.GeoDataFrame(
            records,
            columns=columns,
            geometry="geometry",
            crs=graph.graph.get("crs"),
        )

    removed_nodes = [node for node in graph if node not in principal_nodes]
    graph.remove_nodes_from(removed_nodes)
    connectivity_filter: ConnectivityFilterSummary = {
        "method": "largest_strongly_connected_component",
        "original_strong_component_count": strong_component_count,
        "original_node_count": original_node_count,
        "original_edge_count": original_edge_count,
        "retained_node_count": graph.number_of_nodes(),
        "retained_edge_count": graph.number_of_edges(),
        "removed_node_count": original_node_count - graph.number_of_nodes(),
        "removed_edge_count": original_edge_count - graph.number_of_edges(),
    }
    return connectivity_filter, removed_edges


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

    Args:
        graph (nx.MultiDiGraph): Projected road graph to modify in place.
        ferry_features (gpd.GeoDataFrame | None): OSM ferry-way features.
        ferry_speed_kph (float): Fallback ferry speed when OSM duration is unavailable.

    Returns:
        FerryBuildSummary: Counts describing included and rejected ferry ways.

    Raises:
        ValueError: If the fallback speed or required coordinate reference systems are invalid.
        ImportError: If the optional OSMnx dependency is unavailable.
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
    candidates: list[tuple[pd.Series, LineString, FerryAccessDecision]] = []
    skipped_access = 0
    skipped_geometry = 0
    for _, row in projected.iterrows():
        access = _ferry_access_decision(cast(Mapping[str, Any], row))
        if not access.allowed:
            skipped_access += 1
            continue
        line = _as_line(row.geometry)
        if line is None or line.length <= 0:
            skipped_geometry += 1
            continue
        candidates.append((row, line, access))

    if not candidates:
        return FerryBuildSummary(
            found=len(projected),
            skipped_access=skipped_access,
            skipped_geometry=skipped_geometry,
        )

    endpoints = [(line.coords[0], line.coords[-1]) for _, line, _ in candidates]
    xs = [coordinate[0] for pair in endpoints for coordinate in pair]
    ys = [coordinate[1] for pair in endpoints for coordinate in pair]
    nearest = np.atleast_1d(ox.distance.nearest_nodes(graph, X=xs, Y=ys))

    included = 0
    osm_duration = 0
    estimated_duration = 0
    for index, ((row, line, access), (start, end)) in enumerate(zip(candidates, endpoints)):
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
            "ferry_access_source": access.source,
            "ferry_access_value": access.value,
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


def _repair_nonpositive_edge_speeds(
    graph: nx.MultiDiGraph,
    *,
    fallback_speed_kph: float = DEFAULT_FALLBACK_SPEED_KPH,
) -> SpeedRepairSummary:
    """Replace non-positive/non-finite speeds and recompute affected travel times.

    OSMnx sets ``travel_time = length_km / (speed_kph / 3600)``. A cleaned OSM
    ``maxspeed`` of ``0`` therefore yields ``travel_time = inf``, which later
    fails directed routing. This keeps every edge, imputes a documented fallback
    speed, and records how many edges were touched.

    Args:
        graph (nx.MultiDiGraph): Road graph whose edge attributes are modified in place.
        fallback_speed_kph (float): Positive speed assigned to invalid edges.

    Returns:
        SpeedRepairSummary: Counts and fallback speed for the completed repair.

    Raises:
        ValueError: If the fallback is invalid or an edge remains unroutable after repair.
    """
    if not np.isfinite(fallback_speed_kph) or fallback_speed_kph <= 0:
        raise ValueError("fallback_speed_kph must be finite and positive")

    inspected = 0
    repaired_speed = 0
    repaired_travel_time = 0
    remaining_invalid = 0
    for _, _, _, data in graph.edges(keys=True, data=True):
        inspected += 1
        speed = data.get("speed_kph")
        try:
            speed_value = float(speed) if speed is not None else math.nan
        except (TypeError, ValueError):
            speed_value = math.nan
        if not np.isfinite(speed_value) or speed_value <= 0:
            data["speed_kph"] = float(fallback_speed_kph)
            data["speed_imputed"] = True
            data["speed_imputation_reason"] = "nonpositive_or_nonfinite_speed_kph"
            repaired_speed += 1
            speed_value = float(fallback_speed_kph)

        length = data.get("length")
        try:
            length_value = float(length) if length is not None else math.nan
        except (TypeError, ValueError):
            length_value = math.nan
        travel_time = data.get("travel_time")
        try:
            travel_value = float(travel_time) if travel_time is not None else math.nan
        except (TypeError, ValueError):
            travel_value = math.nan
        if np.isfinite(length_value) and (
            not np.isfinite(travel_value) or travel_value < 0 or data.get("speed_imputed")
        ):
            data["travel_time"] = float(length_value) / (speed_value * 1000.0 / 3600.0)
            repaired_travel_time += 1
        try:
            raw_travel_time = data.get("travel_time")
            travel_value = float(raw_travel_time) if raw_travel_time is not None else math.nan
        except (TypeError, ValueError):
            travel_value = math.nan
        if not np.isfinite(travel_value) or travel_value < 0:
            remaining_invalid += 1
    if remaining_invalid:
        raise ValueError(
            f"{remaining_invalid} road edge(s) still have invalid travel_time after "
            f"imputing fallback_speed_kph={fallback_speed_kph}. Edges were not dropped; "
            "inspect length/speed attributes before retrying."
        )

    summary = SpeedRepairSummary(
        inspected=inspected,
        repaired_speed=repaired_speed,
        repaired_travel_time=repaired_travel_time,
        fallback_speed_kph=float(fallback_speed_kph),
    )
    graph.graph["speed_repair"] = {
        **summary.to_dict(),
        "policy": "impute_fallback_speed_keep_edge",
    }
    return summary


def _open_pyrosm_reader(source: Path, OSM: Any) -> Any:
    """Open a PBF reader, preferring ``keep_metadata=False`` when supported.

    Some state extracts (notably Connecticut under pyrosm 0.17 in-memory) fail in
    ``convert_to_arrays_and_drop_empty`` when metadata integer fields are
    ``None``. Dropping metadata does not change highway topology used for
    routing. ``engine='out_of_core'`` is not used here because this builder still
    requires ``get_network(nodes=True)``, which that engine rejects.
    """
    try:
        return OSM(str(source), keep_metadata=False)
    except TypeError as exc:
        message = str(exc)
        if "keep_metadata" not in message or not any(
            phrase in message
            for phrase in ("unexpected keyword", "invalid keyword", "keyword argument")
        ):
            raise
        return OSM(str(source))


def build_osm_drive_graph(
    pbf_path: str | Path,
    *,
    ferry_speed_kph: float = DEFAULT_FERRY_SPEED_KPH,
    fallback_speed_kph: float = DEFAULT_FALLBACK_SPEED_KPH,
    include_component_audit: bool = False,
) -> tuple[nx.MultiDiGraph, FerryBuildSummary]:
    """Build the principal mutually routable driving graph from an OSM PBF.

    The graph starts from pyrosm's ``driving+service`` filter, removes
    alleys, driveways, parking access, private service, and emergency-only ways,
    then keeps only service components whose attachment nodes span two or more
    ordinary-road strong components. These bridge links may carry routes but service-only nodes are excluded
    from unit snapping. All selected OSM components are retained until eligible
    vehicle ferries have
    been added. The returned graph is then restricted to its largest strongly
    connected component so that representative points cannot snap to isolated
    road fragments or directionally inaccessible components. When
    ``include_component_audit`` is true, a geometry-and-attribute table for the
    removed edges is stored at ``graph.graph["removed_component_edges"]`` for
    diagnostics. Normal builds avoid that additional memory.

    Args:
        pbf_path (str | Path): Local OpenStreetMap ``.osm.pbf`` extract.
        ferry_speed_kph (float): Fallback speed for ferries without durations.
        fallback_speed_kph (float): Fallback speed for invalid road-edge speeds.
        include_component_audit (bool): Preserve removed-edge evidence on the graph.

    Returns:
        tuple[nx.MultiDiGraph, FerryBuildSummary]: Retained routable graph and ferry summary.

    Raises:
        FileNotFoundError: If ``pbf_path`` does not exist.
        ImportError: If the optional Pyrosm or OSMnx dependency is unavailable.
        ValueError: If no usable driving network can be constructed.
        RuntimeError: If Pyrosm fails while parsing the driving network.
    """
    source = Path(pbf_path)
    if not source.is_file():
        raise FileNotFoundError(f"OSM PBF not found: {source}")

    try:
        ox = importlib.import_module("osmnx")
        OSM = importlib.import_module("pyrosm").OSM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install gerrytools[travel-time] to build tables from OSM PBFs") from exc

    osm = _open_pyrosm_reader(source, OSM)
    try:
        nodes, edges = osm.get_network(network_type=_OSM_DRIVE_NETWORK_TYPE, nodes=True)
    except TypeError as exc:
        raise RuntimeError(
            f"pyrosm failed while parsing the driving network from {source}: {exc}. "
            "This matches a known in-memory parser failure on some extracts "
            "(e.g. Connecticut) when integer metadata fields are None. "
            "The builder already opens the reader with keep_metadata=False when "
            "supported. Do not switch to engine='out_of_core' without changing "
            "graph construction: that engine rejects get_network(nodes=True)."
        ) from exc
    if nodes is None or edges is None or len(nodes) == 0 or len(edges) == 0:
        raise ValueError(f"No drivable roads found in {source}")
    nodes, edges, service_filter = _filter_service_edges(nodes, edges)
    if len(nodes) == 0 or len(edges) == 0:
        raise ValueError(f"No drivable roads found in {source} after service-road filtering")
    graph = OSM.to_graph(
        nodes,
        edges,
        graph_type="networkx",
        network_type=_OSM_DRIVE_NETWORK_TYPE,
        retain_all=True,
        osmnx_compatible=True,
    )
    del nodes, edges
    graph = ox.routing.add_edge_speeds(graph, fallback=float(fallback_speed_kph))
    graph = ox.routing.add_edge_travel_times(graph)
    speed_repair = _repair_nonpositive_edge_speeds(graph, fallback_speed_kph=fallback_speed_kph)
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

    connectivity_filter, removed_component_edges = _retain_principal_strong_component(
        graph,
        include_component_audit=include_component_audit,
    )
    snap_ineligible_nodes = _service_only_nodes(graph)
    graph.graph["source_pbf"] = str(source.resolve())
    graph.graph["ferry_speed_kph"] = float(ferry_speed_kph)
    graph.graph["fallback_speed_kph"] = float(fallback_speed_kph)
    graph.graph["network_type"] = _ROAD_NETWORK_POLICY
    graph.graph["pyrosm_network_type"] = _OSM_DRIVE_NETWORK_TYPE
    graph.graph["service_filter"] = {
        "excluded_classes": sorted(_EXCLUDED_SERVICE_CLASSES),
        **service_filter,
        "service_only_snap_ineligible_node_count": len(snap_ineligible_nodes),
    }
    graph.graph["snap_ineligible_nodes"] = snap_ineligible_nodes
    graph.graph["connectivity_filter"] = connectivity_filter
    if removed_component_edges is not None:
        graph.graph["removed_component_edges"] = removed_component_edges
    graph.graph["speed_repair"] = {
        **speed_repair.to_dict(),
        "policy": "impute_fallback_speed_keep_edge",
    }
    if connectivity_filter["removed_node_count"]:
        warnings.warn(
            "The principal-road connectivity filter removed "
            f"{connectivity_filter['removed_node_count']} of "
            f"{connectivity_filter['original_node_count']} road nodes and "
            f"{connectivity_filter['removed_edge_count']} of "
            f"{connectivity_filter['original_edge_count']} edges. "
            "Eligible vehicle ferries were added before this filter. Inspect snap distances "
            "for islands or other geographies whose representative points may have snapped "
            "onto the retained component from far away.",
            UserWarning,
            stacklevel=2,
        )
    return graph, summary
