# Travel-time scoring

Travel-time scores use a precomputed, symmetric table of travel times between geographic units.
They evaluate a districting plan without rebuilding or routing the road network.

## Definitions

For a district $D$, let $U_D$ be its units and let $t(i,j)$ be the symmetric travel time between
units $i$ and $j$.

Pairwise travel time uses every unordered pair:

$$
\mathrm{PTT}(D)=\frac{1}{\binom{|U_D|}{2}}
\sum_{\{i,j\}\subseteq U_D} t(i,j)w(i,j).
$$

The pairwise weight is $1$ for `none`, $p_i+p_j$ for `p+p`, and $p_i p_j$ for `p*p`.
Population-weighted modes intentionally retain the number of unit pairs in the denominator rather
than normalizing by the sum of the weights. They are therefore population-scaled scores, not an
average expressed only in the table's time unit.

Centroid travel time first selects the district unit minimizing weighted travel to every unit. The
selected unit is a discrete travel-time medoid: it is not a geometric centroid or an arbitrary
road-network point.

$$
c_D=\underset{c\in U_D}{\mathrm{arg\,min}}\sum_{u\in U_D}t(c,u)v_u.
$$

Then

$$
\mathrm{CTT}(D)=\frac{1}{|U_D|}
\sum_{u\in U_D}t(c_D,u)v_u.
$$

The centroid weight $v_u$ is $1$ for `none` and the destination population $p_u$ for `p`. The
candidate centroid's population does not independently affect its selection. Population-weighted
CTT intentionally retains the number of district units in the denominator rather than normalizing
by total population, so it is also a population-scaled score.

Both families also expose the maximum weighted trip. When all population weights are zero, the
weighted mean and maximum are both zero. A singleton district has no pairwise trip and receives
`NaN` scores.

## Score one plan

The public functions accept the same plan sources as other single-plan GerryTools functions: a
GerryChain `Partition`, a NetworkX graph plus assignment, or a GeoDataFrame plus assignment.

<!-- docs-test: skip -- requires a caller-supplied plan and table -->
```python
from gerrytools.scoring import pairwise_travel_time

district_scores = pairwise_travel_time(
    partition,
    od=travel_time_table,
    population_attr="TOTPOP",
    weight="p+p",
)

plan_mean = district_scores.mean(skipna=False)
worst_district = district_scores.max(skipna=False)
```

`pairwise_travel_time` and `centroid_travel_time` return a Series indexed by district. Use
`travel_time_scores` when the district means, maximum trips, and selected centroids are all
needed.

Unweighted scoring does not require a population attribute. Population-weighted modes require a
finite, nonnegative population for every unit.

## Table requirements

`TravelTimeTable` requires nonnegative symmetric travel times and a zero diagonal. Every included
unit must reach every other included unit. A disconnected table is rejected before scoring so the
set of included trips cannot change with the districting plan.

## Save and load artifacts

Save the scoring table as a compressed NPZ file:

<!-- docs-test: skip -- requires a caller-supplied table -->
```python
from gerrytools.scoring.travel_time import save_travel_time_table

save_travel_time_table(
    "ks_vtd_travel_times.npz",
    travel_time_table,
    time_unit="seconds",
    metadata={"state": "KS", "geography": "vtd2020"},
)
```

The NPZ artifact contains the numeric matrix plus JSON strings for the unit IDs, schema version,
time unit, and user metadata. It contains no pickled Python objects, and
`load_travel_time_table` always opens it with pickle loading disabled. Writes use a temporary file
and replace the destination only after the complete artifact is on disk.

Snapping information is stored separately because it records how the table was built rather than
values needed to score a plan. The conventional companion path replaces `.npz` with
`.snaps.json`:

<!-- docs-test: skip -- requires caller-supplied snapping mappings -->
```python
from gerrytools.scoring.travel_time import (
    default_snapping_path,
    save_snapping_artifact,
)

table_path = "ks_vtd_travel_times.npz"
save_snapping_artifact(
    default_snapping_path(table_path),
    unit_to_node,
    snap_distances_m,
    metadata={"state": "KS", "geography": "vtd2020"},
)
```

Each JSON record contains one geographic unit ID, its selected road-node ID, and the snap distance
in meters. The schema version permits future validation and migration, while the human-readable
format makes individual snaps easy to inspect. `load_snapping_artifact` validates unique units,
matching fields, serializable identifiers, and finite nonnegative distances.

## Building a table from representative points and OSM

The optional table builder requires OSMnx and Pyrosm. Install the extra from a
checkout with `uv sync --extra travel-time`, or from a published package with
`pip install "gerrytools[travel-time]"`.

Pyrosm is the local parser for OpenStreetMap's binary `.osm.pbf` extract. In this
workflow it reads the driving nodes and ways, extracts tagged ferry features, and
converts the result into an OSMnx-compatible NetworkX graph. OSMnx then adds road
speeds and edge travel times, projects the graph for distance calculations, and
supports nearest-node operations. Pyrosm and OSMnx are needed only to build a
table from an OSM extract; scoring an existing `TravelTimeTable` does not require
either dependency.

The implemented unit-to-unit measure is shortest-path travel time between the
selected nodes of the retained road network. The node-selection rule is part of
the methodology and is recorded in the artifact metadata:

- **Point-only:** the retained road node nearest the supplied representative
  point.
- **Polygon-aware:** the retained road node inside the unit and nearest its
  representative point. If no retained node lies inside the unit, use a node
  nearest an exterior boundary ring; polygon holes are excluded.

Canonical workflow:

1. Create or supply a point GeoDataFrame of representative locations
   (`census_representative_points(...)` for Census `INTPTLAT20` / `INTPTLON20`,
   or any other Point frame with a declared CRS);
2. build a directed driving graph from a local OSM PBF, retain ordinary service
   connectors, add eligible vehicle ferries, then retain the largest strongly
   connected component;
3. select a retained, snap-eligible road node using the documented point-only or
   polygon-aware
   rule and record representative-point-to-node distance;
4. compute directed shortest paths, average opposite directions, and publish a
   table only when every snapped-node pair is mutually reachable.

One-shot helper:

<!-- docs-test: skip -- requires user-supplied Census geography and OSM PBF files -->
```python
import geopandas as gpd
import gerrytools.scoring.travel_time as tt

units = gpd.read_file("tl_2020_20_vtd20.shp")
points = tt.census_representative_points(units, id_column="GEOID20")
result = tt.build_travel_time_table_from_pbf(
    points,
    "kansas-latest.osm.pbf",
    id_column="GEOID20",
    unit_polygons=units,
    output_path="ks_vtd2020_travel_times.npz",
    report_path="ks_vtd2020_travel_times.report.json",
    # Optional directed audit retaining both route directions:
    directed_audit_path="ks_vtd2020_directed_audit.npz",
    metadata={"state": "KS", "geography": "2020 VTD"},
)

if not result.is_applicable:
    print("Travel time is not applicable:", result.report.connectivity_component_sizes)
else:
    print("Max snap distance (m):", result.report.max_snap_distance_m)
```

When `output_path` is supplied and snapping distances are available, the builder also writes the
versioned companion file returned by `default_snapping_path(output_path)`. The table remains the
scoring input; the JSON companion preserves the selected road node and snap distance for review.

`build_travel_time_table_from_pbf(...)` requires a Point GeoDataFrame and accepts
an optional polygon GeoDataFrame with matching unit IDs. With polygons supplied,
the builder chooses the contained retained-network node nearest the representative
point. If no retained node lies inside a unit, it chooses the node nearest any
exterior ring; polygon holes are excluded from this fallback. Reported snap
distances remain representative-point-to-node distances. The builder does not
calculate centroids or repair boundaries. Users without Census internal-point
columns must supply their own representative points.

The road filter starts from Pyrosm's `driving+service` network and removes ways
tagged with service subtypes `alley`, `driveway`, `parking`, `parking_aisle`,
`private`, or `emergency_access`. Standard OSM access restrictions still exclude
private and motor-vehicle-prohibited ways. Of the remaining unclassified service
roads, the builder keeps only connected groups whose attachment nodes span two or
more strongly connected components of the ordinary driving graph. A retained SCC bridge may carry
a route, but a node whose incident road edges are all service roads cannot be selected as a unit's
snap target. This lets divided entrances and similar
connectors preserve genuine road connectivity without snapping units directly to
parking aisles or other service-only road fragments. OSM tagging errors can still
affect the result, so long snaps and removed SCC components remain review items.

Vehicle-ferry inclusion is conservative. Access is resolved from the most specific recognized tag
in this order: `motorcar`, `motor_vehicle`, `vehicle`, then `access`. A denial at a more specific
level overrides a broader permission. The `ferry=*` road classification is used only when those
access tags do not decide. Explicit denials and ambiguous passenger-only routes are excluded and
counted in the report. Each included ferry edge records which tag made the decision. Ferries are
added *before* the largest strongly
connected component is selected, so a correctly mapped vehicle ferry can keep an
island in the principal network. Filtering occurs only in memory and does not
modify the source PBF. A state-only PBF constrains routes to roads in that extract;
use a broader PBF when routes that leave and re-enter the state must be represented.

Road speeds are prepared with OSMnx and a configurable
``fallback_speed_kph`` (48 kph by default). If an edge still has a non-positive
or non-finite speed, the builder keeps the edge, assigns that documented
fallback, and recomputes its travel time. Missing or invalid edge lengths remain
an error rather than being silently dropped. The build report's ``speed_repair``
section records the number of inspected and repaired edges and the fallback used.

The build report records ferry counts, SCC filter counts, snap-distance
statistics, directional asymmetry, and final connectivity. The builder warns when
the connectivity filter removes road nodes or edges. Applicability means the
selected snapped nodes are mutually reachable; it does **not** prove that every
representative point was geographically near its snapped node. Inspect
`result.report.mean_snap_distance_m`, `result.report.max_snap_distance_m`, and
`result.snap_distances_m` before trusting a table.

For road-network review, pass ``include_component_audit=True`` to
``build_osm_drive_graph``. The returned retained graph then stores a compact
GeoDataFrame of removed edge geometry and selected OSM attributes at
``graph.graph["removed_component_edges"]``. This option does not alter the
retained graph or travel times, but it is disabled by default because preserving
discarded-edge evidence increases memory use on large extracts.

If snapped nodes remain disconnected, `result.table` is `None`, the report status
is `not_applicable`, and no scoring artifact is written. The optional directed
audit may still be retained for diagnosis.

The published table is dense, so storage and routing work grow quadratically with
the number of units. A single `float32` matrix for $n$ units requires $4n^2$ bytes,
and construction needs additional working memory for directed and symmetric copies.
Estimate the matrix size and test a deterministic subset before launching a large
build.

### Limitations

- Node snapping is an approximation of unit-to-unit drive time. Polygon-aware
  snapping should be used when trustworthy unit boundaries are available.
- A road segment may cross a small unit without containing an OSM graph node; this
  is especially relevant for Census blocks. Edge-based snapping may be preferable
  for very small units, but is not included in this version.
- Largest-SCC filtering can cause island or secondary-component points to snap far
  away onto the retained network. Review snap distances carefully.
- GerryTools validates but does not repair polygon boundaries. Source-boundary
  repair is a separate, reviewable data-preparation decision.

See the {doc}`scoring API <../../api/scoring>` for full signatures under
`gerrytools.scoring.travel_time`.
