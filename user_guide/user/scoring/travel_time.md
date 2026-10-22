# Travel-time scores (PTT / CTT)

Measure how long it takes to drive *within* a district using a finished
unit-to-unit drive-time table. These scores are **not** part of the compiled
`PlanEvaluator` engine: you build (or load) a `TravelTimeTable`, then call the
convenience functions or reusable metric objects.

Python naming follows the rest of GerryTools: modules and functions use
`snake_case` (`travel_time`, `build_travel_time_table`), while classes and result
types use `PascalCase` (`TravelTimeTable`, `TravelTimeBuildReport`). In prose,
“travel time” is the noun and “travel-time” is used as a compound adjective.

Routes may leave the district because shortest paths use the full statewide
network. Every unit included in the analysis must be mutually reachable through
that network. Scoring rejects a disconnected OD table rather than silently
removing unreachable trips from a district denominator.

## Statewide connectivity

Validate a table once after building or loading it by calling
`od.validate_connected()`.

`validate_connected()` raises `ValueError` when any unit pair has a non-finite
travel time. The error reports mutually reachable component sizes and example
unreachable pairs. `od.connectivity_components()` provides the component membership
for more detailed diagnostics. A unit universe that remains genuinely disconnected
after checking legitimate road and ferry links is not applicable for this score.

Resolve a disconnected table before scoring. Appropriate resolutions may include
repairing the road network, adding legitimate ferry connections, or applying a
documented unit exclusion to the analysis universe. Any exclusion must be decided
before plan scoring and applied identically to every plan. District scoring never
silently drops unreachable pairs.

## Weight modes

PTT connects two population-bearing units and supports three pairwise modes:

| PTT mode | Trip weight | Interpretation |
| --- | --- | --- |
| `none` | `1` | Each unit pair counts equally |
| `p+p` | `pop_a + pop_b` | Population-involved pair weight |
| `p*p` | `pop_a * pop_b` | Person-pair weight |

CTT connects a selected center location to one destination unit and supports two
modes:

| CTT mode | Trip weight | Interpretation |
| --- | --- | --- |
| `none` | `1` | Each destination unit counts equally |
| `p` | `pop_u` | Destination-population weight |

The centroid unit represents a location, not a population endpoint. Its population
never enters centroid selection or CTT scoring. In particular, CTT does not accept
`p+p` or `p*p`.

Whenever a population-weighted mode is selected, every participating unit must
have a numeric, finite, nonnegative population. Missing values, `NaN`, infinity,
negative values, and nonnumeric values raise a unit-specific `ValueError` before
scoring. Zero population is valid.

District scores use $\sum (t w) / (\#\ \mathrm{trips})$, so population weights
stay in the numerator. For PTT, the denominator is the number of unordered unit
pairs, $n(n-1)/2$. For CTT, it is the number of centroid-to-unit trips, including
the zero-time trip from the centroid unit to itself. PTT units are time (`none`),
people-time (`p+p`), or people²-time (`p*p`); CTT units are time (`none`) or
people-time (`p`). Weighted maximums are $\max(tw)$ and have the same units.

More explicitly, for district units $D$:

$$
\operatorname{PTT}_{\mathrm{mean}}(D)
= \frac{1}{\binom{|D|}{2}}\sum_{\{i,j\}\subset D} t(i,j)w(i,j),
$$

$$
\operatorname{PTT}_{\mathrm{max}}(D)
= \max_{\{i,j\}\subset D} t(i,j)w(i,j).
$$

After choosing centroid unit $c$:

$$
\operatorname{CTT}_{\mathrm{mean}}(D)
= \frac{1}{|D|}\sum_{u\in D} t(c,u)w(c,u),
$$

$$
\operatorname{CTT}_{\mathrm{max}}(D)
= \max_{u\in D} t(c,u)w(c,u).
$$


## Symmetric OD tables

Road routing is directed because one-way roads and other restrictions can make the
two directions differ. The score has one mandatory definition for an unordered
unit pair:

$$
t(i,j)=\frac{t_{i\rightarrow j}+t_{j\rightarrow i}}{2}.
$$

`TravelTimeTable.from_directed_matrix(unit_ids, directed_times)` performs this
averaging. A missing direction remains missing, so connectivity validation catches
it. `TravelTimeTable.from_matrix(...)` accepts only an already-symmetric matrix;
it rejects asymmetric input rather than choosing a policy silently.

The normal published artifact is therefore the symmetric scoring table. A builder
may also preserve the directed routing matrix as an optional audit artifact, but
the scoring API never uses it directly.

## Centroid (CTT)

- `none`: choose unit $c$ minimizing $\sum_u t(c,u)$
- `p`: choose unit $c$ minimizing $\sum_u t(c,u)\,p_u$

Every unit is eligible to represent the center, including a zero-population unit.
Only destination population $p_u$ enters the population-weighted definition.

## Plan summaries

Each PTT or CTT family exposes four plan-level summaries:

| `plan_agg` | District statistic | Plan statistic |
| --- | --- | --- |
| `mean` | District mean | Mean across districts |
| `max_district` | District mean | Maximum across districts |
| `mean_of_maxes` | District maximum | Mean across districts |
| `global_max` | District maximum | Maximum across districts |

A singleton district has no pairwise trip and receives `NaN` district scores.
Its unit ID is still required to exist in the OD table, and weighted singleton
inputs still undergo population validation before the `NaN` result is returned.
Any non-finite district score propagates to the corresponding plan summary;
aggregation never removes an invalid district from the plan denominator.

## Example

```python
import networkx as nx
import gerrytools.scoring as gs

G = nx.DiGraph()
G.add_edge(10, 20, travel_time=8.0)
G.add_edge(20, 10, travel_time=12.0)
G.add_edge(20, 30, travel_time=9.0)
G.add_edge(30, 20, travel_time=11.0)

result = gs.build_travel_time_table(G, {0: 10, 1: 20, 2: 30})
assert result.is_applicable
od = result.table
od.validate_connected()
parts = {1: [0, 1], 2: [1, 2]}
pops = {0: 1.0, 1: 1.0, 2: 1.0}

gs.PairwiseTravelTime(weight="none", plan_agg="mean").score(parts, pops, od)
gs.CentroidTravelTime(weight="p", plan_agg="max_district").score(parts, pops, od)
```

The corresponding convenience functions are `pairwise_travel_time(...)` and
`centroid_travel_time(...)`. Functions use `snake_case`; reusable metric classes
use `PascalCase`.

## Saving and loading tables

`save_travel_time_table(...)` writes a compressed, pickle-free NPZ artifact. The
artifact stores the symmetric matrix as `float32`, unit IDs, a schema version, the
declared time unit (`seconds` or `minutes`), the required mean-of-directions policy,
and JSON metadata supplied by the builder. Saving and loading both validate the
table; disconnected tables cannot be published through this API.

Builder metadata should be sufficient to reproduce and audit the table, including
the geography vintage, source file identifiers, OSM snapshot date, routing profile,
representative-point rule, snapping rule, ferry policy, and software versions.

```python
gs.save_travel_time_table(
    "example_travel_times.npz",
    od,
    time_unit="seconds",
    metadata={"state": "example", "geography": "example-units"},
)

artifact = gs.load_travel_time_table("example_travel_times.npz")
od = artifact.table
print(artifact.time_unit, artifact.metadata)
```

## Building a table from representative points and OSM

The optional table builder requires OSMnx and Pyrosm. Install the extra from a
checkout with `uv sync --extra travel-time`, or from a published package with
`pip install "gerrytools[travel-time]"`.

The implemented unit-to-unit measure is:

**Travel time between the connected road-network nodes nearest the supplied
representative points.**

Canonical workflow:

1. Create or supply a point GeoDataFrame of representative locations
   (`census_representative_points(...)` for Census `INTPTLAT20` / `INTPTLON20`,
   or any other Point frame with a declared CRS);
2. build a directed driving graph from a local OSM PBF, add eligible vehicle
   ferries, then retain the largest strongly connected component;
3. snap each point to the nearest retained road node and record snap distances;
4. compute directed shortest paths, average opposite directions, and publish a
   table only when every snapped-node pair is mutually reachable.

One-shot helper:

<!-- docs-test: skip -- requires user-supplied Census geography and OSM PBF files -->
```python
import geopandas as gpd
import gerrytools.scoring as gs

units = gpd.read_file("tl_2020_20_vtd20.shp")
points = gs.census_representative_points(units, id_column="GEOID20")
result = gs.build_travel_time_table_from_pbf(
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

Equivalent stepwise form:

<!-- docs-test: skip -- requires user-supplied Census geography and OSM PBF files -->
```python
points = gs.census_representative_points(units, id_column="GEOID20")
graph, ferry = gs.build_osm_drive_graph("kansas-latest.osm.pbf")
unit_to_node, snap_distances = gs.snap_units_to_graph(
    graph,
    points,
    id_column="GEOID20",
    unit_polygons=units,
)
result = gs.build_travel_time_table(
    graph,
    unit_to_node,
    snap_distances_m=snap_distances,
    ferry=ferry,
)
```

`build_travel_time_table_from_pbf(...)` requires a Point GeoDataFrame and accepts
an optional polygon GeoDataFrame with matching unit IDs. With polygons supplied,
the builder chooses the contained retained-network node nearest the representative
point. If no retained node lies inside a unit, it chooses the node nearest any
exterior ring; polygon holes are excluded from this fallback. Reported snap
distances remain representative-point-to-node distances. The builder does not
calculate centroids or repair boundaries. Users without Census internal-point
columns must supply their own representative points.

Vehicle-ferry inclusion is conservative. A ferry way must affirmatively permit
`motor_vehicle`, `motorcar`, or `vehicle`, or carry a drivable `ferry=*` road
classification. Explicit denials and ambiguous passenger-only routes are excluded
and counted in the report. Ferries are added *before* the largest strongly
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
