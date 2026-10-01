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

| Mode | Trip weight | Interpretation |
| --- | --- | --- |
| `none` | `1` | Each unit pair counts equally |
| `p+p` | `pop_a + pop_b` | Population-involved pair weight |
| `p*p` | `pop_a * pop_b` | Person-pair weight |

District scores use $\sum (t w) / (\#\ \mathrm{trips})$ so population weights stay
in the numerator. For PTT, the denominator is the number of unordered unit pairs,
$n(n-1)/2$. For CTT, it is the number of centroid-to-unit trips, including the
zero-time trip from the centroid unit to itself. Units are time (`none`),
people-time (`p+p`), or people²-time (`p*p`). Weighted maximums are $\max(tw)$
and have the same units.

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
- `p+p` / `p*p`: choose $c$ with $p_c > 0$ minimizing $\sum_u t(c,u)\,p_u$

(The $p_c$ self-penalty is intentionally omitted so centers are not pushed toward
empty units.)

## Plan summaries

Each PTT or CTT family exposes four plan-level summaries:

| `plan_agg` | District statistic | Plan statistic |
| --- | --- | --- |
| `mean` | District mean | Mean across districts |
| `max_district` | District mean | Maximum across districts |
| `mean_of_maxes` | District maximum | Mean across districts |
| `global_max` | District maximum | Maximum across districts |

A singleton district has no pairwise trip and receives `NaN` district scores.
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

od = gs.build_od_from_graph(G, {0: 10, 1: 20, 2: 30})
od.validate_connected()
parts = {1: [0, 1], 2: [1, 2]}
pops = {0: 1.0, 1: 1.0, 2: 1.0}

gs.PairwiseTravelTime(weight="none", plan_agg="mean").score(parts, pops, od)
gs.CentroidTravelTime(weight="p*p", plan_agg="max_district").score(parts, pops, od)
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

## Building a table from Census units and OSM

The optional table builder requires OSMnx and Pyrosm. Install the extra from a
checkout with `uv sync --extra travel-time`, or from a published package with
`pip install "gerrytools[travel-time]"`.

`build_travel_time_table_from_pbf(...)` performs the expensive, plan-independent
work. It:

1. reads Census representative points from `INTPTLAT20` / `INTPTLON20` by default;
2. builds a directed driving graph from a local OSM PBF while retaining every road
   component;
3. adds explicitly vehicle-compatible OSM ferry ways;
4. uses an OSM ferry `duration` when available, otherwise estimates the crossing at
   20 km/h with no additional waiting penalty;
5. snaps representative points to the nearest eligible graph nodes;
6. computes directed shortest paths in batches, then averages opposite directions;
7. validates statewide connectivity and writes a reproducibility report; and
8. publishes a scoring table only when every included unit pair is reachable.

The function is geography-generic: change the ID and representative-point columns
to use block groups or another unit layer. It does not depend on district plans.

<!-- docs-test: skip -- requires user-supplied Census geography and OSM PBF files -->
```python
import geopandas as gpd
import gerrytools.scoring as gs

units = gpd.read_file("tl_2020_20_vtd20.shp")
result = gs.build_travel_time_table_from_pbf(
    units,
    "kansas-latest.osm.pbf",
    id_column="GEOID20",
    output_path="ks_vtd2020_travel_times.npz",
    report_path="ks_vtd2020_travel_times.report.json",
    # Optional, larger audit artifact retaining both route directions:
    directed_audit_path="ks_vtd2020_directed_audit.npz",
    metadata={"state": "KS", "geography": "2020 VTD"},
)

if not result.is_applicable:
    print("Travel time is not applicable:", result.report.connectivity_component_sizes)
```

Vehicle-ferry inclusion is conservative. A ferry way must affirmatively permit
`motor_vehicle`, `motorcar`, or `vehicle`, or carry a drivable `ferry=*` road
classification. Explicit denials and ambiguous passenger-only routes are excluded
and counted in the report. The report also records included and estimated ferries,
road-component counts, snap-distance statistics, directional asymmetry, and final
unit connectivity.

If the universe remains disconnected after legitimate road and ferry links are
included, `result.table` is `None`, the report status is `not_applicable`, and no
scoring artifact is written. The optional directed audit may still be retained for
diagnosis.

See the {doc}`scoring API <../../api/scoring>` for full signatures under
`gerrytools.scoring.travel_time`.
