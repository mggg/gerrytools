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
