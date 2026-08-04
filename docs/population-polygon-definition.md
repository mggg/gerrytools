# PopulationPolygon: Definition, Interpretation, and Implementation

## Summary

`PopulationPolygon` is a district-level compactness score. It divides the population attributed
to a district by the total weight of the population polygons that intersect the district's convex
hull. The score is greater than zero and at most one. A higher score means that less population
attributed to other districts is carried into the denominator by polygons intersecting the hull.

GerryTools uses a full-weight intersection convention:

> If a population polygon has any nonempty intersection with the district hull, its entire
> population weight contributes to the denominator.

This reproduces the previous GerryTools implementation. That implementation clipped a population
GeoDataFrame to each district hull and summed the population attribute of every retained row. It
did not apportion a row's population according to the area remaining after clipping.

The denominator is therefore a whole-polygon approximation to the population within the hull, not
an areally interpolated estimate. Its resolution is part of the metric definition: finer
population polygons can reduce the amount of population carried into the denominator by a
small boundary intersection. The metric remains contour-based because the district's convex hull
determines which population polygons contribute.

## Mathematical definition

Let:

- $V$ be the graph-node set;
- $a:V\rightarrow\mathcal D$ be a district assignment;
- $U_v$ be the geometry of graph node $v$;
- $G_i$ be population polygon $i$;
- $w_i\geq 0$ be its population weight; and
- $c(i)\in V$ be the graph node to which population polygon $i$ is attributed.

For the default population surface, each polygon is a graph-unit geometry and is attributed to
that unit. For an alternative surface, GerryTools requires one graph unit whose interior contains
a point on the polygon's surface and which covers the polygon up to the overlay tolerance described
below. That unit defines $c(i)$.

For district $D$, its geometry and convex hull are

$$
Q(D)=\bigcup_{\{v:a(v)=D\}}U_v
\qquad\text{and}\qquad
H(D)=\operatorname{conv}(Q(D)).
$$

The population attributed to the district is

$$
P(D)=\sum_{\{i:a(c(i))=D\}}w_i.
$$

The full-weight population-polygon measure of its hull is

$$
P_{\cap}(H(D))
=
\sum_{\{i:G_i\cap H(D)\neq\varnothing\}}w_i.
$$

The score is

$$
\operatorname{PopulationPolygon}(D)
=
\frac{P(D)}{P_{\cap}(H(D))}.
$$

Owner coverage guarantees that every population polygon attributed to $D$ intersects $H(D)$.
Consequently, $P_{\cap}(H(D))\geq P(D)$ and the score lies in $(0,1]$ for every district accepted
by the scorer. A score of one means that no positive-weight polygon attributed to another district
intersects the hull.

The denominator predicate is a nonempty set intersection, not a positive-area test. A population
polygon that touches the hull only along an edge or at one point contributes its complete weight.

### Default population surface

By default, the evaluator's aligned geometry GeoDataFrame is the population surface. There is one
population observation per graph node:

$$
G_v=U_v,\qquad c(v)=v,\qquad
w_v=\texttt{geometry.loc[v, population\_col]}.
$$

The geometry rows are aligned to graph-node order when the evaluator prepares its resources.
Containment and ownership are automatic because each observation is its graph unit.

### Alternative population surface

`population_units` replaces the default population surface with another polygon layer, usually
at a finer resolution. Its rows and `population_col` values define both the numerator and the
denominator. A finer surface can reduce the full-weight effect of polygons that intersect only a
small part of the hull, but the scorer still assigns every intersecting polygon's complete weight.
Hold the population surface fixed when comparing scores across plans.

Users do not provide an owner column. During evaluator preparation, GerryTools takes a point on
the surface of each alternative polygon and uses a spatial index to find candidate graph units. It
then requires exactly one candidate to cover the complete polygon, allowing uncovered area no
greater than

$$
10^{-12}\max(\operatorname{area}(G_i),1)
$$

to accommodate overlay roundoff. Finding zero or multiple covering candidates raises a
`ValueError`. The Rust scorer checks the inferred owner and the same coverage tolerance again when
it prepares the population surface.

## Exact intersection convention

For a positive-weight polygon, the following all add its complete weight $w_i$ to the denominator:

1. $G_i$ lies completely inside $H(D)$.
2. $G_i$ crosses the hull boundary.
3. Only a narrow sliver of $G_i$ lies inside the hull.
4. $G_i$ touches the hull boundary without a positive-area overlap.

An intersecting zero-weight polygon has no numerical effect. A positive-weight polygon contributes
nothing only when it is disjoint from the hull.

For example, suppose the surface contains two observations. District $A$ owns a polygon of weight
$10$, while a polygon of weight $90$ attributed to district $B$ touches $A$'s hull at one boundary
point. Then

$$
\operatorname{PopulationPolygon}(A)=\frac{10}{10+90}=0.1.
$$

The result would be the same if almost all of the weight-$90$ polygon lay inside the hull. The
intersection determines whether its complete weight is included; the overlap area does not affect
the contribution.

## Relationship to historical GerryTools

The historical implementation effectively performed the following operation for each district:

```python
intersecting_rows = geopandas.clip(population_frame, district_geometry.convex_hull)
denominator = intersecting_rows[population_column].sum()
score = district_population / denominator
```

For a polygon mask, `geopandas.clip` intersects each input geometry with the mask while retaining
the row's attribute values. Its default `keep_geom_type=False` also permits lower-dimensional
boundary intersections in the result. Summing the population column therefore gives every
retained source row its full population; it does not multiply population by the retained-area
fraction.

The Rust implementation makes this convention explicit without constructing a clipped
GeoDataFrame for every district. In particular, its result does not depend on which geometry type
GeoPandas returns for a boundary-only intersection. The current Python/Rust path requires:

- graph and population geometry must use the same projected CRS; and
- each alternative population polygon must be attributed to exactly one graph unit under the
  coverage rule above.

The inferred owner relation defines the numerator when an alternative population surface is used.

## Public Python interface

The usual interface names the population column in the evaluator's geometry GeoDataFrame:

```python
from gerrytools.scoring import PlanEvaluator, PopulationPolygon

scorer = PlanEvaluator(
    graph,
    geometry=graph_units,
    node_id_column="graph_node",
).add_metric(PopulationPolygon("population"))
```

`population_col` is positional. The optional finer surface is keyword-only:

```python
metric = PopulationPolygon(
    "population",
    population_units=population_blocks,
)
```

When `population_units` is omitted, the evaluator's aligned geometry GeoDataFrame supplies both
geometries and values. When it is present, that frame supplies both. The same `population_col` must
exist in the selected frame.

`PlanEvaluator` records the evaluator GeoDataFrame at construction and snapshots the required
geometry and population column during the first metric preparation. Mutations made before that
preparation can affect the snapshot; later mutations cannot. By contrast, `PopulationPolygon`
copies an alternative surface's geometry as WKB, population values, and CRS metadata when the
metric is constructed.

### Required inputs

- The evaluator geometry must have exactly one row per graph node, aligned by its index or by
  `node_id_column`.
- Evaluator and alternative population geometries must be valid, nonempty `Polygon` or
  `MultiPolygon` values with positive area.
- Evaluator and alternative population geometries must use the same projected CRS. The evaluator
  geometry may be explicitly transformed with `PlanEvaluator(..., target_crs=...)`; the alternative
  surface is not reprojected automatically.
- `population_col` must exist in whichever GeoDataFrame supplies the population surface.
- Population values must be finite and nonnegative, and at least one value must be positive.
- Every alternative population polygon must be covered by exactly one graph-unit geometry, subject
  only to the documented overlay tolerance.
- Every district observed during scoring must have positive attributed population.

Zero-population polygons are permitted. Duplicate alternative rows are also permitted and
contribute independently. Users are responsible for ensuring that duplicates represent distinct
intended observations rather than accidental double counting.

GerryTools does not repair, dissolve, deduplicate, or remap alternative population geometries.
Those operations can change the statistic and belong in explicit data preparation.

## References

- Moon Duchin,
  [Political Geometry](https://data-democracy.org/publications/political-geometry/01-Duchin.pdf),
  for the population-polygon compactness ratio.
- Moon Duchin and Bridget Eileen Tenner,
  [Discrete Geometry for Electoral Geography](https://doi.org/10.1016/j.polgeo.2023.103040), for
  the distinction between contour-based and discrete compactness metrics.
- GeoPandas, [`geopandas.clip` documentation][geopandas-clip].

[geopandas-clip]: https://geopandas.org/en/stable/docs/reference/api/geopandas.clip.html
