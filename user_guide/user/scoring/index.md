# Scoring

GerryTools can score one assignment, a selected collection of plans, a GerryChain run, or a
recorded BEN/BENDL ensemble. Start with the overview, then use the guide for your workflow.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Overview
:link: overview
:link-type: doc

See the scoring interfaces, accepted assignment forms, metric families, and array formulas.
:::

:::{grid-item-card} Convenience Functions
:link: convenience_functions
:link-type: doc

Score one plan directly with simple functions.
:::

:::{grid-item-card} PlanEvaluator
:link: plan_evaluator
:link-type: doc

Prepare metrics once and reuse them across one plan, many plans, or recorded ensembles.
:::

:::{grid-item-card} Scoring BENDL files
:link: bendl
:link-type: doc

Inspect assignments, save Parquet scores, compare runs, and extend an existing score directory.
:::

:::{grid-item-card} Working with GerryChain
:link: basic
:link-type: doc

Use GerryTools metrics as updaters or evaluate selected partitions after a run.
:::

::::

The {doc}`scoring API <../../api/scoring>` contains the complete signatures, formulas, result
shapes, sign conventions, and resource requirements.

```{toctree}
:hidden:
:maxdepth: 1

Overview <overview>
Convenience Functions <convenience_functions>
PlanEvaluator <plan_evaluator>
Scoring BENDL files <bendl>
Working with GerryChain <basic>
```

## Coordinate reference systems

Geometry-backed scoring preserves an existing projected CRS. For geographic input (such as
longitude/latitude in EPSG:4326), it estimates a UTM CRS from the complete scoring dataset when
`target_crs` is omitted. The projection is prepared once per evaluator, not per district or plan.
An explicit `target_crs` overrides this choice. Input geometry must always declare its source CRS.

For example, `PlanEvaluator(graph, geometry=units).add_metric(Reock())` accepts geographic
`units` without requiring an explicit projection.

Automatic UTM selection is a convenience for local or regional data. For wide extents, polar
regions, or data crossing the antimeridian, choose an appropriate projected `target_crs` explicitly.
If a UTM CRS cannot be determined, scoring raises an error instead of measuring in degrees.
Projection can affect compactness scores; use an explicit CRS for comparisons across datasets.
Graph-backed scores using existing area and perimeter attributes do not reproject those values.

`PopulationPolygon(..., population_units=blocks)` transforms the separate population polygons
from their declared CRS into the evaluator's scoring CRS. For a bare Shapely state boundary, use
`StateClippedConvexHullRatio(boundary, state_crs=units.crs)` (or pass `state_crs` to the single-plan
function). Without `state_crs`, the boundary must already use the final scoring CRS, as before.
Caller-owned GeoDataFrames and geometries are not modified.

## GerryChain graph inputs

`PlanEvaluator` and the single-plan scoring functions accept NetworkX graphs, GerryChain
`Graph` objects backed by NetworkX or RustworkX, and frozen graphs such as `partition.graph`.
You can pass a GerryChain graph directly without converting it yourself.

Assignments supplied as mappings and geometry identifiers use the original node labels, including
string and tuple labels. Sequence assignments follow NetworkX node iteration order for
NetworkX-backed inputs, or increasing internal node-index order for RustworkX-backed inputs.
A RustworkX subgraph retains its original labels even when GerryChain renumbers its internal nodes.
Use a mapping when the ordering is uncertain.

RustworkX-backed inputs are converted once at evaluator construction, preserving node and edge
attributes in copied dictionaries. Replacing attribute values later does not change the evaluator.
NetworkX-backed inputs retain the existing lazy resource preparation behavior. All inputs must
remain simple and undirected, without self-loops. Passing a graph alone does not infer its geometry;
continue to supply `geometry=` for geometry-backed scores.
