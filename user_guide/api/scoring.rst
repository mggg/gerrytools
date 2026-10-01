Scoring
=======

For task-oriented examples, see the :doc:`scoring guide <../user/scoring/index>`.

Single plans, evaluators, and results
-------------------------------------

Lowercase functions evaluate one plan from a GerryChain partition or GeoDataFrame. Capitalized
metric descriptions are registered with :class:`PlanEvaluator` when resources should be reused
across metrics or plans.

Evaluator preparation and source lifetime
-----------------------------------------

``PlanEvaluator`` prepares metric resources automatically on the first evaluation. It snapshots
only the requested graph or GeoDataFrame columns and reuses those immutable resources and the
scoring engine on later evaluations. Adding another metric rebuilds the engine and extends the
resource snapshot only when the new metric needs something that was not already prepared.

The evaluator borrows its graph and optional GeoDataFrame. Mutations made before the first
successful preparation are observed. Do not mutate either source afterward: later metric
additions retain old resources while snapshotting only newly requested ones, so source mutation
could produce values from different source states. Graph structure must remain unchanged for the
evaluator's entire lifetime.

If preparation fails after adding a metric to an already-prepared evaluator, the prior resource
snapshot remains intact but the added metric remains registered. Construct a replacement evaluator
with corrected inputs rather than mutating a published source or retrying the old evaluator.

An authoritative GeoDataFrame controls ordinary node and region columns. Its active geometry
column is reserved for geometry-backed metrics and cannot be named as an ordinary metric column.
Row alignment is prepared when an authoritative column is first requested. Reprojection and
geometry validation occur only when a geometry-backed metric is evaluated, so ``crs`` has no
effect on an evaluator that uses only ordinary columns.

Saved ensemble results
----------------------

``evaluate_stream`` creates an ensemble result directory when it does not exist. When the directory
contains an existing ensemble result, score names not already present are added automatically. A
matching name raises ``FileExistsError`` unless ``update=True``; with that flag, only matching names
are replaced. The caller is responsible for using the same assignment stream. See :doc:`the BENDL
scoring guide <../user/scoring/bendl>` for a complete example and the physical file layout.

.. automodule:: gerrytools.scoring
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: CentroidTravelTime, FerryBuildSummary, PairwiseTravelTime, TravelTimeArtifact, TravelTimeBuildReport, TravelTimeBuildResult, TravelTimeScorer, TravelTimeTable, add_vehicle_ferries, build_od_from_graph, build_osm_drive_graph, build_travel_time_table, build_travel_time_table_from_pbf, census_representative_points, centroid_travel_time, compute_directed_travel_times, load_travel_time_table, pairwise_travel_time, parse_osm_duration_seconds, save_travel_time_table, snap_units_to_graph

Array formulas
--------------

These functions operate on arrays already aggregated by district. Their final axis is the district
axis unless an individual function documents another contract.

.. automodule:: gerrytools.scoring.formulas
   :members:
   :exclude-members: TurnoutModel

Travel-time scores
------------------

Pairwise (PTT) and centroid (CTT) drive-time scores over a finished unit-to-unit
OD table. Each scoring time is the arithmetic mean of the two directed route times.
Routes may leave district boundaries, but the table must contain a finite time for
every unit pair. These helpers reject disconnected tables and are independent of
the compiled ``PlanEvaluator`` engine. See :doc:`the travel-time guide
<../user/scoring/travel_time>` for formulas, units, and connectivity policy.

.. automodule:: gerrytools.scoring.travel_time
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
