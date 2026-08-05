# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `evaluate_many` now returns `ManyPlanEvalResult`; streamed ensemble evaluations use
  `EnsembleEvalResult`. `EvaluationRun` has been removed.

## [2.0.0] - 2026-08-04

GerryTools 2.0.0 is a rewrite. Every subpackage changed, the public API is not backward
compatible with 1.x, and no deprecation aliases are provided. Read the migration notes below
before upgrading.

Highlights:

- Plan scoring is now backed by a compiled Rust engine with incremental metric updates, so
  ensembles stream instead of loading into memory.
- Plotting moved from single-call functions to composable plot classes.
- New `colors` and `latex` subpackages, and a new `plan_comparison` subpackage carved out of the
  old `geometry` module.
- Geometry utilities that duplicated [maup](https://github.com/mggg/maup) were dropped rather
  than maintained in two places.

### Added

- **`gerrytools.scoring`: a metric-object API and a streaming evaluator.**
  - `PlanEvaluator` prepares graph and geometry resources once and evaluates one plan
    (`evaluate`), many plans (`evaluate_many`), or a whole ensemble file (`evaluate_stream`).
  - Metrics are now classes (`Tally`, `EfficiencyGap`, `PolsbyPopper`, `RegionSplits`, ...) that
    declare the resources they need, so preparation happens once per evaluator rather than once
    per call.
  - `gerrytools.scoring.single_plan` provides one-shot functions (`efficiency_gap`,
    `polsby_popper`, ...) for the common "score a single plan" case, accepting a `Partition`, an
    `nx.Graph`, or a `GeoDataFrame`.
  - `gerrytools.scoring.formulas` exposes the underlying array formulas for callers that already
    have district-level arrays and want to skip the evaluator entirely.
  - Streamed runs publish a self-describing run directory with a run-named JSON manifest and
    Parquet score tables. Tallies and region tallies use one table per requested attribute. New
    score names can be added later; `update=True` replaces matching names. `EvaluationRun` reads
    the directory with a memory guard (`EvaluationMemoryError`) so an oversized read fails loudly
    instead of exhausting RAM.
- **A compiled scoring engine.** The package now builds with
  [maturin](https://www.maturin.rs/) and ships a Rust extension module
  (`gerrytools._scoring_engine`) implementing tallies, compactness scores, cut edges, region
  metrics, and the partisan formulas, with incremental updates across chain steps. Its results
  are differential-tested against the Python `formulas` mirror.
- **New metrics with no 1.x equivalent:** `DistrictVoteShares`, `DistrictWins`,
  `OverallVoteShare`, `MeanSignedSeatVoteGap`, `MeanAbsoluteSeatVoteGap`,
  `MaxAbsolutePopulationDeviation`, `RegionParts`, `StateClippedConvexHullRatio`, and
  `TallyByRegion`.
- **`gerrytools.colors`: a dedicated color subpackage.** Districtr palettes, LaTeX `xcolor`
  names, seaborn and matplotlib colormaps, and the Lab's standard colors are now one importable
  namespace with `get_named_color`, `resolve_rgba`, `compare_palettes`, and `preview_palette`.
  The LaTeX color table is a Python dictionary rather than a JSON file loaded at import.
- **`gerrytools.latex`: LaTeX document and figure generation.** `TexDocument`, `TexTable`,
  `TikzTable`, `TikzPaintballPlot`, and `TikzSeatsVotesPlot` emit publication-ready `.tex`, with
  `latex_escape` and a formatter system (`highlight_gt`, `round_decimals`,
  `diverging_gradient_formatter`, `compose_formatters`).
- **`gerrytools.plan_comparison`: plan-to-plan comparison.** `population_overlap`,
  `areal_overlap`, `optimal_relabeling`, `population_dispersion`,
  `population_dispersion_by_district`, `minimum_population_dispersion`, and
  `minimum_population_dispersion_with_parity`.
- **`gerrytools.ben`: BENDL recording.** `RecordedChain` records a GerryChain run directly to a
  self-describing BENDL file containing the graph, node permutation, metadata, and assignment
  stream. `RecordedRun` reads one back with zero-based lookup, subsampling, and lazy partition
  reconstruction. Writes are transactional: a failed run preserves its partial recording rather
  than clobbering the destination.
- **`gerrytools.mgrp`: constraints and objectives.** `Constraints` and `Objective` build and
  validate engine configuration, replacing hand-assembled dictionaries. New optimizer run types
  `ShortBurstsRunInfo` and `TiltedRunInfo` (sharing `OptimizerRunInfoBase`).
- **New plotting surfaces:** `DotDensityPlot`, `BarPlot`, `SeatsVotesPlot`, `PaintballPlot`,
  `subway_signs`, and `draw_graph_components`, plus a typed options layer under
  `gerrytools.plotting.mpl`.
- **US Census table descriptors.** `pl_table`, `census_column_name`, and the `ACS*TableInfo` /
  `PL*TableInfo` classes make the variable-to-column mapping explicit and vintage-aware.

### Changed

- **Python 3.11 or newer is now required** (1.x supported older interpreters).
- **The package is no longer pure Python.** It builds with the maturin backend and ships a
  compiled extension module, so installing from source requires a stable Rust toolchain.
- **Project structure.** Previous structure:

```console
gerrytools
├── ben
│   ├── binary_ensemble.py
│   ├── docker_manager.py
│   ├── __init__.py
│   ├── parse.py
│   └── reben.py
├── data
│   ├── acs.py
│   ├── AssignmentCompressor.py
│   ├── census.py
│   ├── estimatecvap.py
│   ├── fetch.py
│   ├── geometries.py
│   ├── __init__.py
│   ├── remap.py
│   └── URLs.py
├── geometry
│   ├── compactness.py
│   ├── dataframe.py
│   ├── dissolve.py
│   ├── dualgraph.py
│   ├── __init__.py
│   ├── optimize.py
│   ├── unitmap.py
│   └── updater.py
├── __init__.py
├── mgrp
│   ├── __init__.py
│   ├── run_container.py
│   └── runners
│       ├── forest.py
│       ├── __init__.py
│       ├── recom.py
│       └── smc.py
├── plotting
│   ├── annotation.py
│   ├── bins.py
│   ├── boxplot.py
│   ├── choropleth.py
│   ├── colors.py
│   ├── districtnumbers.py
│   ├── drawgraph.py
│   ├── drawplan.py
│   ├── gifs.py
│   ├── histogram.py
│   ├── __init__.py
│   ├── latexcolors.json
│   ├── multidimensional.py
│   ├── scatterplot.py
│   ├── sealevel.py
│   ├── utils.py
│   └── violin.py
├── scoring
│   ├── contiguity.py
│   ├── demographics.py
│   ├── __init__.py
│   ├── partisan.py
│   ├── population.py
│   ├── scores.py
│   ├── splits.py
│   └── types.py
└── utilities
    ├── __init__.py
    ├── JSON.py
    └── rename.py
```

New structure (private modules, prefixed with `_`, are omitted):

```console
gerrytools
├── __init__.py
├── logging.py
├── typing.py
├── ben
│   ├── __init__.py
│   └── recorded_chain.py
├── colors
│   ├── __init__.py
│   ├── core.py
│   ├── districtr.py
│   ├── latex.py
│   ├── seaborn.py
│   └── utils.py
├── data
│   ├── __init__.py
│   ├── geometries.py
│   └── uscensus
│       ├── __init__.py
│       ├── acs.py
│       ├── block_cvap.py
│       ├── census.py
│       └── census_tables.py
├── latex
│   ├── __init__.py
│   ├── commands.py
│   ├── document.py
│   ├── formatters.py
│   ├── paintball.py
│   ├── seatsvotes.py
│   ├── table.py
│   └── tikz_table.py
├── mgrp
│   ├── __init__.py
│   ├── constraints.py
│   ├── objectives.py
│   ├── run_config.py
│   ├── run_container.py
│   └── runners
│       ├── forest.py
│       ├── recom.py
│       └── smc.py
├── plan_comparison
│   ├── __init__.py
│   ├── overlap.py
│   └── relabel.py
├── plotting
│   ├── __init__.py
│   ├── utils.py
│   ├── data
│   │   ├── barplot.py
│   │   ├── boxplot.py
│   │   ├── gerryplot.py
│   │   ├── histogram.py
│   │   ├── options.py
│   │   ├── paintball.py
│   │   ├── scatterplot.py
│   │   ├── sealevel.py
│   │   ├── seatsvotes.py
│   │   └── violin.py
│   ├── geometry
│   │   ├── dotdensity.py
│   │   ├── geoplot.py
│   │   └── geoplotbase.py
│   ├── mpl
│   │   ├── axis_title_style.py
│   │   ├── geoplot_options.py
│   │   ├── label_text_options.py
│   │   ├── legend_options.py
│   │   ├── marker_options.py
│   │   └── tick_style.py
│   ├── other
│   │   └── subway.py
│   └── plan
│       └── drawgraph.py
└── scoring
    ├── __init__.py
    ├── evaluator.py
    ├── formulas.py
    ├── metrics
    │   └── __init__.py
    ├── result.py
    └── single_plan
        └── __init__.py

rust/src        # the compiled scoring engine, built as gerrytools._scoring_engine
```

- **`gerrytools.scoring` moved from updater functions to metric objects.** Where 1.x asked for a
  list of updater callables passed to `summarize`, 2.0.0 registers metric instances on a
  `PlanEvaluator`. Name mapping:

  | 1.x                         | 2.0.0 metric class            | 2.0.0 single-plan function   |
  | --------------------------- | ----------------------------- | ---------------------------- |
  | `splits`                    | `RegionSplits`                | `region_splits`              |
  | `pieces`                    | `RegionPieces`                | `region_pieces`              |
  | `competitive_contests`      | `CompetitiveContests`         | `competitive_contests`       |
  | `swing_districts`           | `SwingDistricts`              | `swing_districts`            |
  | `party_districts`           | `PartyDistricts`              | `party_districts`            |
  | `opp_party_districts`       | `OppositionPartyDistricts`    | `opposition_party_districts` |
  | `party_wins_by_district`    | `PartyWinsByDistrict`         | `party_wins_by_district`     |
  | `seats`                     | `Seats`                       | `seats`                      |
  | `aggregate_seats`           | `AggregateSeats`              | `aggregate_seats`            |
  | `efficiency_gap`            | `EfficiencyGap`               | `efficiency_gap`             |
  | `simplified_efficiency_gap` | `SimplifiedEfficiencyGap`     | `simplified_efficiency_gap`  |
  | `mean_median`               | `MeanMedian`                  | `mean_median`                |
  | `partisan_bias`             | `PartisanBias`                | `partisan_bias`              |
  | `partisan_gini`             | `PartisanGini`                | `partisan_gini`              |
  | `eguia`                     | `Eguia`                       | `eguia`                      |
  | `deviations`                | `PopulationDeviations`        | `population_deviations`      |
  | `max_deviation`             | `MaxPopulationDeviation`      | `max_population_deviation`   |
  | `demographic_shares`        | `DemographicShares`           | `demographic_shares`         |
  | `demographic_tallies`       | `Tally`, `TallyByRegion`      | `tally`, `tally_by_region`   |
  | `gingles_districts`         | `DistrictsAboveThreshold`     | `districts_above_threshold`  |
  | `reock`                     | `Reock`                       | `reock`                      |
  | `polsby_popper`             | `PolsbyPopper`                | `polsby_popper`              |
  | `schwartzberg`              | `Schwartzberg`                | `schwartzberg`               |
  | `convex_hull`               | `ConvexHullRatio`             | `convex_hull_ratio`          |
  | `pop_polygon`               | `PopulationPolygon`           | `population_polygon`         |
  | `cut_edges`                 | `CutEdges`                    | `cut_edges`                  |
  | `summarize`                 | `PlanEvaluator.evaluate`      | n/a                          |
  | `summarize_many`            | `PlanEvaluator.evaluate_many` | n/a                          |

- **`gerrytools.plotting` moved from functions to plot classes.** A 1.x call such as
  `histogram(ax, scores)` becomes `Histogram()`, then `add_dataset(...)`, then `.ax` or
  `.save(...)`. Rendering is lazy: the figure is built on first access to `.ax`, so constructing
  a plot in a notebook no longer emits an empty figure. Name mapping:

  | 1.x                                                     | 2.0.0                                                                           |
  | ------------------------------------------------------- | ------------------------------------------------------------------------------- |
  | `histogram`                                             | `Histogram`                                                                     |
  | `violin`                                                | `ViolinPlot`                                                                    |
  | `boxplot`                                               | `BoxPlot`                                                                       |
  | `scatterplot`                                           | `ScatterPlot`                                                                   |
  | `sealevel`                                              | `SeaLevelPlot`                                                                  |
  | `drawplan`                                              | `GeoPlot.add_districting_plan_layer`                                            |
  | `choropleth`                                            | `GeoPlot.add_choropleth_layer`                                                  |
  | `districtnumbers`                                       | `GeoPlot.add_label_layer`                                                       |
  | `drawgraph`                                             | `draw_graph`                                                                    |
  | `arrow`                                                 | `add_arrow` methods (`add_label_arrow`, `add_text_arrow`, and axis variants)    |
  | `ideal`                                                 | `add_vertical_lines` / `add_horizontal_lines` (multiple lines, optional jitter) |
  | `bins`                                                  | `Histogram.set_bins` / `Histogram.set_bin_widths`                               |
  | `districtr`, `flare`, `purples`, `redbluecmap`, `latex` | `gerrytools.colors`                                                             |

- **`gerrytools.data` reorganized around US Census tables.** Census access lives under
  `gerrytools.data.uscensus`, and year selection is a parameter rather than a function name.

  | 1.x                                    | 2.0.0                                                      |
  | -------------------------------------- | ---------------------------------------------------------- |
  | `census10`, `census20`                 | `census(..., year=...)`                                    |
  | `acs5`                                 | `acs`, `acs_full`                                          |
  | `estimatecvap2010`, `estimatecvap2020` | `block_cvap_estimates`                                     |
  | `fetchgeometries`                      | `geometries20`, `vtds20`, `dualgraphs20`                   |
  | `variables`                            | `pl_table`, `census_column_name`, the `*TableInfo` classes |

  Migration details that are not simple renames:
  - `census()` requires an explicit Census API key; 1.x supplied a package default. Its default
    geography is now `"state"`, not `"block"`, so pass `geometry="block"` when porting a
    `census20(state)` call.
  - `cvap()` now requires both `geometry` and `year`.
  - `geometries20()` raises for an unknown geography instead of warning and falling back.

- **Additional 1.x API removals and constructor changes:**
  - The `gerrytools.scoring.types.Score` custom-score extension point is removed with no direct
    replacement. The 2.0 metric set is closed because each metric requires scoring-engine
    support; 1.x `Score` subclasses cannot be registered with `PlanEvaluator`.
  - `SMCRunnerConfig(shapefile_dir, shapefile_name, ...)` now takes one `shapefile_path`.
  - `ForestRunInfo` replaces `region_name`/`subregion_name` with `levels` and replaces the output
    booleans with `writer`.
  - Top-level `gerrytools.__version__`, `gerrytools.mgrp_available`, and the implicit
    `gerrytools.mgrp` attribute are removed. Import subpackages explicitly and use
    `importlib.metadata.version("gerrytools")` for the installed version.
  - Color constants use uppercase names: `defaultGray` is `DEFAULT_GREY`, `citizenBlue` is
    `CITIZEN_BLUE`, `overlays` is `OVERLAYS`, and the old plotting `latex` dictionary is
    `gerrytools.colors.LATEX_COLOR_DICT`.
  - `plotting.utils.sort_elections` and `geometry.dissolve.by_area` are removed with no direct
    replacement.
  - `districtr()` retains its purpose, but extension colors beyond the first 39 and the casing of
    ten base hex strings changed. Do not compare its color strings byte-for-byte across releases.

- **`gerrytools.ben` rewritten around BENDL.** 1.x wrapped the `binary-ensemble` CLI through
  Docker; 2.0.0 depends on `binary-ensemble` 2.0.0 directly and records to BENDL. The
  `ben`, `ben_replay`, `msms_parse`, `smc_parse`, `canonicalize_ben_file`, and `relabel_*`
  functions are replaced by `RecordedChain` and `RecordedRun`.
- `RecordedRun.lookup()` and the `subsample_*()` methods now always return assignment vectors.
  Use `partition_at()` for one reconstructed partition, or wrap vector iterables with
  `partitions()` to reconstruct them lazily.
- **`gerrytools.mgrp` run configuration is validated.** `SMCMapInfo` and `SMCRedistInfo` are
  merged into a single `SMCRunInfo`. Runner configuration is checked at construction, and the
  engine command is assembled from a static template with all run-specific values passed as a
  JSON argument rather than interpolated into a shell string.
- The `mgrp` Docker images were rebuilt for all engines.
- Dispersion calculations moved to a SciPy linear-sum-assignment solve. `minimize_dispersion`
  and `minimize_dispersion_with_parity` are now `minimum_population_dispersion` and
  `minimum_population_dispersion_with_parity` in `gerrytools.plan_comparison`, and the parity
  variant is a true lexicographic optimization rather than a two-stage heuristic.
- Documentation moved to a Sphinx site under `user_guide/`, with runnable tutorial notebooks and
  every prose code block executed in CI.

### Removed

- **`gerrytools.geometry` (entire subpackage).** Its contents were split, moved to
  [maup](https://github.com/mggg/maup), or dropped:
  - `populationoverlap`, `arealoverlap`, `optimalrelabeling`, `calculate_dispersion`,
    `minimize_dispersion`, `minimize_dispersion_with_parity` moved to
    `gerrytools.plan_comparison` under the new names listed above.
  - `dissolve`, `unitmap`, `invert`, `dualgraph`, and `dataframe` are removed. maup covers this
    ground, and maintaining a second implementation was a recurring source of drift.
  - `dispersion_updater_closure` and `minimize_parity` are removed with no replacement.
  - Compactness scores (`reock`, `polsby_popper`, `schwartzberg`, `convex_hull`, `pop_polygon`)
    moved to `gerrytools.scoring` as metric classes.
- **`gerrytools.utilities` (entire subpackage).**
  - `rename.py`: this just let you rename a file. Why do this in Python? It also does nothing for
    our standard data pipelines.
  - `JSON.py`: this just allowed a user to read a JSON object in as a Python object. The `json`
    module already does this perfectly well, and the only thing this added was checking that the
    attribute names in the JSON object were compatible with shapefile attribute specifications.
    We are moving to geopackages, and fixing columns for use in shapefiles is an infrequent
    enough task that it need not have a dedicated function.
- **`gerrytools.plotting`:**
  - `multidimensional.py`: this only had one function in it, and it was broken because the
    signatures of the functions it relied on changed. It also just stacked a scatterplot on top
    of a histogram, and our lab has moved away from stacking plots in Python in favor of making
    individual plots and stacking them in LaTeX.
  - `gifs.py` (`gif_multidimensional`): this only made a gif from the plots made by
    `multidimensional.py`, so it was removed as well.
  - `latexcolors.json`: this was just a JSON file with color names and their RGB values, loaded
    by another module. Rather than waste time with the load, it is now a Python dictionary in
    `gerrytools.colors`.
- **`gerrytools.data`:**
  - `AssignmentCompressor`: superseded by the BEN and BENDL formats in `gerrytools.ben`.
  - `remap`: maup covers this.
  - `fetch.py` (`submissions`, `tabularized`, `Submission`) and `URLs.py` (`ids`, `one`, `csvs`):
    the districtr portal fetch helpers are removed.
- **`gerrytools.scoring`:** `responsive_proportionality`, `stable_proportionality`, `contiguous`,
  `unassigned_units`, `unassigned_population`, and `demographic_updaters` are removed with no
  replacement.
- **`gerrytools.ben`:** the Docker-based `docker_manager.py` is removed; `binary-ensemble` is now
  a direct dependency.

[Unreleased]: https://github.com/mggg/gerrytools/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/mggg/gerrytools/compare/v1.2.1...v2.0.0
