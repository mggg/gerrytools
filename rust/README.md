# GerryTools scoring engine

This crate is the reusable Rust scoring engine migrated from `ben-process`. It deliberately has
no CLI. The repository root packages its PyO3 bindings as `gerrytools._scoring_engine` with
Maturin.

## Upstream baseline

The selected source is the clean local checkout at:

```text
/home/peter/Projects/ben-process
f27b748466a491d49ecf1146347e3ae012610499
```

The prepared scoring core is derived from:

```text
src/adjacency.rs
src/geometry/mod.rs
src/geometry/unit_hulls.rs
src/metrics/area_perimeter.rs
src/metrics/convex_hull_ratio.rs
src/metrics/cut_edges.rs
src/metrics/formulas.rs
src/metrics/hull_metric.rs
src/metrics/polsby_popper.rs
src/metrics/population_polygon.rs
src/metrics/region.rs
src/metrics/region_parts.rs
src/metrics/region_tally.rs
src/metrics/reock.rs
src/metrics/schwartzberg.rs
src/metrics/state_clipped_convex_hull_ratio.rs
src/metrics/tally.rs
src/python.rs
src/scoring/delta.rs
src/scoring/district.rs
src/scoring/input.rs
src/scoring/mod.rs
src/scoring/output.rs
src/scoring/result.rs
src/scoring/stream.rs
```

The standalone core contains full and TwoDelta-incremental Tally, Polsby-Popper, Reock,
convex-hull ratio, state-clipped convex-hull ratio, population-polygon, cut-edge, region, and
region-tally kernels. Shared delta validation rejects stale labels, out-of-range nodes and labels,
and duplicate node changes before a metric mutates.

`PreparedCutEdges` accepts an explicit node count and validated edge list, with a separate weighted
constructor. Finite negative weights retain the upstream signed-sum behavior. `PreparedRegion`
accepts one or more aligned columns, ignores missing values, and internally densifies arbitrary
`u32` region identifiers. Region splits count distinct district intersections. Region pieces also
accept graph edges and count connected components in each region-induced district subgraph, so a
connection outside the region does not merge its pieces. Incremental piece scoring repairs only
components touched by removed nodes and merges the new components adjacent to added nodes. Both
metrics produce plan tables and retain district-set metadata, including districts represented only
by isolated or region-missing nodes.

The stream scorer reads Standard, MkvChain, and TwoDelta streams from BEN, XBEN, and finalized
BENDL inputs. Independent Standard frames are decoded and scored in parallel batches. MkvChain and
TwoDelta streams update registered metrics incrementally. Results are emitted in file order through
a callback with zero-based sample offsets and preserved repetition counts.

BENDL construction is opaque and verifies both the assignment-stream checksum and declared sample
count. The subsequent scoring pass can then use parallel frame decoding or incremental TwoDelta
events without silently bypassing whole-stream integrity.

The result contract distinguishes district-valued and plan-valued metric blocks. Plan-valued
tables retain the observed district ids as row metadata so mixed results can enforce one stable
district set. `Scorer::score_run` checks metric kind, instance, shape, and subkey count before it
creates output.

Completed runs are published as version-1 run directories with one Snappy-compressed Parquet table
per metric and a manifest containing the source, stream summary, district ids, metric options,
subkeys, shapes, and relative table paths. The writer buffers bounded column batches, syncs every
table and the manifest, and renames a temporary sibling only after all output finishes. A failed
push poisons the writer and its drop guard removes unpublished files. Existing output paths are
never opened or replaced.

The writer uses Arrow and Parquet directly rather than migrating the upstream Polars layer.
`PreparedReock::from_wkb` and `PreparedPolsbyPopper::from_wkb` accept ordered Polygon or
MultiPolygon WKB rows that have already been aligned and transformed to a projected CRS. The
constructors reject malformed, empty, non-polygon, topologically invalid, degenerate, and
non-finite geometry. Polsby-Popper preparation also checks graph endpoints, polygon overlap,
missing shared boundaries, and impossible aggregate shared perimeter before constructing the
metric.

GeoParquet loading, CRS parsing and transformation, and general geometry operations remain in
Python, so the scoring engine does not add PROJ or duplicate GeoPandas. The Python bindings
snapshot and align NetworkX-compatible graphs and GeoDataFrames, register engine metrics, return
labeled district and plan tables, and stream BEN, XBEN, or finalized BENDL assignments into
atomic run directories.

The Rust and Python sides both use the official `binary-ensemble` 2.0.0 release.

## Standalone checks

Run these without changing Gerrytools' Python build:

```text
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo test --locked --manifest-path rust/Cargo.toml --all-features
cargo clippy --locked --manifest-path rust/Cargo.toml --all-targets --all-features -- -D warnings
```
