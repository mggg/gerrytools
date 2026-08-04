# Ensemble runners (MGRP)

`gerrytools.mgrp` is the Metric Geometry Replication interface. It runs three redistricting
engines in a pinned Docker environment. The Python interface prepares paths and validates engine
settings; the sampler itself runs inside the container. This keeps the native Rust, Julia, and R
dependencies out of the project's Python environment while giving each engine the same basic
execution pattern.

## Choose a runner

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Rust ReCom
:link: mgrp/recom
:link-type: doc

Run fast ReCom chains, region-aware proposals, short bursts, and tilted optimization on a dual
graph.
:::

:::{grid-item-card} Forest ReCom
:link: mgrp/forest
:link-type: doc

Run the hierarchy-aware Multi-Scale Map Sampler (MSMS) on a dual graph with one or more geographic
levels.
:::

:::{grid-item-card} Sequential Monte Carlo
:link: mgrp/smc
:link-type: doc

Generate a weighted ensemble with `redist` from an ESRI shapefile bundle.
:::

::::

The runners differ in both algorithm and input:

| Runner | Method | Input | Best fit |
| --- | --- | --- | --- |
| Rust ReCom | Markov chain Monte Carlo | NetworkX node-link JSON | Large ReCom ensembles and score-directed searches |
| Forest ReCom | Hierarchical Markov chain Monte Carlo | NetworkX node-link JSON | Ensembles that model nested geographic units explicitly |
| SMC | Sequential Monte Carlo | ESRI shapefile bundle | Independent weighted samples through `redist` |

These methods target different distributions. Choosing among them is a methodological decision,
not a performance toggle. Each detailed guide explains the sampler-specific settings and output
contracts needed to reproduce a run.

## The shared execution model

Every run has three parts:

1. A runner configuration identifies the host input, output directory, and log directory.
2. A run-info object contains the sampler settings and validates them before Docker starts.
3. `RunContainer` mounts the input read-only, runs the configured engine, writes stderr to a log,
   and removes the temporary container when the `with` block exits.

When the container starts, `RunContainer` tries to pull the pinned `mgggdev/replicate` image and
falls back to a local copy if the pull fails. The running container has networking disabled. See
{doc}`Docker setup <../topics/docker>` for installation and daemon troubleshooting.

Output files are written in a subdirectory named for the input. When `output_file_name` is
absent, the runner combines readable headline settings with a short hash of the complete
configuration. Two materially different runs therefore do not silently share the same derived
output path. `run()` returns the primary output path; `expected_files()` also reports metadata,
tally, assignment, or optimizer-score sidecars produced by that runner.

The detailed tutorials use placeholder paths such as `data/dual_graph.json`. Replace those paths
and column names with the project's inputs. The configuration examples execute without Docker;
the sampling calls are displayed but not executed. Run them locally once Docker and the input data
are available.

## Related

- {doc}`Recording chains with BEN <ben>`
- {doc}`Scoring <scoring/index>`
- {doc}`MGRP API <../api/mgrp>`

```{toctree}
:hidden:
:maxdepth: 1

Rust ReCom <mgrp/recom>
Forest ReCom <mgrp/forest>
Sequential Monte Carlo <mgrp/smc>
```
