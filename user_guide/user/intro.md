# Module guide

GerryTools is organized into public modules that can be used independently or together.

| Guide                     | What it provides                                                 |
| ------------------------- | ---------------------------------------------------------------- |
| Data                      | Census tables, CVAP estimates, and processed geographic products |
| Plan comparison           | Plan overlap, relabeling, and population dispersion              |
| Plotting                  | Statistical plots, geographic plots, and Matplotlib composition  |
| Colors                    | Named colors, district palettes, colormaps, and palette previews |
| Ensemble runners (MGRP)   | Docker-backed Rust ReCom, Forest, and SMC runners                |
| Recording chains (BEN)    | Recording and replaying ReCom ensembles with GerryChain          |
| Scoring                   | In-memory GerryChain scoring and streamed BENDL evaluation       |
| LaTeX output              | TeX tables and TikZ-native figures                               |

The modules do not prescribe one end-to-end workflow. A typical analysis might retrieve Census
tables with Data, then prepare geometry and a dual graph with Plan comparison. It could generate
an ensemble with an MGRP runner or a recorded GerryChain, score the resulting plans, and pass those
results to Plotting or LaTeX output. Each guide documents the objects it accepts and the form of
its output, so projects can enter that sequence at the stage their existing data supports.

## Module guides

- {doc}`Data <data/index>`
- {doc}`Plan comparison <geometry>`
- {doc}`Plotting <plotting/index>`
- {doc}`Colors <colors>`
- {doc}`Ensemble runners <mgrp>`
- {doc}`Recording chains <ben>`
- {doc}`Scoring <scoring/index>`
- {doc}`LaTeX output <latex/index>`

The {doc}`../api` reference lists every public object and parameter.
