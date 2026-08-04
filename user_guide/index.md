---
sd_hide_title: true
---

# GerryTools

```{div} sd-text-center sd-fs-2 sd-font-weight-bold
GerryTools
```

```{div} sd-text-center sd-fs-5 sd-text-secondary
Prepare redistricting data, analyze districting plans, and produce publication-ready figures.
```

---

```{image} https://readthedocs.org/projects/gerrytools/badge/?version=latest
:alt: Documentation status
:target: https://gerrytools.readthedocs.io/en/latest/
```

```{image} https://badge.fury.io/py/gerrytools.svg
:alt: PyPI package
:target: https://pypi.org/project/gerrytools/
```

GerryTools is a companion to [GerryChain](https://github.com/mggg/GerryChain). It collects the
data, geometry, plotting, LaTeX, and ensemble-runner tools used in practical redistricting
workflows. The guides are organized by public module.

## Install

Most users can install GerryTools from PyPI:

```console
pip install gerrytools
```

The Docker-backed ensemble runners are an optional extra:

```console
pip install "gerrytools[mgrp]"
```

See {doc}`user/install` for uv, virtual-environment, Jupyter, and Docker instructions.

## Start here

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download` Install GerryTools
:link: user/install
:link-type: doc

Set up Python, Jupyter, and the optional Docker-backed runners.
:::

:::{grid-item-card} {octicon}`package` Browse module guides
:link: user/intro
:link-type: doc

See how the public modules relate and open the guide for the interface you need.
:::

:::{grid-item-card} {octicon}`code` Browse the API
:link: api
:link-type: doc

Look up signatures, parameters, and public objects by module.
:::

::::

## Guides by module

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database` Data
:link: user/data/index
:link-type: doc

Retrieve PL 94-171, ACS, and CVAP tables plus processed 2020 geographic products.
:::

:::{grid-item-card} {octicon}`globe` Plan comparison
:link: user/geometry
:link-type: doc

Move between tabular, geographic, graph, and district-plan representations.
:::

:::{grid-item-card} {octicon}`graph` Plotting
:link: user/plotting/index
:link-type: doc

Build ensemble distributions, seats-votes figures, maps, and composable Matplotlib layouts.
:::

:::{grid-item-card} {octicon}`paintbrush` Colors
:link: user/colors
:link-type: doc

Use named colors, district palettes, colormaps, and palette previews.
:::

:::{grid-item-card} {octicon}`server` Ensemble runners
:link: user/mgrp
:link-type: doc

Run ReCom, Forest, and SMC samplers through the Docker-backed MGRP runners.
:::

:::{grid-item-card} {octicon}`file-binary` Recording chains
:link: user/ben
:link-type: doc

Record GerryChain runs as compact, self-describing BENDL ensemble files.
:::

:::{grid-item-card} {octicon}`meter` Scoring
:link: user/scoring/index
:link-type: doc

Score a few GerryChain plans in memory or stream a complete BENDL ensemble.
:::

:::{grid-item-card} {octicon}`typography` LaTeX output
:link: user/latex/index
:link-type: doc

Produce report tables and TeX-native figures with rendered examples.
:::

::::

```{toctree}
:hidden:
:caption: Start here
:maxdepth: 1

Module guide <user/intro>
Install GerryTools <user/install>
Tutorial data <user/data/example_data>
```

```{toctree}
:hidden:
:caption: Guides
:maxdepth: 2

Data <user/data/index>
Plan comparison <user/geometry>
Plotting <user/plotting/index>
Colors <user/colors>
Ensemble runners (MGRP) <user/mgrp>
Recording chains (BEN) <user/ben>
Scoring Plans & Ensembles <user/scoring/index>
LaTeX output <user/latex/index>
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

API reference <api>
Changelog <changelog>
```

```{toctree}
:hidden:
:caption: Help and project
:maxdepth: 1

Docker setup <topics/docker>
Contributing <topics/contributing>
```
