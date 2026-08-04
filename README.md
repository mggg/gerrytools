# gerrytools

[![tests](https://github.com/mggg/gerrytools/actions/workflows/tests.yml/badge.svg)](https://github.com/mggg/gerrytools/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/gerrytools.svg)](https://badge.fury.io/py/gerrytools)
[![docs](https://img.shields.io/badge/%E2%93%98-Documentation-%230099cd)](https://gerrytools.readthedocs.io/en/latest/)
[![website](https://img.shields.io/badge/%F0%9F%8C%90%20-DDRI%20Lab-%230099cd)](https://data-democracy.org)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-Ruff-d7ff64.svg)](https://docs.astral.sh/ruff/)

A companion to [GerryChain](https://github.com/mggg/GerryChain), GerryTools provides tools for
retrieving redistricting data, recording and scoring ensembles, comparing districting plans, and
producing publication-ready figures and tables. It is developed by the
[Data and Democracy Research Initiative](https://data-democracy.org) and distributed under the
[3-Clause BSD License](https://opensource.org/licenses/BSD-3-Clause).

## Installation

GerryTools requires Python 3.11 or newer. Published wheels include the compiled scoring engine;
building from source requires a stable Rust toolchain and a C++ compiler.

To install GerryTools from [PyPI](https://pypi.org/project/gerrytools/), run:

```console
pip install gerrytools
```

The `gerrytools.ben` API is included in the base installation. To use the Docker-backed ensemble
runners in `gerrytools.mgrp`, install the optional dependency:

```console
pip install "gerrytools[mgrp]"
```

The runners also require Docker Desktop, Docker Engine, or another compatible daemon. See the
[Docker documentation](https://gerrytools.readthedocs.io/en/latest/topics/docker/) for platform
requirements and setup instructions.

## Usage

GerryTools is organized into public modules that can be used independently or together:

- **`gerrytools.data`** retrieves decennial Census and American Community Survey tables, estimates
  block-level citizen voting-age population, and downloads processed 2020 geographic products.

- **`gerrytools.plan_comparison`** compares plans by population or area, finds optimal district
  relabelings, and measures population dispersion.

- **`gerrytools.plotting`** provides composable statistical and geographic plot builders that work
  with Matplotlib figures and axes.

- **`gerrytools.colors`** provides named colors, district palettes, color conversion, and
  Matplotlib and seaborn colormaps.

- **`gerrytools.mgrp`** runs Rust ReCom, Forest ReCom, Sequential Monte Carlo, and optimization
  workflows through a versioned Docker image.

- **`gerrytools.ben`** records GerryChain runs as self-describing BENDL files and reconstructs
  recorded partitions for lookup, subsampling, and analysis.

- **`gerrytools.scoring`** prepares graph and geometry resources once, evaluates plans with the
  compiled Rust scoring engine, and provides one-shot functions and array formulas.

- **`gerrytools.latex`** builds LaTeX documents and tables plus TikZ-native paintball and
  seats-votes figures.

See the [module guides](https://gerrytools.readthedocs.io/en/latest/user/intro/) for examples and
the [API reference](https://gerrytools.readthedocs.io/en/latest/api/) for signatures and parameters.

## Contributing

GerryTools is an active project, and has multiple contributors. If you'd like to contribute, here
are a few house rules:

1. Install `task` and `uv`, then run `task setup` from the repository root. Use
   `task --list-all` to see the available development workflows.

1. Run `task check` before opening a pull request. Use `task format` for Ruff formatting and
   import sorting, `task lint` for Ruff linting, `task test` for the test suite, and `task docs`
   to build the documentation locally.

1. **Write tests.** All changes, major or minor, **must** be accompanied by testing code. Code and
   tests will be immediately reviewed by Lab maintainers.

1. Test coverage must stay **at least** the same; this can be checked by running
   `uv run pytest --cov=gerrytools --cov-report=term-missing` after the tests are added to
   `tests/`.

1. **Write documentation.** Document public APIs and non-obvious invariants. Prefer clear names
   and focused functions over comments that merely restate the code.

We look forward to your contributions!
