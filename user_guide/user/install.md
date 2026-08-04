# Installation and environments

GerryTools requires Python 3.11 or newer. A project-specific virtual environment is strongly
recommended because the geospatial stack includes compiled packages whose versions must agree.

## Install with uv

[uv](https://docs.astral.sh/uv/) manages Python, the environment, and the dependency record for a
project. Create a project and add GerryTools:

```console
uv init redistricting-analysis
cd redistricting-analysis
uv add gerrytools
```

Run scripts through the environment without activating it:

```console
uv run python analysis.py
```

For a notebook workflow, add Jupyter and start it through uv:

```console
uv add jupyter ipykernel
uv run jupyter lab
```

Select the interpreter from the project's `.venv` when Jupyter or VS Code asks for a kernel.

## Install with pip and venv

If you already manage Python yourself, create and activate a virtual environment before installing
GerryTools:

```console
python -m venv .venv
```

Activate it on macOS or Linux:

```console
source .venv/bin/activate
```

Activate it in PowerShell on Windows:

```console
.venv\Scripts\Activate.ps1
```

Then install the package:

```console
python -m pip install --upgrade pip
python -m pip install gerrytools
```

Using `python -m pip` makes it explicit which Python receives the package.

## Optional Docker-backed runners

The `gerrytools.mgrp` module requires the Docker Python SDK and a running Docker engine. Install
the extra with either tool:

```console
uv add "gerrytools[mgrp]"
```

```console
python -m pip install "gerrytools[mgrp]"
```

Then install and start Docker Desktop, Docker Engine, or another compatible daemon. Continue with
{doc}`../topics/docker` before trying an ensemble run.

## Check the environment

First confirm that the Python you are running belongs to the project:

```console
python -c "import sys; print(sys.executable)"
```

The path should point into the project's `.venv`. Under uv, use `uv run python` in place of
`python` for each check.

Next import GerryTools and report its installed version:

```console
python -c "import importlib.metadata as md; import gerrytools; print(md.version('gerrytools'))"
```

Finally, check the core geographic and plotting dependencies:

```console
python -c "import geopandas, matplotlib, gerrychain; print('geospatial and plotting imports work')"
```

If the command-line checks pass but a notebook import fails, the notebook is using a different
kernel. Run this in a notebook cell and compare it with the command-line result:

```python
import sys

print(sys.executable)
```

## Run the tutorials locally

The {doc}`tutorial data page <data/example_data>` provides direct downloads for the plotting and
scoring examples. Save those files in a project `data/` directory rather than relying on the
documentation repository's relative paths.

## Install a development checkout

Contributors should clone the repository and let the project tasks create the complete environment:

```console
git clone https://github.com/mggg/gerrytools.git
cd gerrytools
task setup
```

Use `task --list-all` to inspect the available checks. The documentation workflow is described in
{doc}`../topics/contributing`.

## Frequent installation problems

`ModuleNotFoundError: No module named 'gerrytools'`
: The package was installed into a different Python. Compare `sys.executable` in the failing
  environment with the environment used for installation.

`ModuleNotFoundError: No module named 'docker'`
: Install the `mgrp` extra. The core package can be used without Docker, so the SDK is optional.

An error importing GeoPandas, Shapely, or PyProj
: Create a fresh environment and reinstall rather than replacing compiled packages one at a time.
  Mixing packages from multiple environment managers is a common cause.

A Docker connection or permission error
: The Python dependency is installed, but the Docker daemon is not running or the current user
  cannot reach it. See {doc}`../topics/docker`.
