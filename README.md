
# gerrytools

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/mggg/gerrytools/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/mggg/gerrytools/tree/main) 
[![codecov](https://codecov.io/gh/mggg/gerrytools/branch/main/graph/badge.svg?token=O09GYF7C9X)](https://codecov.io/gh/mggg/gerrytools) 
[![PyPI version](https://badge.fury.io/py/gerrytools.svg)](https://badge.fury.io/py/gerrytools) 
[![docs](https://img.shields.io/badge/%E2%93%98-Documentation-%230099cd)](https://mggg.github.io/gerrytools/) 
[![website](https://img.shields.io/badge/%F0%9F%8C%90%20-MGGG%20Redistricting%20Lab-%230099cd)](https://mggg.org) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


A companion to [GerryChain](https://github.com/mggg/GerryChain), GerryTools is
a robust suite of geometric and algorithmic tools to analyze districting plans
and related data. GerryTools is actively developed and used by the
[MGGG Redistricting Lab](https://mggg.org) and our collaborators to prepare
accurate, precise, and clean information for our projects. It is distributed
under a [3-Clause BSD License](https://opensource.org/licenses/BSD-3-Clause).


## Installation

### Using `pip` (recommended)

To install GerryTools from [PyPi](https://pypi.org/project/gerrytools/), run

```console
pip install gerrytools
```

from the command line. if you would like to use the `mgrp` and `ben` modules as well,
you can invoke 

```console
pip install gerrytools[mgrp]
```

you will need to make sure that [Docker Desktop](https://www.docker.com/get-started/)
is installed on your machine an updated to version >= 4.28.0. For more information on
getting this set up, please see 
[our documentation page](https://gerrytools.readthedocs.io/en/latest/topics/docker/)

## Usage

GerryTools is split up into multiple sub-packages, each designed to simplify and
standardize redistricting workflows.

* **`gerrytools.ben`** BEN (binary-ensemble) is our general purpose compression
    algorithm for working with ensembles of plans. In general, the ben algorithm can
    improve the storage of an ensemble of plans by an order of magnitude. When combined
    with the special XBEN (eXtreme BEN) portion of the algorithm, many ensembles of
    plans can be compressed small enough to fit into an email (~25Mb).

* **`gerrytools.data`** deals with the retrieval and processing of data. Here, you can
    find tools for grabbing decennial Census ('10 and '20), ACS 5-year ('12-'20), ACS CVAP
    Special Tab ('12-'20), districtr portal, and 2020 decennial Census geometric data. You
    can also find tools for moving CVAP data to other levels of geometry (e.g. prorating
    2019 CVAP on 2019 Census tracts to 2020 blocks).

* **`gerrytools.geometry`** provides facilities for dealing with geometric and related
    data. There are tools for translating and evaluating GerryChain
    [`Partition`](https://mggg.github.io/GerryChain/api.html#module-gerrychain.partition)s, 
    performing fast geometric dissolutions, creating unit maps (e.g. 2020 blocks to
    2020 VTDs), creating 
    [dual graphs for GerryChain](https://mggg.github.io/GerryChain/api.html#adjacency-graphs),
    and optimization algorithms for renaming districts.

* **`gerrytools.mgrp`** this module uses a Docker container to allow users to access several
    ensemble methods for generating districting plans on a state. In particular our Rust
    implementation of our `gerrychain` library, `frcw`, the Julia implementation of 
    [Forest Recom](https://arxiv.org/pdf/2008.08054.pdf), and the R/C++ implementation of
    [Sequential Monte Carlo (SMC)](https://github.com/alarm-redist/redist) are available
    through this module. 

* **`gerrytools.plotting`** contains methods for generating extremely
    high-quality Lab-standard data visualizations.

* **`gerrytools.scoring`** provides a vast array of redistricting plan scores.
    These can be used standalone _or_ as GerryChain 
    [updaters](https://mggg.github.io/GerryChain/api.html#module-gerrychain.updaters).

* **`gerrychain.utilities`** has ease-of-use methods for renaming
    directories containing shapefiles (which comes in handy more often than you'd
    think) and making JSON objects out of Python objects (useful when trying to
    organize information for many districting plans in a standard format).

<!-- ### Example

GerryTools is easy to use and is designed to simplify data- and redistricting-related
workflows. For example, let's say we want to analyze Alabama's citizen voting-age
population data on 2020 Census tracts. First, we can download the geometric data
for the state using `gerrytools.data.geometry20()`:

```python
from gerrytools.data import geometry20
from us import states

# Retrieves Lab-processed and -cleaned geometry data, and writes it to file.
geometry20(states.AL, "~/project/AL-tracts.zip", geometry="tract")
```
Next, we can download cleaned citizen voting-age population (CVAP) data from the 2020 ACS special tabulation release:

```python
from gerrytools.data import cvap
from us import states

# A pandas DataFrame of 2020 CVAP data for the state of Alabama at the 2020 Census
# tract level.
data = cvap(states.AL, geometry="tract", year=2020)
data.to_csv("~/project/AL-cvap.csv")
```

Finally, we can merge the two datasets, attaching the CVAP demographic data to
the geometric data:

```python
import geopandas as gpd
import pandas as pd

# Import geometric and demographic data.
geometric = gpd.read_file("~/project/AL-tracts.zip")
demographic = pd.read_csv("~/project/AL-cvap.csv")

# Merge. Note that the DataFrames have different unique identifier columns --- this
# is intentional, and serves to differentiate datasets of varying geometric levels.
merged = geometric.merge(demographic, left_on="GEOID20", right_on="TRACT20")
```

And there we are — what once took hours of setup and parsing now takes less than a
minute. -->

## Contributing

GerryTools is an active project, and has multiple contributors. If you'd like to
contribute, here are a few house rules:

1. After cloning this repository, run `sh setup.sh` to download and install
necessary git hooks and linting configurations.

2. **Follow the [PEP8 style guide](https://peps.python.org/pep-0008/)**. After
installing the above git hooks, linting is performed before every push. PEP8 errors can be automatically corrected by running `autopep8 --in-place --aggressive -r gerrytools` on the command line from the root directory.

3. **Write tests.** All changes, major or minor, **must** be accompanied by testing
code. Code and tests will be immediately reviewed by Lab maintainers.

4. Test coverage must stay **at least** the same; this can be checked by running
`pytest --cov=evaltools` after the tests are added to `tests/`.

5. **Write documentation.** All changes should be documented via docstrings,
and code should be repletely commented. It's much easier to decipher commented
code! Docstring documentation is compiled on every commit via git hooks.

We look forward to your contributions!
