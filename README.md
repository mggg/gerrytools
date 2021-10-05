# plan-evaluation-tools
A set of tools and resources for evaluating and visualizing proposed districting plans.

## Installation
### Use Cases
Before installing, it's important to determine your use case: if you want to install
the _tools_ to evaluate plans, then this is the repo for you; if you want the
_evaluation data_, then [head to this repo](https://github.com/mggg/plan-evaluation).

### Instructions
If you want to use this package to evaluate districting plans, the recommended
way to install is by running
```
$ pip install git+https://github.com/mggg/plan-evaluation-tools
```
in your favorite CLI. 

## Usage
### Mapping
### `mapping.drawplan`
Before creating the plan map, we need geometric data for each district.
Generally, this geometric data is a GeoDataFrame with one row per district;
if we wish to overlay additional shapes (e.g counties) on the districting plan,
we should have geometric data for those as well.

```python
from plan_evaluation.mapping import drawplan
import geopandas as gpd
import matplotlib.pyplot as plt

# Read in geometric data.
districts = gpd.read_file("<path>/<to>/<districts>")
counties = gpd.read_file("<path>/<to>/<counties>")

# Specify the column on the GeoDataFrame which contains the districts' names.
name = "DISTRICTN"

# Plot, overlaying county shapes on top of our district map.
ax = drawplan(districts, name, overlay=counties)
```

Because `drawplan` returns a `matplotlib.Axes` object, it can be manipulated as
typical matplotlib plots are. Additionally, `drawplan` has a `numbers` keyword
argument which, if `True`, plots the name of each district at its centroid:

```python
# Plot, overlaying county shapes and plotting district numbers.
ax = drawplan(districts, name, overlay=counties, numbers=True)
```

Internally, `drawplan` attempts to convert district names to integers and sort
them to get proper color orderings, but the name of the district (as specified
by the `assignment` parameter) is plotted when `numbers` is `True`.
