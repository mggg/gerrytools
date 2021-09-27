# plan-evaluation
A set of tools for evaluating and visualizing proposed districting plans.

## Installation
```
pip install git+https://github.com/mggg/plan-evaluation
```

## Usage
### Mapping
Before creating the plan map, we need geometric data for each district. Generally,
this geometric data is a GeoDataFrame with one row per district; if we wish to overlay
additional shapes (e.g. counties) on the districting plan, we should have
geometric data for those as well.

```python
from plan_evaluation import map_districts
import geopandas as gpd
import matplotlib.pyplot as plt

# Read in geometric data.
districts = gpd.read_file("<path>/<to>/<districts>")
counties = gpd.read_file("<path>/<to>/<counties>")

# Specify the column on the GeoDataFrame which contains the districts' names.
name = "DISTRICTN"

# Plot, overlaying county shapes on top of our district map.
ax = map_districts(districts, name, overlay=counties)
```

Because `map_districts` returns a `matplotlib.Axes` object, it can be manipulated
as typical matplotlib plots are. Additionally, `map_districts` has a `numbers`
keyword argument which, if `True`, plots the name of each district at its
centroid:

```python
# Plot, overlaying county shapes and plotting district numbers.
ax = map_districts(districts, name, overlay=counties, numbers=True)
```

Internally, `map_districts` attempts to convert district names to integers and
sort them to get proper color orderings, but the name of the district (as
specified by the `assignment` parameter) is plotted when `numbers` is `True`.