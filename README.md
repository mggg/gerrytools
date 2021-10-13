# plan-evaluation-tools
A set of tools and resources for evaluating and visualizing proposed districting plans.

## Installation
### Use Cases
Before installing, it's important to determine your use case: if you want to install
the _tools_ to evaluate plans, then this is the repo for you; if you want the
_evaluation data_, then [head to this repo](https://github.com/mggg/plan-evaluation-reporting).

### Instructions
If you want to use this package to evaluate districting plans, the recommended
way to install is by running
```
$ pip install git+https://github.com/mggg/plan-evaluation-processing
```
in your favorite CLI. If you want to contribute to the development of this repo
(or not worry so often about having to pull new versions), run
```
$ git clone https://github.com/mggg/plan-evaluation-processing.git
```
then navigate into the `plan-evaluation-processing` repository and run
```
$ python setup.py install
```
This way, whenever changes are made, you can simply `git pull` and they will be
immediately usable by all programs importing `evaltools`.

## Example Usage
Let's say we want to find the number of county pieces induced by a districting
plan on VTDs; that is, the number of disjoint county chunks produced by the
districting plan. First, we want to create a dual graph for the underlying geometries,
where `VTDID20` is the unique identifier for each VTD; also ensure the geometries
have a column specifying a county assignment (typically, this column is something
like `COUNTYFP10`, `COUNTYFP20`, `COUNTY`, etc.).

```python
import geopandas as gpd
from evaltools.geography import dualgraph

vtds = gpd.read_file("<path>/<to>/<vtds>")
graph = dualgraph(vtds, index="VTDID20")
graph.to_file("<path>/<to>/<graph>.json")
```
It is **strongly** recommended that users pre-compute dual graphs – especially
those dual to geometries with a large number of polygons – as they are computationally
expensive to compute.

Next, we want to find the number of county pieces are induced by the districting
plan. We can do so using the `pieces` function from `evaltools.evaluation`,
assuming that the dual graph has an assignment column called `DISTRICT` which
denotes the district each vertex's district assignment:

```python
from evaltools.evaluation import pieces
from evaltools import Partition, Graph

# Read in the dual graph and create a Partition object.
graph = Graph.from_file("<path>/<to>/<graph>.json")
districts = Partition(graph, "DISTRICT")

# Find the number of county pieces: note that pieces consumes a list of unit
# names, so if we want to find the number of county and block group splits,
# we can pass a column corresponding to block group assignments as well (e.g
# ["COUNTYFP20", "BLOCKGROUP20"]).
chunks = pieces(districts, ["COUNTYFP20"])
```
`chunks` now contains a dictionary mapping the column names passed to the number
of unit pieces induced by the districting plan; for example, a result of
```python
chunks = {
    "COUNTYFP20": 16
}
```
indicates that the districting plan creates 16 pieces of county.

## Documentation
Read the documentation [here](https://mggg.github.io/plan-evaluation-tools/). To
create documentation after adding features, please ensure you're following the
[Google Python style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings), then run `sh docs.sh`. Pushing to the repository
will modify the documentation automatically.
