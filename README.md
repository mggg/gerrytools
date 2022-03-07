# evaltools 
A set of tools and resources for evaluating and visualizing proposed districting plans.

## Installation
### Use Case
Before installing, it's important to determine your use case. For discrete
evaluation tasks like measuring a district's compactness in a proposed plan,
you want to install the _tools_. For a more thorough ensemble-based set of tools,
check out this [complementary library](https://github.com/mggg/plan-evaluation-reporting).

### Instructions
If you want to use this package to evaluate districting plans, the recommended
way to install is by running
```
$ git clone https://github.com/mggg/evaltools.git
```
then navigate into the `evaltools` repository and run
```
$ python setup.py install
```
in your favorite CLI. This way, whenever changes are made, you can simply
`git pull` and they will be immediately usable by all programs importing
`evaltools`. Alternatively, you can install through pip using
```
$ pip install git+https://github.com/mggg/evaltools
```
although this may require frequent updating as the package iterates rapidly.

## Example Usage
### [Retrieving districtr data](#retrieving-districtr-data)
Let's say we want to create a "citizen ensemble" of districting plans – that is,
the collection of (possibly incomplete) districting plans drawn and submitted by
districtr users. To do so, we use the `evaltools.processing` subpackage.

```python
import us
from evaltools.processing import submissions, tabularized

# Set the state.
state = us.states.WI

# Retrieve submissions and tabularize them.
subs = submissions(state)
plans, cois, written = tabularized(state, subs)
```
Now, `plans`, `cois`, and `written` are pandas `DataFrame`s which contain
districting plan, community of interest, and written submissions, respectively.
Each of the DataFrames have the following columns:

| Column | Description |
| ------ | ----------- |
| `id` | districtr identifier. |
| `type` | Type of submission. |
| `title` | Title of the submission. |
| `districttype` | If a districting plan, the legislative chamber for which it's drawn. |
| `first`, `last`, `city` | Submitter name and location. |
| `datetime` | Submission timestamp. |
| `tags` | Submission tags. |
| `numberOfComments`, `comments` | Number of comments and comment text. |
| `text`, `draft` | Submission text; whether this submission is a draft. |
| `link` | Link to plan. |
| `units`, `unitsType` | Name of units; type of units (districtr-process). |
| `tileset` | Location of the plan's tileset. |
| `plan` | The actual mapping from unit unique identifiers to districts. |


### [Converting to alternate units](#converting-units)
In the [previous step](#retrieving-districtr-data), we saw how to get submissions from
districtr and convert them into a tabular format. If we want to study the citizen 
ensemble defined by the submissions, we need to put the districting assignments on
a common set of units: that's where the `unitmap()`, `invert()`, and `remap()` functions
come in handy.

Suppose we have our tabular data `tabs`, and we want to convert each of the
assignments in the `plans["plan"]` column to a common set of units. We first need
a mapping from each unit type to a base set of units; typically, these are 2020 Census
blocks. To create this mapping, we use the `unitmap()` function, which maps source
geometries (blocks) to target geometries (VTDs):

```python
import geopandas as gpd
import json
from evaltools.geography import unitmap, invert

# Read in geometric data.
vtds = gpd.read_file("<path>/<to>/<vtds>")
blocks = gpd.read_file("<path>/<to>/<blocks>")

# Create mapping from blocks to VTDs.
mapping = unitmap(blocks, vtds)

# Write the mapping to file.
with open("<path>/<to>/<destination>.json", "w"): json.dump(mapping, f)
```

Create a mapping for each set of units we wish to convert: for example, if the
Wisconsin citizen ensemble has plans on 2020 VTDs, 2020 Precincts, and 2016
Precincts, we should have mappings from each of these units to 2020 blocks. Once
these mappings have been created, we can use the `remap()` function on our `plans`
(or `cois`) dataframes to convert the districting assignments to 2020 blocks.

```python
import us
from evaltools.processing import submissions, tabularized, remap

# Set the state.
state = us.states.WI

# Retrieve submissions and tabularize them.
subs = submissions(state)
plans, cois, written = tabularized(state, subs)

# Create a dictionary of mappings for each unit type.
unitmaps = {
    "2020 VTDs": vtds20_to_blocks,
    "2020 Precincts": precincts20_to_blocks,
    "2016 Precincts": precincts16_to_blocks
}

# Re-map district assignments.
plans = remap(plans, unitmaps)
```

Here, we ensure that each of the keys in `unitmaps` corresponds to a unit type
in `plans["units"]`.

### Compressing districtr submissions
In the [previous step](#converting-units)), we saw how to get submissions
directly from the districtr database. Because each (plan- and COI-based) submission
contains a districting assignment, the total size of these assignments can be
prohibitively large. To help, we use the `AssignmentCompressor` class of the
`evaltools.processing` package. _Note that this compression is only necessary
when the size (generally >20MB) of the saved assignment data file renders it
impractical to easily store or share._

```python
from evaltools.processing import AssignmentCompressor
import geopandas as gpd

# Get identifiers for the assignment we're compressing. This method
# of compression assumes all assignments are on the *same units*: in this case,
# we assume that all assignments are on 2020 Census blocks.
identifiers = gpd.read_file("<path>/<to>/<blocks>")["BLOCKS20"]

# Create a new compressor, which we'll use to compress all the assignments
# we've generated in previous steps.
ac = AssignmentCompressor(identifiers, location="<compressed>.ac")

# The first method of compressing districts uses the `with` statement to create
# a safe context from which we can read compressed objects to the compressor.
with ac as compressor:
    for assignment in plans["plan"]:
        compressor.compress(assignment)

# The second method is just a wrapper for the above, which can help with code
# readability.
ac.compress_all(plans["plan"])
```
After compressing the plans, they'll be stored at the filepath in the `location`
parameter of the call to `AssignmentCompressor()`. We can decompress them using
the `.decompress()` method, ensuring that the `identifiers` are the _same_ as
those used during compression:

```python
...

ac = AssignmentCompressor(identifiers, location="<compressed>.ac")

for assignment in ac.decompress():
    <do whatever!>

...
```

### Reporting Statistics
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
It is _strongly_ recommended that users pre-compute dual graphs – _especially
those dual to large set of geometries, like Census blocks_ – as they are
computationally expensive and time-consuming to compute.

Next, we want to find the number of county pieces are induced by the districting
plan. We can do so using the `pieces` function from `evaltools.evaluation`,
assuming that the dual graph has a column assigning each vertex to a districting
plan:

```python
from evaltools.evaluation import pieces
from evaltools import Partition, Graph

# Read in the dual graph and create a Partition object. In this case, the dual
# graph has a column called `"DISTRICT"` which assigns each vertex to a district.
graph = Graph.from_file("<path>/<to>/<graph>.json")
districts = Partition(graph, "DISTRICT")

# Find the number of county pieces: note that `pieces()` consumes a list of unit
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
