# Data

The data guides cover retrieving Census tables and processed geographic products. Start with the
overview for the practices every retrieval shares, then open the guide for the data family you
need.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`light-bulb` Overview
:link: overview
:link-type: doc

API keys, GEOID handling, processed geographic downloads, and error handling.
:::

:::{grid-item-card} {octicon}`law` Decennial PL 94-171
:link: decennial
:link-type: doc

Official population, VAP, group-quarters, and housing counts by state, county, tract, and block
group.
:::

:::{grid-item-card} {octicon}`graph` American Community Survey
:link: acs
:link-type: doc

Between-census estimates with margins of error, including the citizenship data PL does not carry.
:::

:::{grid-item-card} {octicon}`stack` Block-level CVAP estimates
:link: block_cvap
:link-type: doc

Estimated citizen voting-age population by race for every block, from PL and ACS together.
:::

:::{grid-item-card} {octicon}`code` Data API
:link: ../../api/data
:link-type: doc

Signatures and table-definition classes for every public data function.
:::

::::

## Join Census data to geography

`census()` returns a DataFrame indexed by GEOID. Reset that index to join the result to a
GeoDataFrame:

<!-- docs-test: skip -- requires the reader's Census API key and geographic file -->
```python
import geopandas as gpd
import us

from gerrytools.data import census

counties = gpd.read_file("data/ga_counties.gpkg")
population = census(us.states.GA, geometry="county", table="P1")

population = population.reset_index()

joined = counties.merge(
    population[["GEOID", "total_pop_20", "black_pop_20"]],
    on="GEOID",
    how="left",
)
```

Save the joined geography when later work should not depend on another network request:

<!-- docs-test: skip -- continues the external-data example above -->
```python
joined.to_file("data/ga_counties_with_population.gpkg", driver="GPKG")
```

The {doc}`geometry guide <../geometry>` covers dissolves, dual graphs, unit mappings, and plan
assignments.

```{toctree}
:hidden:
:maxdepth: 1

Overview <overview>
Decennial PL 94-171 <decennial>
American Community Survey <acs>
Block-level CVAP <block_cvap>
```
