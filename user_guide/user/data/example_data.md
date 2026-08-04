# Tutorial data

<div style="text-align: center;"><a class="sd-sphinx-override sd-btn sd-text-wrap sd-btn-primary reference external" href="https://www.dropbox.com/scl/fo/s22x9phl0hldiakn8nbuz/ABKfxHBaak5ra3eBGkNFWMM?rlkey=igpo7qi07oz5tfgjki317o79t&amp;st=gcxkicnc&amp;dl=1">Download tutorial data</a></div>

The tutorials use the datasets in this bundle. Extract it so the `data` directory is next to the
tutorial notebook:

```text
tutorial.ipynb
data/
```

The individual files are also available below.

## Georgia precinct geography

{download}`Download the Georgia precinct GeoPackage <../../_static/data/ga_2016_precincts.gpkg>`
(9.4 MB).

The file contains 2,664 precinct geometries, 2016 presidential and U.S. Senate returns,
demographic counts, and source congressional, state House, and state Senate assignments. The
geometry is projected to EPSG:5070 and simplified for documentation. It is used throughout the
geographic and statistical plotting guides.

After saving it as `data/ga_2016_precincts.gpkg`, open it with:

<!-- docs-test: skip -- requires the reader to download the linked GeoPackage -->
```python
import geopandas as gpd
from pathlib import Path

data_dir = Path("data")
precincts = gpd.read_file(data_dir / "ga_2016_precincts.gpkg")
print(precincts.shape)
print(precincts.crs)
```

## Georgia demonstration ensembles

{download}`Download Georgia ensemble A
<../../_static/data/ga_congressional_ensemble_a.json>` (947 KB) or
{download}`Georgia ensemble B <../../_static/data/ga_congressional_ensemble_b.json>` (948 KB).

Each file contains 1,000 plans with fourteen district-level Black and White voting-age population
shares per plan, plus signed Democratic disproportionality scores for thirteen statewide elections
from 2016 through 2024. The two independent ReCom runs support examples that compare several
datasets and summarize partisan performance across elections. They are compact plotting fixtures
rather than ensembles for substantive analysis.

<!-- docs-test: skip -- requires the reader to download the linked JSON file -->
```python
import json

records = json.loads((data_dir / "ga_congressional_ensemble_a.json").read_text(encoding="utf-8"))
print(len(records), records[0].keys())
```

{download}`Download the 100,000-position Georgia disproportionality scores
<../../_static/data/ga_congressional_disproportionality_100000.parquet>` (378 KB).

This Parquet file contains signed Democratic disproportionality scores for thirteen statewide
elections from 2016 through 2024. It supports the mean-versus-variance example in the scatter plot
guide without requiring readers to score the underlying chain.

<!-- docs-test: skip -- requires the reader to download the linked Parquet file -->
```python
import pandas as pd

scores = pd.read_parquet(data_dir / "ga_congressional_disproportionality_100000.parquet")
print(scores.shape)
```

## Colorado scoring bundle

{download}`Download the Colorado scoring BENDL bundle
<../../_static/data/co_vtd_scoring_10000.bendl>` (17 MB).

The bundle contains a 10,000-step ReCom chain, its dual graph, projected VTD geometry, population
and election columns, and fixture provenance. The {doc}`BENDL scoring guide <../scoring/bendl>` explains
how to verify and inspect its embedded resources before evaluation.

## Census examples

The Census guides show the live `gerrytools.data` calls and execute against small committed
response extracts. In a project, make the live call with your API key and save the returned frame
locally. The {doc}`data overview <overview>` explains credentials, GEOIDs, and rate limits.
