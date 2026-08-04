# Tutorial data

The notebooks use compact committed datasets so every example builds without a network connection
or credentials. They are documentation fixtures, not replacements for current official data.

## `census/`

Small real extracts fetched once with the public `gerrytools.data` functions and committed so the
data guides (`data/decennial`, `data/acs`, `data/block_cvap`) run offline. Each file is the exact
frame returned by the call named in the guide next to it: decennial PL tables for Georgia (2020
statewide, county-level P1/P2/P3, and 2010 statewide), ACS 2023 5-year and 1-year pulls for
Georgia (default tables, VAP + Hispanic-by-race, CVAP, and the ungrouped VAP cells), and
District of Columbia block-CVAP estimates (`acs_year=2023`, `pl_year=2020`, thresholds 20 and
500). Regenerate them by re-running those calls with a `CENSUS_API_KEY` set and saving with
`DataFrame.to_csv` (gzipped for the block files).

## `ga_2016_precincts.gpkg`

This GeoPackage contains 2,664 Georgia precinct geometries from the predecessor GerryTools example,
with 2016 presidential and U.S. Senate returns, demographic counts, and congressional, state House,
and state Senate district labels.

The documentation copy:

- retains only the columns needed by the tutorials;
- transforms geometry to NAD83 / Conus Albers (EPSG:5070);
- applies coverage-aware simplification at a 25-metre tolerance so shared precinct boundaries stay
  aligned;
- stores all features as multipolygons.

### Schema

| Columns                                    | Meaning                                                    |
| ------------------------------------------ | ---------------------------------------------------------- |
| `ID`, `PRECINCT_N`, `CTYNAME`              | Precinct and county identifiers                            |
| `PRES16D`, `PRES16R`, `PRES16L`            | 2016 presidential vote counts                              |
| `SEN16D`, `SEN16R`, `SEN16L`               | 2016 U.S. Senate vote counts                               |
| `TOTPOP`, `VAP`                            | Total population and voting-age population counts          |
| `NH_WHITE`, `NH_BLACK`, `NH_ASIAN`, `HISP` | Population counts used by the dot-density guide            |
| `WVAP`, `BVAP`, `HVAP`                     | Voting-age population counts                               |
| `CD`, `HDIST`, `SEND`                      | Source congressional, state House, and state Senate labels |
| `geometry`                                 | Simplified precinct geometry in EPSG:5070                  |

Election columns are counts, not percentages. The source assignment and election vintage are
retained for teaching aggregation and plotting; they are not the current Georgia districts,
precincts, or election record.

## `ga_congressional_plans.csv.gz` and `ga_congressional_plans.gpkg`

These files support the geometry guide's comparison of Georgia's 2021 and 2023 congressional
plans. The compressed CSV contains every Georgia 2020 Census block with positive population,
2020 Census P1 total population, and the block's assignments in the two plans. Zero-population
blocks are omitted because they do not affect population overlap or dispersion.

The GeoPackage contains fourteen district polygons for each plan in NAD83 / Conus Albers
(EPSG:5070). The polygons were dissolved from all 232,717 Georgia blocks, including blocks with
zero population, then simplified as separate coverages at a 100-metre tolerance. The simplified
geometry is suitable for the guide's maps and area comparison; use unsimplified source geometry
when exact area is the subject of an analysis.

### Schema

| File | Columns | Meaning |
| ---- | ------- | ------- |
| CSV | `GEOID20`, `TOTPOP20` | 2020 Census block identifier and P1 total population |
| CSV | `CD_2021`, `CD_2023` | Congressional district under the 2021 and 2023 plans |
| GeoPackage | `PLAN`, `DISTRICT`, `geometry` | Plan vintage, district label, and dissolved polygon |

`CD_2021` comes from the Census Bureau's [118th-Congress block-equivalency file][cd118],
representing Georgia SB 2EX (2021). `CD_2023` comes from the [119th-Congress
block-equivalency file][cd119], representing Georgia SB 3EX (2023). Both files use 2020 Census
blocks, so the assignments join exactly to the P1 population table.

[cd118]: https://www.census.gov/geographies/mapping-files/2023/dec/rdo/118-congressional-district-bef.html
[cd119]: https://www.census.gov/geographies/mapping-files/2025/dec/rdo/119-congressional-district-bef.html

## `ga_congressional_ensemble_a.json` and `ga_congressional_ensemble_b.json`

These files contain two 1,000-plan Georgia congressional demonstration ensembles generated from
the 2020 VTD graph in the V2 Districtr Georgia release package. Both ReCom chains start from the
same generated seed, enforce a 2% population tolerance on `total_pop_20`, and include the seed as
step zero. They use independent random streams.

Every record contains a `step` value; fourteen district-level `BVAP` and `WVAP` shares; and a
`disproportionality` mapping with scores for thirteen statewide elections from 2016 through 2024.
The election suite covers president, U.S. Senate, governor, lieutenant governor, attorney general,
and secretary of state. The shares divide the corresponding population count by `total_vap_20`.
Each score subtracts statewide two-party Democratic vote share from Democratic seat share, so
positive values indicate a larger share of seats than votes and negative values indicate a
smaller share.

The runs are compact plotting fixtures, not validated ensembles for substantive analysis. They
exist to demonstrate rank ordering, comparisons between datasets, and builder behavior.

## `ga_congressional_disproportionality_100000.parquet`

This file contains signed Democratic disproportionality scores for 100,000 positions in a Georgia
congressional ReCom-B chain, including the seed at step zero. The chain was generated on the 2020
VTD graph in the V2 Districtr Georgia release package with RustReCom 0.2.0, a 2% population
tolerance on `total_pop_20`, RNG seed 2026080203, one thread, and batch size one. The starting
assignment is a generated population-balanced seed.

The `step` column runs from 0 through 99,999. The other thirteen columns contain scores for
statewide elections from 2016 through 2024: `AG18`, `AG22`, `GOV18`, `GOV22`, `LTG18`, `LTG22`,
`PRES16`, `PRES20`, `PRES24`, `SEN16`, `SEN20`, `SEN22`, and `SOS22`. Each score subtracts
statewide two-party Democratic vote share from Democratic seat share. The scatter plot guide
computes each plan's mean and variance across these columns.

The Parquet file retains the scored chain positions needed by the tutorial, not the assignment
vectors or the intermediate RustReCom JSONL output.

## `co_vtd_scoring_10000.bendl`

This BENDL bundle is the offline fixture for the scoring guide. It contains a 10,000-step
Colorado VTD ReCom-B chain, its dual graph and RustReCom metadata, and two custom assets:
`co_vtds_2020.parquet` and `fixture_metadata.json`. The GeoParquet has 3,158 unsimplified VTD
geometries in NAD83 / Conus Albers (EPSG:5070), with total population, voting-age population,
Black voting-age population, 2016/2020/2024 presidential votes, counties, and the seed assignment.

The chain was generated with district-pairs MST, a 5% population tolerance, RNG seed 2026072701,
one thread, and batch size one. It uses the `gerrydb_graph_edge` adjacency in
`co_districtr_vtd_view_v2.gpkg`; shared perimeter is stored on graph edges. The seed comes from
the Colorado block assignment retained for the 15-state benchmark. Source district labels 1-8
were normalized to 0-7.

Assigning split VTDs from their internal points left two tiny singleton components. The fixture
moves `vtd:08013013400-datadem-1` (141 people) from source district 2 to normalized district 6
and `vtd:08005005475` (52 people) from source district 6 to normalized district 0, their only
adjacent districts. After these repairs every seed district is connected, and the maximum
population deviation is 1.12%.
