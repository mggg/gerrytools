import io
from urllib.request import urlopen
from zipfile import ZipFile

import censusdata
import pandas as pd


def cvap(state, geometry="tract", year=2020) -> pd.DataFrame:
    """
    Retrieves and CSV-formats 5-year CVAP data for the provided state at
    the specified geometry level. Geometries from the **2010 Census**.
    Variables and descriptions are [listed here](https://tinyurl.com/3mnrm56s).

    Args:
        state (us.State): The `State` object for which we're retrieving 2019
            ACS CVAP Special Tab.
        geometry (str, optional): Level of geometry for which we're getting
            data. Accepted values are `"block group"` for 2010 Census Block
            Groups, and `"tract"` for 2010 Census Tracts. Defaults to `"tract"`.
        year (int, optional): Year for which data is retrieved. Defaults to
            2020.

    Returns
        A `DataFrame` with a `GEOID` column and corresponding CVAP columns from
        the ACS CVAP Special Tab for the specified year.
    """
    # Maps line numbers to descriptors.
    descriptions = {
        1: "CVAP",
        2: "NHCVAP",
        3: "NHAMINCVAP",
        4: "NHASIANCVAP",
        5: "NHBLACKCVAP",
        6: "NHNHPICVAP",
        7: "NHWHITECVAP",
        8: "NHWHITEAMINCVAP",
        9: "NHWHITEASIANCVAP",
        10: "NHWHITEBLACKCVAP",
        11: "NHBLACKAMINCVAP",
        12: "NHOTHCVAP",
        13: "HCVAP",
    }

    # First, load the raw data requested; allowed geometry values are
    # "block group" and "tract."
    if geometry not in {"block group", "tract"}:
        print(f'Requested geometry "{geometry}" is not allowed; ' "loading tracts.")
        geometry = "tract"

    abbrv = geometry if geometry == "tract" else "block group"

    # Load the raw data.
    raw = _raw(abbrv, year)

    # Create a STATE column for filtering and remove all rows which don't match
    # the state FIPS code.
    raw["GEOID"] = raw["geoid"].str.split("US").str[1]
    raw["STATE"] = raw["GEOID"].str[:2]
    instate = raw[raw["STATE"] == str(state.fips)]

    # Now that we have the in-state data, we aim to pivot the table. Because
    # the ACS data is in a line-numbered format (i.e. each chunk of 13 lines
    # matches to an individual geometry, and each of the 13 lines describes
    # an individual statistic) we need to first collapse each chunk of 13
    # lines, then build a dataframe from the resulting collapsed lines.
    # First we send the dataframe to a list of records.
    instate_records = instate.to_dict(orient="records")
    collapsed = []

    # Get year stuff.
    decade = "10" if year < 2020 else "20"
    yearsuffix = str(year)[2:]

    # Next, we collapse these records to a single record.
    for i in range(0, len(instate_records), 13):
        # Create an empty records.
        record = {}

        # For each of the records in the block, "collapse" them into a single
        # record.
        block = instate_records[i : i + 13]
        for line in block:
            record[geometry.replace(" ", "").upper() + decade] = line["GEOID"]
            record[descriptions[line["lnnumber"]] + yearsuffix] = line["cvap_est"]
            record[descriptions[line["lnnumber"]] + f"{yearsuffix}e"] = line["cvap_moe"]

        collapsed.append(record)

    # Create a dataframe and a POCCVAP column; all people minus non-Hispanic
    # White.
    data = pd.DataFrame().from_records(collapsed)
    data[f"POCCVAP{yearsuffix}"] = (
        data[f"CVAP{yearsuffix}"] - data[f"NHWHITECVAP{yearsuffix}"]
    )

    return data


def acs5(
    state, geometry="tract", year=2020, columns=[], white="NHWHITEVAP"
) -> pd.DataFrame:
    """
    Retrieves ACS 5-year population estimates for the provided state, geometry
    level, and year. Also retrieves ACS-reported CVAP data, which closely
    matches that reported by the CVAP special tabulation; CVAP data are only
    returned at the tract level, and are otherwise reported as 0.

    Args:
        state (us.State): `State` object for the desired state.
        geometry (str, optional): Geometry level at which data is retrieved.
            Acceptable values are `"tract"` and `"block group"`. Defaults to
            `"tract"`, so data is retrieved at the 2020 Census tract level.
        year (int, optional): Year for which data is retrieved. Defaults to
            2020.
        columns (list, optional): Columns to retrieve. If `None`, a default set
            of columns including total populations by race and ethnicity and
            voting-age populations by race and ethnicity are returned, along
            with a GEOID column.
        white (str, optional): The column removed from totals when calculating
            POC populations.

    Returns:
        A DataFrame containing the formatted data.
    """
    # Columns for total populations.
    yearsuffix = str(year)[-2:]
    popcolumns = {
        "B01001_001E": "TOTPOP" + yearsuffix,
        "B03002_003E": "WHITE" + yearsuffix,
        "B03002_004E": "BLACK" + yearsuffix,
        "B03002_005E": "AMIN" + yearsuffix,
        "B03002_006E": "ASIAN" + yearsuffix,
        "B03002_007E": "NHPI" + yearsuffix,
        "B03002_008E": "OTH" + yearsuffix,
        "B03002_009E": "2MORE" + yearsuffix,
        "B03002_002E": "NHISP" + yearsuffix,
    }

    # Create a dictionary of column groups.
    groups = {}

    # Get VAP columns. The columns listed here are by race, irrespective of
    # ethnicity; for example, WVAP19 is the group of people who identified
    # White as their *only* race, including people who identified as
    # Hispanic and White.
    vapnames = [
        "WHITEVAP",
        "BLACKVAP",
        "AMINVAP",
        "ASIANVAP",
        "NHPIVAP",
        "OTHVAP",
        "2MOREVAP",
        "NHWHITEVAP",
        "HVAP",
    ]
    vaptables = list(
        zip(
            [column + yearsuffix for column in vapnames],
            ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        )
    )
    groups.update(
        {
            column: _variables(f"B01001{table}", 7, 16)
            + _variables(f"B01001{table}", 22, 31)
            for column, table in vaptables
        }
    )

    # Get CVAP columns; the same goes for these columns as does the above,
    # except these columns are 18 years and older *and* citizens.
    cvapnames = [
        "WHITECVAP",
        "BLACKCVAP",
        "AMINCVAP",
        "ASIANCVAP",
        "NHPICVAP",
        "OTHCVAP",
        "2MORECVAP",
        "NHWHITECVAP",
        "HCVAP",
    ]
    cvaptables = list(
        zip(
            [name + yearsuffix for name in cvapnames],
            ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        )
    )
    groups.update(
        {
            column: _variables(f"B05003{table}", 9, 9)
            + _variables(f"B05003{table}", 11, 11)
            + _variables(f"B05003{table}", 20, 20)  # men
            + _variables(f"B05003{table}", 22, 22)  # women
            for column, table in cvaptables
        }
    )

    # Get all voting-age people and citizen voting-age people.
    groups["VAP" + yearsuffix] = _variables("B01001", 7, 25) + _variables(
        "B01001", 31, 49
    )
    groups["CVAP" + yearsuffix] = (
        _variables("B05003", 9, 9)
        + _variables("B05003", 11, 11)
        + _variables("B05003", 20, 20)
        + _variables("B05003", 22, 22)
    )

    # TODO: all variables used across the data submodule should be packaged up
    # as a class, so we can access individual dictionaries of variables to add.
    # For example, we should have a `Variables.acs5.vap` property which gives
    # us the voting-age population variables for the ACS 5-year estimates.

    # Get the list of all columns.
    allcols = (
        list(popcolumns.keys()) + [c for k in groups.values() for c in k] + columns
    )

    # Retrieve the data from the Census API.
    data = censusdata.download(
        "acs5",
        year,
        censusdata.censusgeo(
            [("state", str(state.fips).zfill(2)), ("county", "*"), (geometry, "*")]
        ),
        ["GEO_ID"] + allcols,
    )

    # Rework columns.
    data = data.reset_index(drop=True)
    data["GEO_ID"] = data["GEO_ID"].str.split("US").str[1]
    data = data.rename(
        {"GEO_ID": geometry.replace(" ", "").upper() + ("10" if year < 2020 else "20")},
        axis=1,
    )
    data = data.rename(popcolumns, axis=1)

    # Collapse column groups.
    for column, group in groups.items():
        data[column] = data[group].sum(axis=1)
        data = data.drop(group, axis=1)

    # Create a POCVAP column.
    data[f"POCVAP{yearsuffix}"] = (
        data[f"VAP{yearsuffix}"] - data[f"{white}{yearsuffix}"]
    )
    return data


def _variables(prefix, start, stop, suffix="E") -> list:
    """
    Returns the ACS variable names from the provided prefix, start, stop, and
    suffix parameters. Used to generate batches of names, especially for things
    like voting-age population. Variable names are formatted like
    `<prefix>_<number identifier><suffix>`, where `<prefix>` is a
    population grouping,
    `<number identifier>` is the number of the variable in that grouping, and
    `<suffix>` designates the file used. [Variables are listed
    here ](https://tinyurl.com/43ajptky>).

    Args:
        prefix (str): Population grouping; typically "B01001." These prefixes
            change based on subpopulation: for example, the prefix for Black
            age-by-sex tables is "B01001B"; for Hispanic and Latino, it is
            "B01001I."
        start (int): Where to start numbering.
        stop (int): Where to stop numbering. Inclusive.
        suffix (str): Suffix designating the file. For most purposes, this is "E."

    Returns:
        A list of ACS5 variable names.
    """
    return [f"{prefix}_{str(t).zfill(3)}{suffix}" for t in range(start, stop + 1)]


def _retrieve(year, geometry="tract"):
    """
    Downloads and extracts compressed CVAP data for the specified year.

    Args:
        year (int): Year for which we're grabbing CVAP data.
        geometry (str, optional): Geometry level for which we're grabbing CVAP
            data. Defaults to `"tract"`.

    Returns:
        In-memory text stream of decompressed CSV data.
    """
    # Create a mapping from geometry names to filenames.
    levels = {"block group": "BlockGr.csv", "tract": "Tract.csv"}
    inverted = {v: k for k, v in levels.items()}

    # Construct the URL.
    start, stop = year - 4, year
    root = "https://www2.census.gov/programs-surveys/decennial/rdo/datasets/"
    suffix = f"{stop}/{stop}-cvap/CVAP_{start}-{stop}_ACS_csv_files.zip"

    # Make the request and extract only the required files.
    with urlopen(root + suffix) as resource:
        with ZipFile(io.BytesIO(resource.read())) as archive:
            files = {
                inverted[file]: archive.read(file).decode(encoding="ISO-8859-1")
                for file in archive.namelist()
                if file == levels[geometry]
            }

    # Return the raw extracted file.
    return files[geometry]


def _raw(geometry, year) -> pd.DataFrame:
    """
    Reads raw CVAP data from the local repository.

    Args:
        geometry (str): Level of geometry for which we're getting 2019 CVAP
            data.
        year (int): Year for which data is retrieved.

    Returns:
        A DataFrame, where each block of 13 rows corresponds to an individual
        geometric unit (2010 Census Block Group, 2010 Census Tract) and
        each row in a given block corresponds to a CVAP statistic for that
        block's geometric unit.

    """
    # Retrieve the data at the specified geometry level and return
    # it as a dataframe.
    return pd.read_csv(io.StringIO(_retrieve(year, geometry)), encoding="ISO-8859-1")
