
import pandas as pd
import censusdata
from pathlib import Path


def cvap(state, geometry="tract") -> pd.DataFrame:
    """
    Retrieves and CSV-formats 2019 5-year CVAP data for the provided state at
    the specified geometry level. Geometries from the **2010 Census**. Variables
    and descriptions are listed `here <https://tinyurl.com/3mnrm56s>`_.

    Args:
        state (us.State): The `State` object for which we're retrieving 2019 ACS
            CVAP Special Tab.
        geometry (str, optional): Level of geometry for which we're getting data.
            Accepted values are `"block group"` for 2010 Census Block Groups, and
            `"tract"` for 2010 Census Tracts. Defaults to `"tract"`.

    Returns
        A `DataFrame` with a `GEOID` column and corresponding CVAP columns from
        the 2019 ACS CVAP Special Tab.
    """
    # Maps line numbers to descriptors.
    descriptions = {
        1: "CVAP",
        2: "NHCVAP",
        3: "NHAICVAP",
        4: "NHACVAP",
        5: "NHBCVAP",
        6: "NHNHPICVAP",
        7: "NHWCVAP",
        8: "NHAIWCVAP",
        9: "NHAWCVAP",
        10: "NHBWCVAP",
        11: "NHAIBCVAP",
        12: "NHOCVAP",
        13: "HCVAP" 
    }

    # First, load the raw data requested; allowed geometry values are "bg10" and
    # "tract10."
    if geometry not in {"block group", "tract"}:
        print(f"Requested geometry \"{geometry}\" is not allowed; loading tracts.")
        geometry = "tract10"

    # Load the raw data.
    raw = _raw(geometry)

    # Create a STATE column for filtering and remove all rows which don't match
    # the state FIPS code.
    raw["GEOID"] = raw["geoid"].str.split("US").str[1]
    raw["STATE"] = raw["GEOID"].str[:2]
    instate = raw[raw["STATE"] == str(state.fips)]

    # Now that we have the in-state data, we aim to pivot the table. Because the
    # ACS data is in a line-numbered format (i.e. each chunk of 13 lines matches
    # to an individual geometry, and each of the 13 lines describes an individual
    # statistic) we need to first collapse each chunk of 13 lines, then build a
    # dataframe from the resulting collapsed lines. First we send the dataframe
    # to a list of records.
    instate_records = instate.to_dict(orient="records")
    collapsed = []

    # Next, we collapse these records to a single record.
    for i in range(0, len(instate_records), 13):
        # Create an empty records.
        record = {}

        # For each of the records in the block, "collapse" them into a single
        # record.
        block = instate_records[i:i+13]
        for line in block:
            record[geometry.replace(" ", "").upper() + "10"] = line["GEOID"]
            record[descriptions[line["lnnumber"]] + "19"] = line["cvap_est"]

        collapsed.append(record)

    # Create a dataframe and a POCCVAP column; all people minus non-Hispanic
    # White.
    data = pd.DataFrame().from_records(collapsed)
    data["POCCVAP19"] = data["CVAP19"] - data["NHWCVAP19"]

    return data

def acs5(state, geometry="tract", year=2019, columns=[]) -> pd.DataFrame:
    """
    Retrieves ACS 5-year population estimates for the provided state, geometry
    level, and year. Geometries are from the **2010 Census**.
    
    Args:
        state (us.State): `State` object for the desired state.
        geometry (str, optional): Geometry level at which data is retrieved.
            Acceptable values are `"tract"` and `"block group"`. Defaults to
            `"tract"`, so data is retrieved at the 2010 Census tract level.
        year (int, optional): Year for which data is retrieved. Defaults to 2019.
        columns (list, optional): Columns to retrieve. If `None`, a default set
            of columns including total populations by race and ethnicity and voting-age
            populations by race and ethnicity are returned, along with a GEOID
            column.

    Returns:
        A DataFrame containing the formatted data.
    """
    # Columns for total populations.
    popcolumns = {
        "B01001_001E": "TOTPOP19",
        "B03002_003E": "WHITE19",
        "B03002_004E": "BLACK19",
        "B03002_005E": "AMIN19",
        "B03002_006E": "ASIAN19",
        "B03002_007E": "NHPI19",
        "B03002_008E": "OTH19",
        "B03002_009E": "2MORE19",
        "B03002_002E": "NHISP19",
    }

    # Column *groups* for tables. This is a little messy.
    columns_tables = list(zip(
        [
            "WVAP19", "BVAP19", "AMINVAP19", "ASIANVAP19", "NHPIVAP19", "OTHVAP19",
            "2MOREVAP19", "NHWVAP19", "HVAP19"
        ],
        ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    ))

    groups = {
        column: variables(f"B01001{table}", 7, 16, suffix="E") + variables(f"B01001{table}", 22, 31, suffix="E")
        for column, table in columns_tables
    }
    groups["VAP19"] = variables("B01001", 7, 25, suffix="E") + variables("B01001", 31, 49, suffix="E")

    # Get the list of all columns.
    allcols = list(popcolumns.keys()) + [c for k in groups.values() for c in k] + columns

    # Retrieve the data from the Census API.
    data = censusdata.download(
        "acs5", year,
        censusdata.censusgeo(
            [("state", str(state.fips).zfill(2)), ("county", "*"), (geometry, "*")]
        ),
        ["GEO_ID"] + allcols
    )

    # Rework columns.
    data = data.reset_index(drop=True)
    data["GEO_ID"] = data["GEO_ID"].str.split("US").str[1]
    data = data.rename({"GEO_ID": geometry.replace(" ", "").upper() + "10"}, axis=1)
    data = data.rename(popcolumns, axis=1)

    # Collapse column groups.
    for column, group in groups.items():
        data[column] = data[group].sum(axis=1)
        data = data.drop(group, axis=1)

    # Create a POCVAP column.
    data["POCVAP19"] = data["VAP19"] - data["NHWVAP19"]

    return data

def variables(prefix, start, stop, suffix="E") -> list:
    """
    Returns the ACS variable names from the provided prefix, start, stop, and
    suffix parameters. Used to generate batches of names, especially for things
    like voting-age population. Variable names are formatted like
    `<prefix>_<number identifier><suffix>`, where `<prefix>` is a population grouping,
    `<number identifier>` is the number of the variable in that grouping, and
    `<suffix>` designates the file used. Variables are listed
    `here <https://tinyurl.com/43ajptky>`_.

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
    return [
        f"{prefix}_{str(t).zfill(3)}{suffix}"
        for t in range(start, stop+1)
    ]

def _raw(geometry) -> pd.DataFrame:
    """
    Reads raw CVAP data from the local repository.

    Args:
        geometry (str): Level of geometry for which we're getting 2019 CVAP data.

    Returns:
        A DataFrame, where each block of 13 rows corresponds to an individual
        geometric unit (2010 Census Block Group, 2010 Census Tract) and each row
        in a given block corresponds to a CVAP statistic for that block's
        geometric unit.
    """
    # Get the filepath local to the repository, load in the raw data, and return
    # it to the caller.
    local = Path(__file__).parent.absolute()
    return pd.read_csv(local/f"local/{geometry}.zip", encoding="ISO-8859-1")
