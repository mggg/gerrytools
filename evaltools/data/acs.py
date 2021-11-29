
import pandas as pd
from pathlib import Path


# Maps line numbers to descriptors.
descriptions = {
    1: "CVAP",
    2: "NHCVAP",
    3: "AICVAP",
    4: "ACVAP",
    5: "BCVAP",
    6: "NHPICVAP",
    7: "WCVAP",
    8: "AIWCVAP",
    9: "AWCVAP",
    10: "BWCVAP",
    11: "AIBCVAP",
    12: "OCVAP",
    13: "HCVAP" 
}


def cvap(state, geometry="bg10") -> pd.DataFrame:
    """
    Retrieves and CSV-formats 2019 CVAP data for the provided state at the specified
    geometry level.

    Args:
        state (us.State): The `State` object for which we're retrieving 2019 ACS
            CVAP Special Tab.
        geometry (str, optional): Level of geometry for which we're getting data.
            Accepted values are `bg10` for 2010 Census Block Groups, and `tract10`
            for 2010 Census Tracts.

    Returns
        A `DataFrame` with a `GEOID` column and corresponding CVAP columns from
        the 2019 ACS CVAP Special Tab.
    """
    # First, load the raw data requested; allowed geometry values are "bg10" and
    # "tract10."
    if geometry not in {"bg10", "tract10"}:
        print(f"Requested geometry \"{geometry}\" is not allowed; loading block groups.")
        geometry = "bg10"

    # Load the raw data.
    _raw = raw(geometry)

    # Create a STATE column for filtering and remove all rows which don't match
    # the state FIPS code.
    _raw["GEOID"] = _raw["geoid"].str.split("US").str[1]
    _raw["STATE"] = _raw["GEOID"].str[:2]
    instate = _raw[_raw["STATE"] == str(state.fips)]

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
            record["GEOID"] = line["GEOID"]
            record[descriptions[line["lnnumber"]] + "19"] = line["cvap_est"]

        collapsed.append(record)

    # Create a dataframe out of the listed records and return.
    return pd.DataFrame().from_records(collapsed)
    

def raw(geometry) -> pd.DataFrame:
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
