from functools import reduce
from itertools import combinations
from typing import Iterable

import censusdata
import pandas as pd
import requests


def _rjoin(df, columns) -> Iterable:
    """
    Private method for elementwise concatenating string dataframe columns.

    Args:
        df (pd.DataFrame): DataFrame to which the columns belong.
        columns (list): List of column names to be concatenated in
            left-to-right order.

    Returns:
        An iterable representing the column of concatenated column entries.
    """
    stringified = [df[c].astype(str) for c in columns]
    return reduce(lambda left, right: left + right, stringified[1:], stringified[0])


def census10(state, table="P8", columns={}, geometry="block"):
    """
    Retrieves `geometry`-level 2010 Summary File 1 data via the Census API.

    Args:
        state (State): `us.State` object (e.g. `us.states.WI`).
        table (string, optional): Table from which we retrieve data.
           Defaults to the P8 table, which contains population by race
           regardless of ethnicity.
        columns (dict, optional): Dictionary which maps Census column names
            (from the correct table) to human-readable names. We require this
            to be a dictionary, _not_ a list, as specifying human-readable
            names will implicitly protect against incorrect column names
            and excessive API calls.
        geometry (string, optional): Geometry level at which we retrieve data.
            Defaults to `"block"` to retrieve block-level data for the state
            provided. Accepted values are `"block"`, `"block group`", and
            `"tract"`.

    Returns:
        A DataFrame with columns renamed according to their Census description
        designation and a unique identifier column for joining to geometries.
    """
    # Check whether the geometry is right. If not, warn the user and set it
    # properly.
    if geometry not in {"block", "tract", "block group"}:
        raise ValueError(f'Geometry "{geometry}" not accepted.')

    # Check whether we're providing an appropriate table name.
    if table not in {"P8", "P9", "P10", "P11"}:
        raise ValueError(f'Unknown table "{table}".')

    # Create the right geometry identifiers.
    geometries = [("state", str(state.fips)), ("county", "*"), ("tract", "*")]
    if geometry in {"block group", "block"}:
        geometries += [(geometry, "*")]

    # Create an identifier column.
    identifier = geometry.replace(" ", "").upper() + "10"

    varmap = columns if columns else variables(table)
    vars = list(varmap.keys())
    # Download data.
    raw = censusdata.download(
        "sf1",
        2010,
        censusdata.censusgeo(geometries),
        ["GEO_ID"] + vars,
    )

    # Rename columns and send back to the caller!
    raw = raw.rename({"GEO_ID": identifier, **columns}, axis=1)
    raw[identifier] = raw[identifier].str[9:]
    clean = raw.reset_index(drop=True)

    clean = clean.rename(varmap, axis=1)
    return clean


def census20(
    state,
    table="P1",
    columns={},
    geometry="block",
    key="75c0c07e6f0ab7b0a9a1c14c3d8af9d9f13b3d65",
) -> pd.DataFrame:
    """
    Retrieves `geometry`-level 2020 Decennial Census PL94-171 data via the
    Census API.

    Args:
        state (State): `us.State` object (e.g. `us.states.WI`).
        table (string, optional): Table from which we retrieve data.
            Defaults to the P1 table, which gets populations by race
            regardless of ethnicity.
        columns (dict, optional): Dictionary which maps Census column names
            (from the correct table) to human-readable names. We require this
            to be a dictionary, _not_ a list, as specifying human-readable
            names will implicitly protect against incorrect column names and
            excessive API calls.
        geometry (string, optional): Geometry level at which we retrieve data.
            Defaults to `"block"` to retrieve block-level data for the state
            provided. Accepted values are `"block"`, `"block group`",
            and `"tract"`.
        key (string, optional): Census API key.

    Returns:
        A DataFrame with columns renamed according to their Census description
        designation and a `GEOID20` column for joining to geometries.
    """
    # Check whether the geometry is right. If not, warn the user and set it
    # properly.
    if geometry not in {"block", "tract", "block group"}:
        print(f'Geometry "{geometry}" not accepted; defaulting' 'to "block".')
        geometry = "block"

    # Check whether we're providing an appropriate table name.
    if table not in {"P1", "P2", "P3", "P4"}:
        print(f'Table "{table}" not accepted; defaulting to "P1."')
        table = "P1"

    # Set the base Census API URL and get the keys for the provided table.
    base = "https://api.census.gov/data/2020/dec/pl"
    varmap = columns if columns else variables(table)
    vars = list(varmap.keys())

    # Create the end part of the query string.
    q = [
        ("key", key),
        ("for", f"{geometry.replace(' ', r'%20')}:*"),
        ("in", f"state:{str(state.fips).zfill(2)}"),
        ("in", "county:*"),
    ]

    # Based on the geometry type, add an additional entry; this is required to
    # match the Census geographic hierarchy.
    if geometry in {"block", "block group"}:
        q.append(("in", "tract:*"))

    # Now, since the Census doesn't allow us to request more than 50 variables
    # at once, we request things in two parts and then merge them together.
    mergeable = []

    # Split up start and stop positions based on the number of variables.
    if len(vars) < 45:
        positions = [(0, len(vars))]
    else:
        positions = [(0, 45), (45, len(vars))]

    for start, stop in positions:
        # Get the chunk of variables and create a tail of columns (geographic
        # identifiers).
        varchunk = vars[start:stop]
        last = [geometry] if geometry in {"block group", "block"} else []
        tail = ["state", "county", "tract"] + last

        # Create an unescaped query string.
        unescaped = q.copy()
        unescaped.append(("get", ",".join(varchunk)))

        # Create an escaped query string from the previous.
        escaped = "?" + "&".join(f"{param}={value}" for param, value in unescaped)

        # Send the request and create a dataframe.
        req = requests.get(base + escaped).json()
        header, data = req[0], req[1:]
        chunk = pd.DataFrame(data, columns=header)

        # Get a GEOID column and drop old columns.
        chunk["GEOID20"] = _rjoin(chunk[tail], tail)
        chunk = chunk.drop(tail, axis=1)
        mergeable.append(chunk)

    # Merge the dataframes, rename everything, make the columns
    # ints, and return.
    merged = reduce(lambda left, right: pd.merge(left, right, on="GEOID20"), mergeable)
    merged = merged.rename(varmap, axis=1)
    merged = merged.astype({var: int for var in varmap.values()})

    # Make the GEOID20 column the first column.
    merged = merged[["GEOID20"] + list(varmap.values())]

    return merged


def variables(table) -> dict:
    """
    Produces variable names for the 2020 Census PL94-171 tables. Variables are
    determined from patterns apparent in PL94 variable
    [lists for tables P1 through P4](https://tinyurl.com/2s3btptn).

    Args:
        table (string): The table for which we're generating variables.

    Returns:
        A dictionary mapping Census variable codes to human-readable ones.
    """
    # List the categories of Census variables and find the combinations in the
    # correct order. This *should* be the original order in which they're
    # listed, but these have been spot-checked to verify their correctness.
    # These names are also modified based on the table passed; for example,
    # if the table passed is P2 or P4, we prepend an "NH" to the beginning,
    # as these columns are explicitly non-hispanic people. If the table
    # passed is P3 or P4, we append a "VAP" to the end to signify these
    # are people of voting age; otherwise, we add "POP."
    categories = ["WHITE", "BLACK", "AMIN", "ASIAN", "NHPI", "OTH"]
    prefix = "NH" if table in {"P2", "P4", "P9", "P11"} else ""
    suffix = "VAP" if table in {"P3", "P4", "P10", "P11"} else "POP"
    year_suff = "20" if table in {"P1", "P2", "P3", "P4"} else "10"
    combos = list(
        pd.core.common.flatten(
            [
                prefix + "".join(list(combo)) + suffix + year_suff
                for i in range(1, len(categories) + 1)
                for combo in list(combinations(categories, i))
            ]
        )
    )

    # Now, for each of the combinations, we map the appropriate variable
    # name to the descriptor. Each of these tranches should have a width
    # of 6 choose i, where i is the number of categories in the
    # combination. For example, the second tranch (from 13 to 27) has
    # width 15, as 6C2=15.
    if table in {"P1", "P3", "P8", "P10"}:
        tranches = [(3, 8), (11, 25), (27, 46), (48, 62), (64, 69), (71, 71)]
    else:
        tranches = [(5, 10), (13, 27), (29, 48), (50, 64), (66, 71), (73, 73)]

    # Create variable numbers.
    numbers = list(pd.core.common.flatten([list(range(i, j + 1)) for i, j in tranches]))

    # Edit these for specific tables. For example, in tables P2 and P3, we want
    # to get the total Hispanic population and the total population.
    if table in {"P2", "P4", "P9", "P11"}:
        numbers = [1, 2] + numbers
        if table in {"P4", "P11"}:
            hcol = f"HVAP{year_suff}"
            tcol = f"VAP{year_suff}"
        else:
            hcol = f"HPOP{year_suff}"
            tcol = f"TOTPOP{year_suff}"
        combos = [tcol, hcol] + combos
    else:
        numbers = [1] + numbers
        if table in {"P3", "P4", "P10", "P11"}:
            tcol = f"VAP{year_suff}"
        else:
            tcol = f"TOTPOP{year_suff}"
        combos = [tcol] + combos

    # Create the variable names and zip the names together
    # with the combinations.
    if year_suff == "20":
        names = [f"{table}_{str(n).zfill(3)}N" for n in numbers]
    else:
        names = [
            f"P{str(table.split('P')[-1]).zfill(3)}{str(n).zfill(3)}" for n in numbers
        ]
    return dict(zip(names, combos))
