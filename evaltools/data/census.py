
import pandas as pd
import requests
from itertools import combinations
from functools import reduce
import json


def _rjoin(df, columns):
    stringified = [df[c].astype(str) for c in columns]
    return reduce(lambda l, r: l + r, stringified[1:], stringified[0])

def census(state, table="P2", geometry="block"):
    """
    """
    # Check whether the geometry is right. If not, warn the user and set it
    # properly.
    if geometry not in {"block", "tract"}:
        print(f"Geometry \"{geometry}\" not accepted; defaulting to \"block\".")
        geometry = "block"
    
    # Keeping this key here in plaintext is fine, I don't want others to have to
    # configure their own keys. There aren't any API limits (I don't think?) and
    # it's not consequential for me to leave this here.
    key = "75c0c07e6f0ab7b0a9a1c14c3d8af9d9f13b3d65"

    # Set the base Census API URL and get the keys for the provided table.
    base = f"https://api.census.gov/data/2020/dec/pl"
    varmap = variables(table)
    vars = list(varmap.keys())

    # Create the end part of the query string.
    q = [
        ("key", key),
        ("for", f"{geometry}:*"),
        ("in", f"state:{str(state.fips).zfill(2)}"),
        ("in", "county:*"),
    ]

    # Based on the geometry type, add an additional entry; this is required to
    # match the Census geographic hierarchy.
    if geometry == "block": q.append(("in", "tract:*"))

    # Now, since the Census doesn't allow us to request more than 50 variables
    # at once, we request things in two parts and then merge them together.
    mergeable = []

    for start, stop in [(0, 50), (50, len(vars))]:
        # Get the chunk of variables and create a tail of columns (geographic
        # identifiers).
        varchunk = vars[start:stop]
        tail = ["state", "county", "tract"] + (["block"] if geometry == "block" else [])

        # Create an unescaped query string.
        unescaped = q.copy()
        unescaped.append(("get", ",".join(varchunk)))

        # Create an escaped query string from the previous.
        escaped = "?" + "&".join(
            f"{param}={value}" for param, value in unescaped
        )

        # Send the request and create a dataframe.
        req = requests.get(base + escaped).json()
        header, data = req[0], req[1:]
        chunk = pd.DataFrame(data, columns=header)

        # Get a GEOID column and drop old columns.
        chunk["GEOID20"] = _rjoin(chunk[tail], tail)
        chunk = chunk.drop(tail, axis=1)
        mergeable.append(chunk)

    # Merge the dataframes and rename.
    merged = reduce(lambda l, r: pd.merge(l, r, on="GEOID20"), mergeable)
    merged = merged.rename(varmap, axis=1)


def variables(table="P2"):
    """
    Produces variable names for the 2020 Census PL94-171 tables.

    Args:
        table (string, optional)
    """
    # List the categories of Census variables and find the combinations in the
    # correct order. This *should* be the original order in which they're listed,
    # but these have been spot-checked to verify their correctness.
    categories = ["WHITE", "BLACK", "AMIN", "ASIAN", "NHPI", "OTH"]
    combos = list(pd.core.common.flatten(
        [
            "".join(list(combo)) + "20"
            for i in range(1, len(categories)+1)
            for combo in list(combinations(categories, i))
        ]
    ))

    # Now, for each of the combinations, we map the appropriate variable name to
    # the descriptor. Each of these tranches should have a width of 6 choose i,
    # where i is the number of categories in the combination. For example, the
    # second tranch (from 13 to 28) has width 15, as 6C2=15.
    numbers = list(range(5, 11)) + list(range(13, 28)) + list(range(29, 49)) \
        + list(range(50, 65)) + list(range(66, 72)) + [73]

    # Create the variable names and zip the names together with the combinations.
    names = [f"{table}_{str(n).zfill(3)}N" for n in numbers]
    return dict(zip(names, combos))

