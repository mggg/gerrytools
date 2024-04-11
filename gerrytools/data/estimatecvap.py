import geopandas as gpd
import numpy as np
import pandas as pd
from pandas import DataFrame
import warnings

from ..geometry import unitmap
from .acs import acs5, cvap
from .census import census20


def estimatecvap2020(state) -> pd.DataFrame:
    """
    Estimates 2020 CVAP on 2020 blocks using 2020 PL94 data. **This method
    serves a different purpose than `gerrytools.data.estimatecvap.
    estimatecvap2010()`:** rather than using geometric procedures to
    put CVAP data on old geometries, this method takes advantage of the
    Census's geographic hierarchy, and associates finer-grained 2020 CVAP
    data with 2020 blocks. *No geometric data or procedures are used here*.
    The resulting data can then be adjoined to 2020 block geometries (or
    assigned to VTDs, assigned to districts, etc.) and be used to build
    other units of varying size.

    Args:
        state (State): The `us.State` for which CVAP will be estimated.

    Returns:
        A `DataFrame` of combined Census and ACS data at the Census
        block level.
    """

    # First, get the Census data for blocks and block groups.
    bg = census20(state, table="P4", geometry="block group")
    block = census20(state, table="P4", geometry="block")

    # Now, get 2020 Census data at the block group level and merging it
    # with the block group-level Census data.
    cvap20 = cvap(state, geometry="block group", year=2020)
    bg = bg.merge(cvap20, left_on="GEOID20", right_on="BLOCKGROUP20")

    # Name the VAP columns.
    vapcolumns = [
        "NHWHITEVAP20",
        "NHASIANVAP20",
        "NHBLACKVAP20",
        "NHNHPIVAP20",
        "NHAMINVAP20",
        "NHWHITEASIANVAP20",
        "NHWHITEAMINVAP20",
        "NHWHITEBLACKVAP20",
        "NHBLACKAMINVAP20",
    ]

    # Create "remainder" column.
    for universe in [block, bg]:
        universe["NHVAP20"] = universe["VAP20"] - universe["HVAP20"]
        universe["OTHVAP20"] = universe["NHVAP20"] - universe[vapcolumns].sum(axis=1)

    # Get the block group ID for blocks.
    block["BLOCKGROUP20"] = block["GEOID20"].astype(str).str[:-3]

    # Mapping from block group IDs to block group total populations.
    bgtotalvapmap = dict(zip(bg["BLOCKGROUP20"].astype(str), bg["VAP20"]))

    # Add all columns.
    allvapcols = vapcolumns + ["VAP20", "NHVAP20", "HVAP20", "NHOTHVAP20"]

    # Estimate CVAP data for all VAP columns.
    for vapcolumn in allvapcols:
        # Crete a mapping from block group names to totals for the VAP column.
        popmap = dict(zip(bg["GEOID20"].astype(str), bg[vapcolumn]))

        # Create column names.
        colpct = vapcolumn + "%"
        cvapcolumn = vapcolumn.replace("VAP", "CVAP")

        # Calculate ratios.
        block[colpct] = block["BLOCKGROUP20"].map(popmap)
        block[colpct] = block[vapcolumn] / block[colpct]

        # Create a mapping from block group IDs to CVAP groups.
        cvapcolumn = vapcolumn.replace("VAP", "CVAP")
        cvapmap = dict(zip(bg["BLOCKGROUP20"].astype(str), bg[cvapcolumn]))

        # Create two temporary columns: the first sets the block's CVAP to the
        # total CVAP for its block group; the second sets the block's VAP
        # to the total VAP for its block group. (Note: each of these C/VAP
        # columns are with respect to the current VAP column.)
        block["tmp"] = block["BLOCKGROUP20"].map(cvapmap)
        block["BGVAP20"] = block["BLOCKGROUP20"].map(bgtotalvapmap)

        # Next, compute the estimated CVAP by multiplying the VAP column
        # percent for the block group by the total CVAP population of the
        # block group.
        block[cvapcolumn] = block[colpct] * block["tmp"]

        # If the above doesn't work — which is the case if the VAP column
        # percent is NaN (0/0) or inf (k/0), we estimate the CVAP of the
        # block using the VAP ratio outright rather than the column-specific
        # VAP ratio.
        ni = block[block[colpct].isna()].index
        block.loc[ni, cvapcolumn] = (
            block.loc[ni, "VAP20"] / block.loc[ni, "BGVAP20"]
        ) * block.loc[ni, "tmp"]

        # Assert that our summed disaggregated numbers and totals are close!
        assert np.isclose(bg[cvapcolumn].sum() - block[cvapcolumn].sum(), 0)

    # Fill NaNs with 0 and drop unnecessary columns.
    block = block.fillna(0)
    block = block.drop(["tmp", "BGVAP20"], axis=1)

    # Return!
    return block


def fetchgeometries(state, geometry) -> gpd.GeoDataFrame:
    """
    Fetches the 2010 Census geometries on which ACS data are reported.

    Args:
        state (State): The `us.State` for which CVAP will be estimated.
        geometry10 (str): Level of geometry we're fetching. Accepted values are
            `"tract"` and `"block group"`.

    Returns:
        A `GeoDataFrame` of 2010 geometries.
    """
    # Get a Census locator for the provided State by replacing spaces in its
    # name with underscores (should they exist).
    clocator = state.name.replace(" ", "_")

    # Validate geometry level indicators.
    if geometry not in {"block group", "tract", "block"}:
        raise ValueError(f"Geometry level {geometry} not supported; aborting.")

    if geometry == "block group":
        geometry = "bg"
    if geometry == "block":
        geometry = "tabblock"

    # Construct the Census URL.
    fips = state.fips
    head = "https://www2.census.gov/geo/pvs/tiger2010st/"
    tail = f"{fips}_{clocator}/{fips}/tl_2010_{fips}_{geometry}10.zip"
    url = head + tail

    # Download, extract, and return the geometries from the URL.
    return gpd.read_file(url)


def mapbase(base, state, geometry, baseindex="GEOID20"):
    """
    Maps the provided geometries in `base` to the 2010 Census geometries
    specified by `geometry`.

    Args:
        base (GeoDataFrame): GeoDataFrame with the desired units for cvap to be
            estimated on.
        state (State): The `us.State` for which CVAP will be estimated.
        geometry (str): Level of geometry we're fetching. Accepted values are
            `"tract"` and `"block group"`.

    Returns:
       `base` with 2010 geometry assignments adjoined.
    """
    # Get the 2010 geometries from the Census.
    geometry10 = fetchgeometries(state, geometry)

    # Get the right name and rename the 2010 geometry index this way.
    geometry10id = "TRACT10" if geometry == "tract" else "BLOCKGROUP10"
    geometry10 = geometry10.rename({"GEOID10": geometry10id}, axis=1)

    # Create a unit mapping from the provided base units to those retrieved
    # from the Census. If the `base` passed has been sliced or could possibly
    # be a copy, *this will throw a SettingWithCopy warning*.
    mapping = unitmap((base, baseindex), (geometry10, geometry10id))
    base[geometry10id] = base[baseindex].map(mapping)

    return base


def estimatecvap2010(
    base, state, groups, ceiling, zfill, geometry10="tract", year=2019
) -> DataFrame:
    r"""
    Function for turning old (2019) CVAP data on 2010 geometries into
    estimates for current CVAP data on 2020 geometries. **This method
    serves a different purpose than `gerrytools.data.estimatecvap.
    estimatecvap2020()`:** this method is intended to put 2010-era CVAP
    data on 2020-era geometries, and uses geometric properties to do so.

    Users must supply a base `GeoDataFrame`
    representing their chosen U.S. state. Additionally, users must specify the
    demographic groups whose CVAP statistics are to be estimated. For each
    group, users specify a triple \((X, Y, Z)\) where \(X\) is the old CVAP
    column for that group, \(Y\) is the old VAP column for that group, and
    \(Z\) is the new VAP column for that group, which must be an existing
    column on `base`. Then, the estimated new CVAP for that group will be
    constructed by multiplying \((X / Y) \cdot Z\) for each new geometry.

    <div style="text-align: center;">
    </br>
    <img width="75%" src="../images/cvap-estimation.png"/>
    </div>

    Args:
        base (GeoDataFrame): A `GeoDataFrame` with the appropriate columns for
            estimating CVAP.
        state (State): The `us.State` object for which CVAP data is retrieved.
        groups (list): `(X, Y, Z)` triples for each desired CVAP group to be
            estimated, where each of the parameters are column names: `X` is
            the column on the 2010 geometries which contains the relevant CVAP
            data; `Y` is the column on the 2010 geometries which contains the
            relevant VAP data; `Z` is the column on the 2020 geometries to be
            weighted by the ratio of the per-unit ratios in `X` and `Y`. For
            example, if we wish to estimate Black CVAP, this triple would be
            `(NHBCVAP19, BVAP19, BVAP20)`, which takes the ratios of the
            `NHBCVAP19` and `BVAP19` columns on the 2010 geometries, and
            multplies the 2020 geometries' respective `BVAP20` values
            by these ratios.
        ceiling (float): Number representing where to cap the weighting
            ratio of CVAP to VAP20. After this percentage ceiling is
            passed, the percentage will be set to 1. We recommend
            setting this to 1.
        zfill (float): Fill in ratio for CVAP to VAP20 when there is 0 CVAP
            in the area. We recommend setting this parameter to `0.1`.
        geometry10 (str, optional): The 2010 geometry on which cvap will
            be pulled. Acceptable values are `"tract"` or `"block group"`.
            As tracts are less susceptible to change across Census vintages,
            setting this parameter to `"tract"` is recommended, as it is
            more likely that the 2020 Census blocks fit neatly into the
            2010 Census tracts.

    Returns:
       `base` geometries with 2019 CVAP-weighted 2020 CVAP estimates attached.
    """
    if geometry10 not in {"block group", "tract"}:
        print(f'Requested geometry "{geometry10}" is not allowed; ' "loading tracts.")
        geometry10 = "tract"

    # Grab ACS and CVAP special-tab data, and make sure our triples are correct
    cvap_geoid = "TRACT10" if geometry10 == "tract" else "BLOCKGROUP10"
    acs_source = acs5(state, geometry10, year=year)
    cvap_source = cvap(state, geometry10, year=year)

    # Validate the columns passed, issuing user warnings when it's inadvisable
    # to estimate CVAP given the passed columns.
    for cvap10, vap10, vap20 in groups:
        # If any of the CVAP columns passed correspond to columns which
        # tabulate people of multiple races, notify the user that there
        # isn't an appropriate 2019 VAP column to match them against.
        if any(substring in cvap10 for substring in {"AIW", "AW", "BW", "AIB"}):
            print(
                f"Warning: Estimating CVAP among {cvap10} is not advisable, "
                "since there isn't a reasonable VAP column from which to "
                "construct _CVAP / _VAP rates (because you seem to be "
                "combining two racial groups)."
            )

        # If the CVAP or ACS5 columns passed aren't present in the
        # set of possible columns, raise an error.
        if not (cvap10 in acs_source or cvap10 in cvap_source):
            possible_columns = set(acs_source).union(set(cvap_source))
            raise ValueError(
                f"Your CVAP column '{cvap10}' must be contained in either "
                f"the ACS or Special Tab columns: {possible_columns}"
            )

        if vap10 not in acs_source:
            raise ValueError(
                f"Your old VAP column '{vap10}' must be contained in the ACS "
                "columns: {set(acs_source)}"
            )

        # If the VAP20 column passed doesn't exist on the user-provided
        # geometries, raise an error.
        if vap20 not in base:
            raise ValueError(
                f"Your new VAP column '{vap20}' must be contained in your "
                f"base dataframe: {set(base)}"
            )

    # Remove ACS 5 columns that overlap with special-tab ones.
    non_overlaps = list(set(acs_source).difference(set(cvap_source)))
    acs_source = acs_source[[cvap_geoid] + non_overlaps]
    source = cvap_source.merge(acs_source, on=cvap_geoid)

    # Get the right columns from the base geometry, and map the base
    # geometries to the units with CVAP data. Apparently dropping bad
    # columns by using slicing incudes a SettingWithCopy warning, so
    # we're just dropping using .drop() instead.
    correct = ["geometry"] + [
        col for col in list(base) if any(sub in col for sub in ["POP", "VAP", "GEOID"])
    ]
    bads = list(set(base) - set(correct))

    # Warn the user of column removal:
    print("Removing the following columns: " + ", ".join(bads))

    pared = base.drop(bads, axis=1)
    pared = mapbase(pared, state, geometry10)

    # Compute weights.
    for cvap10, vap10, _ in groups:
        source[cvap10 + "%"] = source[cvap10] / source[vap10]

    # Fill in values according to the following rules:
    #
    #   1.  if there are 0 *CVAP reported and 0 *VAP reported, we set the
    #        weight to the average *CVAP/*VAP ratio within the county;
    #   2.  if there are 0 *CVAP reported and *VAP > 0, we set the weight to
    #        zero_fill;
    #   3.  if *CVAP > 0 but *VAP = 0 or *CVAP/*VAP > percentage_cap, we set
    #       the weight to 1.
    statewide = {
        cvap
        + "%": source[cvap].sum() / source[vap].sum() if source[vap].sum() != 0 else 0
        for (cvap, vap, _) in groups
    }

    # Rename colunms with percentages.
    cvappcts = [cvap + "%" for (cvap, _, _) in groups]

    # Get the county names and compute population-weighted CVAP averages. This
    # serves as a replacement for the statewide average.
    source["_county"] = source[cvap_geoid].str[2:5]
    counties = list(set(source["_county"]))
    countyaverages = {pct: {} for pct in cvappcts}

    for county in counties:
        chunk = source[source["_county"] == county]

        for cvap19, vap19, _ in groups:
            # Calculate the CVAP19-to-VAP19 ratio. Set numpy to ignore
            # runtimewarnings, but warn the user if one is encountered.
            # We do this so that the user doesn't get spooked by a numpy
            # warning, but we're still noisy about the weird value
            # encountered.
            cvap19total = chunk[cvap19].sum()
            vap19total = chunk[vap19].sum()

            np.seterr(divide="ignore", invalid="ignore")
            ratio = cvap19total / vap19total

            # Check whether the ratio of the above is less than the ceiling.
            if not np.isfinite(ratio):
                print(
                    county,
                    f"Encountered an invalid ratio: there are "
                    f"{cvap19total} {cvap19} persons and {vap19total} "
                    f"{vap19} persons, for a ratio of {cvap19total}/"
                    f"{vap19total}. @e have substituted it for the "
                    f"statewide {cvap19}-to-{vap19} share of "
                    f"{statewide[cvap19 + '%']}.",
                )
                ratio = statewide[cvap19 + "%"]

            # Set the county-average ratio.
            countyaverages[cvap19 + "%"][county] = ratio

    # Reset the numpy error catching thing.
    np.seterr(all="warn")

    # For each of the percentage columns, we want to apply the rules specified
    # by the user.
    for pct in cvappcts:
        # Fill NaNs with the *county-wide* average.
        countywidepcts = countyaverages[pct]

        source[pct] = source[pct].replace(np.inf, np.nan)
        nanindices = source[source[pct].isna()].index
        source.loc[nanindices, pct] = source.loc[nanindices, "_county"].map(
            countywidepcts
        )

        # Fill zeroes with the `zfill` value, and cap all the percentages.
        source[pct] = (
            source[pct].replace(0, zfill).apply(lambda c: 1 if c > ceiling else c)
        )

    # Assert we don't have any percentages over percentage_cap.
    assert all(np.all(source[p + "%"] <= ceiling) for (p, _, __) in groups)

    # Assert we don't have any zeros.
    assert all(np.all(source[p + "%"] > 0) for (p, _, __) in groups)

    # Set indices and create a mapping from IDs to weights.
    source = source.set_index(cvap_geoid)
    source = source[cvappcts]
    weights = source.to_dict(orient="index")

    # If the first character of the first key is "0", then we need to pad
    # the keys since ix gets interpreted as a float later.
    if [*weights.keys()][0][0] == "0":
        pad = "0"
    else:
        pad = ""

    # Group by the CVAP GEOID.
    groupedtogeometry = list(pared.groupby(cvap_geoid))

    # Get the year suffix so we can replace columns.
    yearsuffix = str(year)[2:]

    # For each of the geometry groups (e.g. a set of rows of blocks
    # corresponding to a single tract), and for each of the CVAP groups,
    # apply the appropriate weight to the blocks' 2020 VAP populations.
    for ix, group in groupedtogeometry:
        for cvap10, vap10, vap20 in groups:
            weight = cvap10 + "%"
            cvap20 = cvap10.replace(yearsuffix, "20_EST")
            group[weight] = weights[pad + str(int(ix))][weight]
            group[cvap20] = group[weight] * group[vap20]

    # Re-create a dataframe and strip out % columns, leaving only the estimate
    # columns.
    weightedbase = pd.concat(frame for _, frame in groupedtogeometry)
    weightedbase = weightedbase.drop(columns=[p + "%" for (p, _, _) in groups])
    return weightedbase
