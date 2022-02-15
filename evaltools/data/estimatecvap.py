from cv2 import groupRectangles
import pandas as pd
import geopandas as gpd
import numpy as np
from pandas import DataFrame
import us
import os
import glob
import requests
import zipfile
from io import BytesIO
from evaltools.data import acs
from evaltools.geometry import unitmap

def fetchgeom(geometry10, state): 
    """
    Fetches the 2010 geometry that CVAP was pulled on. 
  
    Arguments:
        geometry10 (string): The 2010 geometry that CVAP was pulled on. Will either be "tract" or "bg".
        state (us.states): The US state for which cvap will be estimated. 
    Returns: 
        geom10 (GeoDataFrame): A GeoDataFrame representing the 2010 geometry that was pulled.
    """
    state_string = state.name.replace(" ", "_")
    
    geometry10_url = f"https://www2.census.gov/geo/pvs/tiger2010st/" \
        f"{state.fips}_{state_string}/{state.fips}/tl_2010_{state.fips}_{geometry10}10.zip"

    geometry10_zip = requests.get(geometry10_url)
    zip_name = geometry10_url.split("/")[-1].split(".zip")[0]
    zip = zipfile.ZipFile(BytesIO(geometry10_zip.content))
    zip.extractall(zip_name)
    
    geom10 = gpd.read_file(f"{zip_name}/{zip_name}.shp")
    return geom10

def mapbase(base, geometry10, state):
    """
    Wrapper for unit map to map base to the 2010 geometry cvap was pulled on. 
    
    Arguments:
        base (GeoDataFrame): GeoDataFrame with the desired units for cvap to be estimated on. 
        geometry10 (string): The 2010 geometry on which cvap will be pulled. Will be "tract" or "block group".
        state (us.state): The `State` object for which CVAP data is being retrieved. 
 
    Returns: 
       base (GeoDataFrame): Base with new column added for the mapping to the 2010 geometry.
    """
    cvap_geoid = "TRACT10" if geometry10 == "tract" else "BLOCKGROUP10"
    
    state_string = state.name.replace(" ", "_")
    geometry = fetchgeom(geometry10, state)
    mapping = unitmap((base, "GEOID20"), (geometry, "GEOID10"))
    map_df = pd.DataFrame.from_dict(mapping, orient="index", columns=[cvap_geoid])\
        .reset_index() \
        .rename(columns={"index":"GEOID20"})
    base = base.merge(map_df, on = "GEOID20", how="left")

    for file in glob.glob(f"tl_2010_{state.fips}_{geometry10}10/*"): os.remove(file)
    os.rmdir(f"tl_2010_{state.fips}_{geometry10}10")

    return base

def estimatecvap(base, state, cvap_groups, percentage_cap, zero_fill, geometry10="tract") -> DataFrame:
    """
    Function for turning old (2019) CVAP data on 2010 geometries into estimates for current CVAP data
    on 2020 geometries. We must supply a base GeoDataFrame representing their chosen U.S. state.
    Additionally, we'll need to specify the demographic groups whose CVAP numbers we wants
    to estimate. For each group, we want to specify a triple (X, Y, Z) where X is the old CVAP column
    for that group, Y is the old VAP column for that group, and Z is the new VAP column for that group
    (Z must be a column in our base geodataframe). Then, the estimated new CVAP for that group will 
    be constructed by multiplying (X / Y) * Z for each new geometry.

    Arguments:
       base (GeoDataFrame): A dataframe containing population values for a state on the desired geographical units.
       state (us.state): The `State` object for which CVAP data is being retrieved. 
       cvap_groups (list): (X, Y, Z) triples for each desired CVAP group to be estimated.
       percentage_cap (float): Number representing where to cap the weighting ratio of CVAP to VAP20. After 
                               this percentage barrier is passed, the percentage will be set to 1. 
                               Suggested choice: 1.
       zero_fill (float): Fill in ratio for CVAP to VAP20 when there is 0 CVAP in the area. 
                          Suggested choice: 0.1
       geometry10 (string): The 2010 geometry on which cvap will be pulled. Will be "tract" or "block group".

    Returns: 
       weightedbase (DataFrame): A dataframe containing the newly estimated 2020 CVAP numbers on the same geographical units as base.  
    """
    if geometry10 not in {"block group", "tract"}:
        print(f"Requested geometry \"{geometry10}\" is not allowed; loading tracts.")
        geometry10="tract"

    # Grab ACS and CVAP special-tab data, and make sure our triples are correct
    cvap_geoid = "TRACT10" if geometry10 == "tract" else "BLOCKGROUP10"
    acs_source = acs.acs5(state, geometry10)
    cvap_source = acs.cvap(state, geometry10)
    for (cvap, vap, new_vap) in cvap_groups:
        if not (cvap in acs_source or cvap in cvap_source):
            possible_columns = set(acs_source).union(set(cvap_source))
            raise ValueError(f"Your CVAP column '{cvap}' must be contained in either the ACS or Special Tab columns: {possible_columns}")
        if not vap in acs_source:
            raise ValueError(f"Your old VAP column '{vap}' must be contained in the ACS columns: {set(acs_source)}")
        if not new_vap in base:
            raise ValueError(f"Your new VAP column '{new_vap}' must be contained in your base dataframe: {set(base)}")
    
    # Remove ACS 5 columns that overlap with special-tab ones
    non_overlaps = list(set(acs_source).difference(set(cvap_source)))
    acs_source = acs_source[[cvap_geoid] + non_overlaps]
    source = cvap_source.merge(acs_source, on=cvap_geoid)

    geomname = "tract" if geometry10 == "tract" else "block group"
    base = base[[col for col in list(base) if "POP" in col or "VAP" in col or "geometry" in col or "GEOID" in col]]
    base = mapbase(base, geomname, state)

    # Compute weights.
    for (cvap, vap, _) in cvap_groups: 
       source[cvap + "%"] = source[cvap]/source[vap]

    # Fill in values according to the following rules:
    # 
    #   1.  if there are 0 *CVAP reported and 0 *VAP reported, we set the weight to
    #       the average *CVAP/*VAP ratio within the county;
    #   2.  if there are 0 *CVAP reported and *VAP > 0, we set the weight to zero_fill;
    #   3.  if *CVAP > 0 but *VAP = 0 or *CVAP/*VAP > percentage_cap, we set the weight to 1.
    statewide = {
        cvap + "%": source[cvap].sum()/source[vap].sum() if source[vap].sum() != 0 else 0
        for (cvap, vap, _) in cvap_groups
    }

    cvappcts = [cvap + "%" for (cvap, _, _) in cvap_groups]

    for pct in cvappcts:        
        for ix, row in source.iterrows():
            if pd.isna(row[pct]):
                county = row[cvap_geoid][2:5]
                county_avg = np.mean(source[pct][source[cvap_geoid].str[2:5] == county])
                source.at[ix, pct] = county_avg if not pd.isna(county_avg) else statewide[pct]
        
        source[pct] = source[pct] \
            .replace(0, zero_fill)  \
            .apply(lambda c: 1 if c > percentage_cap else c)

    # Assert we don't have any percentages over percentage_cap.
    assert all(
        np.all(source[p + "%"] <= percentage_cap)
        for (p, _, _) in cvap_groups
    )

    # Set indices and create a mapping from IDs to weights.
    source = source.set_index(cvap_geoid)    
    source = source[[p + "%" for (p, _, _)in cvap_groups]]
    weights = source.to_dict(orient="index")

    # Weight base!
    groups = list(base.groupby(cvap_geoid))
  
    for ix, group in groups:
        for (cvap, vap, new_vap) in cvap_groups:
            weight = cvap + "%"
            cvap_est = cvap.replace("19", "20_EST")
            group[weight] = weights[ix][weight]
            group[cvap_est] = group[weight] * group[new_vap]

    # Re-create a dataframe and strip out % columns
    weightedbase = pd.concat(frame for _, frame in groups)
    weightedbase = weightedbase.drop(columns=[p + "%" for (p, _, _) in cvap_groups])

    return weightedbase
