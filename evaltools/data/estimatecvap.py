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
from acs import acs5, cvap
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

def estimatecvap(base, state, percentage_cap, zero_fill, geometry10="tract") -> DataFrame:
    """
    Arguments:
       base (GeoDataFrame): A dataframe containing population values for a state on the desired geographical units.
                         Must contain a column for cvap_source units. 
       
       state (us.state): The `State` object for which CVAP data is being retrieved. 
       percentage_cap (float): Number representing where to cap the weighting ratio of CVAP to VAP20. After 
                               this percentage barrier is passed, the percentage will be set to 1. 
       zero_fill (float): Fill in ratio for CVAP to VAP20 when there is 0 CVAP in the area. 
       geometry10 (string): The 2010 geometry on which cvap will be pulled. Will be "tract" or "block group".

    Returns: 
       weightedbase (DataFrame): A dataframe containing the newly estimated 2020 CVAP numbers on the same geographical units as base.  
    """
    if geometry10 not in {"block group", "tract"}:
        print(f"Requested geometry \"{geometry10}\" is not allowed; loading tracts.")
        geometry10="tract"

    cvap_geoid = "TRACT10" if geometry10 == "tract" else "BLOCKGROUP10"
    geomname = "tract" if geometry10 == "tract" else "block group"

    base = base[[col for col in list(base) if "POP" in col or "VAP" in col or "geometry" in col or "GEOID" in col]]
    cvap_source = acs5(state, geometry10)
    base = mapbase(base, geomname, state)

    # Get VAP and CVAP columns.
    vaps = [c for c in list(base) if "VAP20" in c]
    cvaps = [c for c in list(cvap_source) if "CVAP19" in c]


    # Compute weights.
    names = list(zip(
        ["CVAP19", "BCVAP19", "HCVAP19", "ASIANCVAP19", "WCVAP19"],
        ["VAP19", "BVAP19", "HVAP19", "ASIANVAP19", "WVAP19"]
    ))
    for cvap, vap in names: 
       cvap_source[cvap + "%"] = cvap_source[cvap]/cvap_source[vap]

    # Fill in values according to the following rules:
    # 
    #   1.  if there are 0 *CVAP reported and 0 *VAP reported, we set the weight to
    #       the average *CVAP/*VAP ratio within the county;
    #   2.  if there are 0 *CVAP reported and *VAP > 0, we set the weight to zero_fill;
    #   3.  if *CVAP > 0 but *VAP = 0 or *CVAP/*VAP > percentage_cap, we set the weight to 1.
    statewide = {
        cvap + "%": cvap_source[cvap].sum()/cvap_source[vap].sum() if cvap_source[vap].sum() != 0 else 0
        for cvap, vap in names
    }

    cvappcts = [cvap + "%" for cvap, _ in names]

    for pct in cvappcts:        
        for ix, row in cvap_source.iterrows():
            if pd.isna(row[pct]):
                county = row[cvap_geoid][2:5]
                county_avg = np.mean(cvap_source[pct][cvap_source[cvap_geoid].str[2:5] == county])
                cvap_source.at[ix, pct] = county_avg if not pd.isna(county_avg) else statewide[pct]
        
        cvap_source[pct] = cvap_source[pct] \
            .replace(0, zero_fill)  \
            .apply(lambda c: 1 if c > percentage_cap else c)

    # Assert we don't have any percentages over percentage_cap.
    assert all(
        np.all(cvap_source[p + "%"] <= percentage_cap)
        for p, _ in names
    )
    # Set indices and create a mapping from IDs to weights.

    cvap_source = cvap_source.set_index(cvap_geoid)    
    cvap_source = cvap_source[[p + "%" for p, _ in names]]
    weights = cvap_source.to_dict(orient="index")

    # Weight base!
    basepairs = list(zip(
        ["CVAP20_EST", "BCVAP20_EST", "HCVAP20_EST", "ACVAP20_EST", "WCVAP20_EST"],
        ["CVAP19%", "BCVAP19%", "HCVAP19%", "ASIANCVAP19%", "WCVAP19%"],
        ["VAP20", "APBVAP20", "HVAP20", "ASIANVAP20", "WVAP20"]
    ))

    groups = list(base.groupby(cvap_geoid))
  
    for ix, group in groups:
        for name, weight, source in basepairs:
            group[weight] = weights[ix][weight]
            group[name] = group[weight] * group[source]

    # Re-create a dataframe.
    weightedbase = pd.concat(frame for _, frame in groups)

    return weightedbase
my_blocks = gpd.read_file("../../../NC_alt/sc_block_20/sc_block.shp")
estimate = estimatecvap(my_blocks, us.states.SC, 1, 0.05)
estimate.to_csv("final_test.csv", index=False)
