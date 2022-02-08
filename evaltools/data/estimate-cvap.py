
import pandas as pd
import numpy as np
from pandas import DataFrame

def estimate_cvap(base, cvap_source, cvap_source_identifier, percentage_cap=None, zero_fill=None) -> DataFrame:
    """
    Arguments:
       base (DataFrame): A dataframe containing population values for a state on the desired geographical units.
                         Must contain a column for cvap_source units. 
       cvap_source(DataFrame): A dataframe containing the cvap estimates. This will likely be on 2010 tracts or 
                               block groups. Assumes that source unit unique identifiers are census geometries
                               with county fips in the identifier. 
       cvap_source_identifier(string): Column in cvap_source representing the unique identifier. 
       percentage_cap (float): Number representing where to cap the weighting ratio of CVAP to VAP20. After 
                               this percentage barrier is passed, the percentage will be set to 1. 
       zero_fill (float): Fill in ratio for CVAP to VAP20 when there is 0 CVAP in the area.

    Returns: 
       weightedbase (DataFrame): A dataframe containing the newly estimated 2020 CVAP numbers on the same geographical units as base.  
    """

    if percentage_cap is None or zero_fill is None:
       raise ValueError("Cannot compute CVAP without percentage cap and zero fill args.")

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
        cvap + "%": cvap_source[cvap].sum()/cvap_source[vap].sum()
        for cvap, vap in names
    }

    cvappcts = [cvap + "%" for cvap, _ in names]

    for pct in cvappcts:        
        for ix, row in cvap_source.iterrows():
            if pd.isna(row[pct]):
                county = row[cvap_source_identifier][2:5]
                county_avg = np.mean(cvap_source[pct][cvap_source[cvap_source_identifier].str[2:5] == county])
                cvap_source.at[ix, pct] = county_avg if not pd.isna(county_avg) else statewide[pct]
        
        cvap_source[pct] = cvap_source[pct] \
            .replace(0, zero_fill)  \
            .apply(lambda c: 1 if c > 1 else c)
        
    # Assert we don't have any percentages over percentage_cap.
    assert all(
        np.all(cvap_source[p + "%"] <= 1)
        for p, _ in names
    )
    # Set indices and create a mapping from IDs to weights.

    cvap_source = cvap_source.set_index(cvap_source_identifier)    
    cvap_source = cvap_source[[p + "%" for p, _ in names]]
    weights = cvap_source.to_dict(orient="index")

    # Weight base!
    basepairs = list(zip(
        ["CVAP20_EST", "BCVAP20_EST", "HCVAP20_EST", "ACVAP20_EST", "WCVAP20_EST"],
        ["CVAP19%", "BCVAP19%", "HCVAP19%", "ASIANCVAP19%", "WCVAP19%"],
        ["VAP20", "APBVAP20", "HVAP20", "ASIANVAP20", "WVAP20"]
    ))

    groups = list(base.groupby(cvap_source_identifier))
  
    for ix, group in groups:
        for name, weight, source in basepairs:
            group[weight] = weights[ix][weight]
            group[name] = group[weight] * group[source]

    # Re-create a dataframe.
    weightedbase = pd.concat(frame for _, frame in groups)

    return weightedbase

