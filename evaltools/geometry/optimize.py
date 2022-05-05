
from typing import Dict, List, Any, Callable, Tuple
import geopandas as gpd
import tqdm
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linear_sum_assignment as lsa
import math
import pandas as pd


def populationoverlap(
        left: pd.DataFrame, right: pd.DataFrame, identifier="GEOID20",
        population="TOTPOP20", assignment="DISTRICT"
    ) -> pd.DataFrame:
    """
    Given two unit-level districting assignments with some population attached,
    report the amount of population shared by district \(k\) in `left` and district
    \(k\) in `right`.

    Args:
        left (pd.DataFrame): DataFrame whose labels are the domain of the relabeling.
        right (pd.DataFrame): DataFrame whose labels are the image of the relabeling.
        identifier (str): Column on `left` and `right` which contains the unique
            identifier for each unit.
        population (str): Column on `left` and `right` which contains the population
            total for each unit. This can be modified to be `any` population.
        assignment (str): Column on `left` and `right` that denotes district membership.

    Returns:
        A DataFrame whose row names are the domain of the relabeling, column names
        are the image of the relabeling, and values edge weights.
    """
    # Identify the district labels in the right dataframe (i.e. the district labels
    # we're mapping to).
    left[assignment] = left[assignment].astype(str)
    right[assignment] = right[assignment].astype(str)

    # The domain is the intersection of the available districts on each plan.
    domain = list(set(left[assignment]) & set(right[assignment]))

    # Create a bucket for results. This should be a list of dictionaries mapping
    # column names to weights.
    records = []

    # For each district in the image, identify the overlap and report weights.
    for fromdistrict in domain:
        # First, get all the rows in the left dataframe in the desired district.
        # Then, get all the same geometries in the right dataframe.
        subleft = left[left[assignment] == fromdistrict]
        subright = right[right[identifier].isin(subleft[identifier])]

        # Now, aggregate each.
        rightaggregate = subright[[assignment, population]].groupby(assignment, as_index=False).sum()
        leftaggregate = subleft[[assignment, population]].groupby(assignment, as_index=False).sum()

        # Get the total population and sum; there should be exactly one row in
        # `leftaggregate`.
        totpop = leftaggregate[population].sum()

        # Create a record.
        record = {}

        # Iterate over (district, totpop) pairs to create weighted edges.
        for todistrict, subpopulation in zip(rightaggregate[assignment], rightaggregate[population]):
            record[todistrict] = subpopulation/totpop

        records.append(record)
    
    # Set up a dataframe from the records.
    weighting = pd.DataFrame.from_records(records)
    weighting.index = domain
    weighting = weighting.fillna(0)

    return weighting


def optimalrelabeling(left, right, maximize=True, cost=populationoverlap):
    """
    Given two dataframes, each with three columns --- one for unique geometric
    identifiers, one for districts, and one for some score (e.g. total population)
    --- we compute an optimal relabeling, where "optimal" is the relabeling which
    maximizes(/minimizes) the population overlap between two districts.

    Args:
        left: 
    """
    # Our cost function should compute the weights between left and right. First,
    # we want to identify the indices tho.
    C = populationoverlap(left, right)
    domain, image = list(C.index), list(C)
    
    # Now we do our linear sum assignment, getting back the indices which maximize
    # the total weight on the edges!
    domainindices, imageindices = lsa(C, maximize=maximize)
    domain = [domain[i] for i in domainindices]
    image = [image[i] for i in imageindices]

    # Zip the domain and image into a dict, and we're done!
    return dict(zip(domain, image))


def ensure_column_types(units: gpd.GeoDataFrame, columns: List[str], expression: Callable[[Any], bool] = lambda x: x.startswith("int")) -> bool:
    """
    Ensure that the given columns in the GeoDataFrame have a type matching the 
    filtering expression. This gives more flexibility over other type checking 
    methods.

    Args:
        units: The GeoDataFrame to check.
        colunms: A list of the columns to type check.
        expression: A function that returns true if the type name is correct and false otherwise.

    Returns:
        A boolean indicating if all the columns checked match the expression.
    """
    return all([expression(x.name) for x in units[columns].dtypes])

def minimize_dispersion(units: gpd.GeoDataFrame, enacted_col: str, proposed_col: str, pop_col: str, extra_constraints = None, verbose: bool = False) -> Dict[str, str]:
    """
    Minimize core dispersion in a state given an column with enacted districts
    and a column with proposed numberings. Returns a dictionary relabeling the
    proposed cols. Used in WI. Assumes that district labels are 1-indexed.

    Args:
        units: The units to optimize on. E.g. Census blocks.
        enacted_col: The column in the GeoDataFrame with the enacted districts.
        proposed_col: The column in the GeoDataFrame with the proposed districts.
        extra_constraints: Optional; A function that can add extra constraints 
            to the model, such as parity (in the case of WI).
        verbose: If true, do not suppress solver output. Otherwise, stay quiet.

    Returns:
        A dictionary mapping proposed labels to optimized labels.
    """
    if not ensure_column_types(units, [enacted_col, proposed_col]):
        raise TypeError("Your enacted and proposed columns must be an int type!")

    if not ensure_column_types(units, [pop_col], lambda x: x.startswith("int") or x.startswith("float")):
        raise TypeError("Your pop col must be an int or float type!") 

    districts = list(set(units[proposed_col].astype(int)))
    model = gp.Model("state_model")
    model.setParam('OutputFlag', int(verbose))

    numbering = model.addVars(len(districts), len(districts), vtype=GRB.BINARY, name="numbering")

    exprs = []
    if verbose:
        wrapper = lambda x: tqdm.tqdm(x)
    else:
        wrapper = lambda x: x

    for district in wrapper(districts):  # iter over proposed
        enacted_intersection = units[units[proposed_col] == district].groupby(enacted_col).sum()[pop_col].to_dict()

        for enacted, overlap_pop in enacted_intersection.items(): # maximize overlap; minimize dispersion
            exprs.append(numbering[district-1, enacted-1]*overlap_pop)
            
        if extra_constraints is not None:
            extra_constraints(model, numbering, district, districts)

    model.addConstrs((numbering.sum("*", v)==1 for v in range(len(districts))), name="v")
    model.addConstrs((numbering.sum(v, "*")==1 for v in range(len(districts))), name="h")

    obj = gp.quicksum(exprs)
    model.setObjective(obj, GRB.MAXIMIZE)

    model.optimize()

    solution = model.getVars()

    numbering_mapping = {}
    for i in range(len(districts)):
        for j in range(len(districts)):
            if solution[i*len(districts)+j].x:
                numbering_mapping[districts[i]] = districts[j]

    return numbering_mapping

def minimize_parity(units: gpd.GeoDataFrame, enacted_col: str, proposed_col: str, pop_col: str, verbose: bool = False) -> Dict[str, bool]:
    """
    Minimize odd->even parity shift in a state given an column with enacted districts
    and a column with proposed numberings. Returns a dictionary with the parity of the
    proposed cols. Used in WI. Assumes that district labels are 1-indexed.

    Args:
        units: The units to optimize on. E.g. Census blocks.
        enacted_col: The column in the GeoDataFrame with the enacted districts.
        proposed_col: The column in the GeoDataFrame with the proposed districts.
        verbose: If true, do not suppress solver output. Otherwise, stay quiet.

    Returns:
        A dictionary mapping proposed labels to booleans values representing the optimal parity. 
        (True if even, False odd).
    """
    if not ensure_column_types(units, [enacted_col, proposed_col]):
        raise TypeError("Your enacted and proposed columns must be an int type!")

    if not ensure_column_types(units, [pop_col], lambda x: x.startswith("int") or x.startswith("float")):
        raise TypeError("Your pop col must be an int or float type!") 

    model = gp.Model("parity_model")
    model.setParam('OutputFlag', int(verbose))

    districts = list(set(units[proposed_col].astype(int)))
    districts_even = model.addVars(len(districts), vtype=GRB.BINARY, name="districts_even")

    exprs = []
    if verbose:
        wrapper = lambda x: tqdm.tqdm(x)
    else:
        wrapper = lambda x: x
    for i, block in wrapper(units[[enacted_col, proposed_col, pop_col]].iterrows()):
        district = districts.index(int(block[proposed_col]))
        isOdd = bool((int(block[enacted_col]) % 2) == 1)
        exprs.append(isOdd * districts_even[district] * block[pop_col])

    obj = gp.quicksum(exprs)
    model.addConstr(gp.quicksum(districts_even) == math.floor(len(districts)/2), "c0")

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    mapping = {}
    for i, v in enumerate(model.getVars()):
        mapping[districts[i]] = bool(v.x)

    return mapping

def minimize_dispersion_with_parity(units: gpd.GeoDataFrame, enacted_col: str, proposed_col: str, pop_col: str, extra_constraints = None) -> Dict[str, str]:
    """
    Minimize dispersion and odd->even parity shift in a state given an column with 
    enacted districts and a column with proposed numberings. Returns a dictionary 
    relabeling the proposed cols. Used in WI. Assumes that district labels are 1-indexed.

    Args:
        units: The units to optimize on. E.g. Census blocks.
        enacted_col: The column in the GeoDataFrame with the enacted districts.
        proposed_col: The column in the GeoDataFrame with the proposed districts.
        extra_constraints: Optional; A function that can add extra constraints 
            to the model, such as parity (in the case of WI).

    Returns:
        A dictionary mapping proposed labels to optimized labels.
    """
    optimal_parity_mapping = minimize_parity(units, enacted_col, proposed_col, pop_col)
    def parity_constraint(model, numbering, district, districts):
        if optimal_parity_mapping[district]:
            model.addConstr(
                gp.quicksum(
                    numbering[district-1, x-1] for x in range(1, len(districts)+1) if x%2 == 0
                ) == 1
            )

        extra_constraints(model, numbering, district, districts)

    return minimize_dispersion(units, enacted_col, proposed_col, pop_col, parity_constraint)


def calculate_dispersion(units: gpd.GeoDataFrame, enacted_col: str, proposed_col: str, pop_col: str) -> int:
    """
    Calculates core dispersion in a state given an column with enacted districts
    and a column with proposed numberings. Used in WI.

    Args:
        units: The units to optimize on. E.g. Census blocks.
        enacted_col: The column in the GeoDataFrame with the enacted districts.
        proposed_col: The column in the GeoDataFrame with the proposed districts.

    Returns:
        An integer of the absolute number of people who changed districts.
    """
    if units[enacted_col].dtype != units[proposed_col].dtype:
        raise TypeError("Your enacted and proposed columns must have the same type!")

    return units[units[enacted_col] != units[proposed_col]][pop_col].sum()