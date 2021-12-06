from typing import Dict
import geopandas as gpd
import tqdm
import gurobipy as gp
from gurobipy import GRB
import math


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
    return units[units[enacted_col] != units[proposed_col]][pop_col].sum()