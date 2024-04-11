import math
from typing import Any, Callable, Dict, List

import geopandas as gpd
import gurobipy as gp
import pandas as pd
import tqdm
from gurobipy import GRB
from scipy.optimize import linear_sum_assignment as lsa


def arealoverlap(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    assignment: str = "DISTRICT",
    crs=None,
) -> pd.DataFrame:
    r"""
    Given two GeoDataFrames, each encoding districting plans, computes the areal
    overlap between each pair of districts. `left` is the districting plan to be
    relabeled (e.g. a proposed districting plan) and `right` is the districting
    plan with district labels we're trying to match (e.g. an enacted districting
    plan). If `left` (denoted :math:`L`) has :math:`n` districts and `right` (denoted :math:`R`) has
    :math:`m` districts, an :math:`n \times m` matrix :math:`C` is computed, where the entry
    :math:`M_{ij}` represents the area of the intersection of the districts :math:`L_i` and
    :math:`R_j`. :math:`C` is represented as a pandas DataFrame, where the row indices are
    the labels in `left`, and are the preimage of the label mapping; column indices
    are the labels in `right`, and are the image of the label mapping.

    Args:
        left (pd.DataFrame): GeoDataFrame whose labels are the preimage of the
            relabeling.
        right (pd.DataFrame): GeoDataFrame whose labels are the image of the
            relabeling.
        assignment (str): Column on `left` and `right` which contains the district
            identifier.

    Returns:
        Cost matrix :math:`C`, represented as a DataFrame.
    """
    # Force the two things to be the same CRS (or the provided CRS)
    if crs:
        left = left.to_crs(crs)
        right = right.to_crs(crs)
    else:
        right = right.to_crs(left.crs)

    # An empty list of records for everything.
    records = []

    # Figure out the image as we go along.
    image = []

    # Now, iterate over each dissolved district, finding the areal overlap with
    # enacted districts.
    for pid, pdistrict in zip(left[assignment], left["geometry"]):
        image.append(pid)
        records.append(
            {
                eid: pdistrict.intersection(edistrict).area
                for eid, edistrict in zip(right[assignment], right["geometry"])
            }
        )

    # Create a dataframe from that!
    weighted = pd.DataFrame.from_records(records)
    weighted.index = image

    return weighted


def populationoverlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    identifier: str = "GEOID20",
    population: str = "TOTPOP20",
    assignment: str = "DISTRICT",
) -> pd.DataFrame:
    """
    Given two unit-level DataFrames — i.e. two dataframes where each row represents
    an atomic unit like Census blocks or VTDs, and each row contains a district
    assignment — computes the amount of population shared by each pair of districts.
    `left` is the districting plan to be relabeled (e.g. a proposed districting
    plan) and `right` is the districting plan with district labels we're trying
    to match (e.g. an enacted districting plan). If `left` (denoted :math:`L`) has
    :math:`n` districts and `right` (denoted :math:`R`) has :math:`m` districts, an
    :math:`n \times m` matrix :math:`C` is computed, where the entry :math:`M_{ij}` represents
    the population shared by the districts :math:`L_i` and :math:`R_j`. :math:`C` is
    represented as a pandas DataFrame, where the row indices are the labels in
    `left`, and are the preimage of the label mapping; column indices are the labels
    in `right`, and are the image of the label mapping.

    Args:
        left (pd.DataFrame): DataFrame whose labels are the preimage of the relabeling.
        right (pd.DataFrame): DataFrame whose labels are the image of the relabeling.
        identifier (str): Column on `left` and `right` which contains the unique
            identifier for each unit.
        population (str): Column on `left` and `right` which contains the population
            total for each unit. This can be modified to be `any` population.
        assignment (str): Column on `left` and `right` that denotes district membership.

    Returns:
        A DataFrame whose row names are the preimage of the relabeling, column names
        are the image of the relabeling, and values edge weights; a cost matrix.
    """
    # Make sure types are appropriate.
    right[assignment] = right[assignment].astype(str)
    right[identifier] = right[identifier].astype(str)

    left[assignment] = left[assignment].astype(str)
    left[identifier] = left[identifier].astype(str)

    # The preimage is the set of proposed-plan labels.
    preimage = list(left[assignment].unique())

    # Create a list of records from which we'll make a dataframe!
    records = []

    # For each label in the preimage, find the district which shares the most
    # population; this is the cost matrix.
    for fromlabel in preimage:
        # Find the blocks in the "from" district and see how much population each
        # district shares with each enacted district.
        subleft = left[left[assignment] == fromlabel]
        subright = right[right[identifier].isin(subleft[identifier])]

        # Aggregate shared blocks based on district label.
        agg = subright.groupby(assignment, as_index=False).sum()

        # Now figure out the shared populations.
        records.append(
            {i: shared for i, shared in zip(agg[assignment], agg[population])}
        )

    # Create the cost matrix!
    C = pd.DataFrame.from_records(records).fillna(0)
    C.index = preimage

    return C


def optimalrelabeling(
    left: Any,
    right: Any,
    maximize: bool = True,
    costmatrix: Callable = populationoverlap,
) -> dict:
    """
    Finds the optimal relabeling for two districting plans.

    Args:
        left (Any): Data structure which can be passed to `costmatrix` to construct
            a cost matrix. District labels will be the preimage of the relabeling.
            If the default `costmatrix` function is used, these must be pandas
            DataFrames where one row corresponds to one atomic unit (e.g. Census
            blocks), with at least three columns: one denoting a unique geometric
            identifier (e.g. `GEOID20`), one denoting the districting assignment,
            and another denoting the population of choice. If
            `gerrytools.geometry.optimize.arealoverlap()` is used, these must be
            GeoDataFrames where one row corresponds to one district, and one
            column denotes the districts' unique identifiers.
        right (Any): Data structure which can be passed to `costmatrix` to construct
            a cost matrix. District labels will be the image of the relabeling.
            If the default `costmatrix` function is used, these must be pandas
            DataFrames where one row corresponds to one atomic unit (e.g. Census
            blocks), with at least three columns: one denoting a unique geometric
            identifier (e.g. `GEOID20`), one denoting the districting assignment,
            and another denoting the population of choice. If
            `gerrytools.geometry.optimize.arealoverlap()` is used, these must be
            GeoDataFrames where one row corresponds to one district, and one
            column denotes the districts' unique identifiers.
        maximize (bool): Are we finding the largest or smallest linear sum over
            the cost matrix? Defaults to `maximize=True`.
        costmatrix (Callable): The function (or partial function) which consumes
            `left` and `right` and spits out a cost matrix. This cost matrix is
            assumed to be a pandas DataFrame, with row indices old district labels
            and column names new district labels. Examples of these are
            `gerrytools.geometry.optimize.populationoverlap()` and
            `gerrytools.geometry.optimize.arealoverlap()`.

    Returns:
        A dictionary which maps district labels in `left` to district labels in
        `right`, according to the weighting scheme applied in `costmatrix`.

    This is an `assignment problem <https://bit.ly/3wnyS4F>`_
    and is equivalently a `(min/max)imal bipartite matching problem <http://bit.ly/2OfwUeh>`_.
    Consider two districting plans :math:`L` and :math:`R`, with :math:`n` and :math:`m` districts
    respectively. Set :math:`V_L` and :math:`V_R` to be sets of vertices such that
    a vertex :math:`l_i` in :math:`V_L` corresponds to the district :math:`L_i` in :math:`L`, and
    similarly for vertices :math:`r_j` in :math:`V_R`; draw edges :math:`(l_i, r_j)` for each
    :math:`i` from :math:`1` to :math:`n,` and each :math:`j` from :math:`1` to :math:`m.`
    In doing so, we construct the `bipartite graph <https://bit.ly/39rDldy>` :math:`K_{n,m}`:

    We then assign each edge a weight according to some function
    :math:`f: L\times R \to \\mathbb{R}`, which consumes a pair of districts and returns
    a number. For example, this function could be the amount of area shared by
    the districts :math:`L_i` and :math:`R_j`, like in
    :func:`gerrytools.geometry.optimize.arealoverlap()`,
    or the amount of population the districts share, like in
    :func:`gerrytools.geometry.optimize.populationoverlap()`.

    We then seek to find the set of weighted edges :math:`M` such that all vertices
    :math:`l_i` and :math:`r_j` appear at most once in :math:`M`, and that the sum of :math:`M`'s
    weights is as small (or as large) as possible. To do so, we take the adjacency
    matrix :math:`A` of our graph :math:`K_{n,m}`, where the :math:`i, j`th entry records
    the weight of the edge :math:`(l_i, r_j`). Then, we want to select at most one entry
    in each row and column, and ensure those entries have the smallest (or greatest)
    possible sum. Using the `Jonker-Volgenant algorithm <DOI:10.1109/TAES.2016.140952>`_ (as
    implemented by scipy), we can find the row and column indices of these entries,
    and retrieve the district label pairs corresponding to each. The algorithm
    achieves :math:`\textbf{O}(N^3)` worst-case running time, where :math:`N = \\max(n, m)`.
    """
    # Our cost function should compute the weights between left and right. First,
    # we want to identify the indices of the preimage (row index) and column
    C = costmatrix(left, right)
    preimage, image = list(C.index), list(C)

    # Now we do our linear sum assignment, getting back the indices which maximize
    # the total weight on the edges!
    preimageindices, imageindices = lsa(C, maximize=maximize)
    preimage = [preimage[i] for i in preimageindices]
    image = [image[i] for i in imageindices]

    # Zip the preimage and image into a dict, and we're done!
    return dict(zip(preimage, image))


def ensure_column_types(
    units: gpd.GeoDataFrame,
    columns: List[str],
    expression: Callable[[Any], bool] = lambda x: x.startswith("int"),
) -> bool:
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


def minimize_dispersion(
    units: gpd.GeoDataFrame,
    enacted_col: str,
    proposed_col: str,
    pop_col: str,
    extra_constraints=None,
    verbose: bool = False,
) -> Dict[str, str]:
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

    if not ensure_column_types(
        units, [pop_col], lambda x: x.startswith("int") or x.startswith("float")
    ):
        raise TypeError("Your pop col must be an int or float type!")

    districts = list(set(units[proposed_col].astype(int)))
    model = gp.Model("state_model")
    model.setParam("OutputFlag", int(verbose))

    numbering = model.addVars(
        len(districts), len(districts), vtype=GRB.BINARY, name="numbering"
    )

    exprs = []
    if verbose:

        def wrapper(x):
            return tqdm.tqdm(x)

    else:

        def wrapper(x):
            return x

    for district in wrapper(districts):  # iter over proposed
        enacted_intersection = (
            units[units[proposed_col] == district]
            .groupby(enacted_col)
            .sum()[pop_col]
            .to_dict()
        )

        for (
            enacted,
            overlap_pop,
        ) in enacted_intersection.items():  # maximize overlap; minimize dispersion
            exprs.append(numbering[district - 1, enacted - 1] * overlap_pop)

        if extra_constraints is not None:
            extra_constraints(model, numbering, district, districts)

    model.addConstrs(
        (numbering.sum("*", v) == 1 for v in range(len(districts))), name="v"
    )
    model.addConstrs(
        (numbering.sum(v, "*") == 1 for v in range(len(districts))), name="h"
    )

    obj = gp.quicksum(exprs)
    model.setObjective(obj, GRB.MAXIMIZE)

    model.optimize()

    solution = model.getVars()

    numbering_mapping = {}
    for i in range(len(districts)):
        for j in range(len(districts)):
            if solution[i * len(districts) + j].x:
                numbering_mapping[districts[i]] = districts[j]

    return numbering_mapping


def minimize_parity(
    units: gpd.GeoDataFrame,
    enacted_col: str,
    proposed_col: str,
    pop_col: str,
    verbose: bool = False,
) -> Dict[str, bool]:
    """
    Minimize odd->even parity shift in a state given an column with enacted districts
    and a column with proposed numberings. Returns a dictionary with the parity of the
    proposed cols. Used in WI. Assumes that district labels are 1-indexed.

    Args:
        units: The units to optimize on. E.g. Census blocks.
        enacted_col: The column in the GeoDataFrame with the enacted districts.
        proposed_col: The column in the GeoDataFrame with the proposed districts.
        pop_col: The column in the GeoDataFrame with population counts.
        verbose: If true, do not suppress solver output. Otherwise, stay quiet.

    Returns:
        A dictionary mapping proposed labels to booleans values representing the optimal parity.
        (True if even, False odd).
    """
    if not ensure_column_types(units, [enacted_col, proposed_col]):
        raise TypeError("Your enacted and proposed columns must be an int type!")

    if not ensure_column_types(
        units, [pop_col], lambda x: x.startswith("int") or x.startswith("float")
    ):
        raise TypeError("Your pop col must be an int or float type!")

    model = gp.Model("parity_model")
    model.setParam("OutputFlag", int(verbose))

    districts = list(set(units[proposed_col].astype(int)))
    districts_even = model.addVars(
        len(districts), vtype=GRB.BINARY, name="districts_even"
    )

    exprs = []
    if verbose:

        def wrapper(x):
            return tqdm.tqdm(x)

    else:

        def wrapper(x):
            return x

    for i, block in wrapper(units[[enacted_col, proposed_col, pop_col]].iterrows()):
        district = districts.index(int(block[proposed_col]))
        isOdd = bool((int(block[enacted_col]) % 2) == 1)
        exprs.append(isOdd * districts_even[district] * block[pop_col])

    obj = gp.quicksum(exprs)
    model.addConstr(gp.quicksum(districts_even) == math.floor(len(districts) / 2), "c0")

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    mapping = {}
    for i, v in enumerate(model.getVars()):
        mapping[districts[i]] = bool(v.x)

    return mapping


def minimize_dispersion_with_parity(
    units: gpd.GeoDataFrame,
    enacted_col: str,
    proposed_col: str,
    pop_col: str,
    extra_constraints=None,
) -> Dict[str, str]:
    """
    Minimize dispersion and odd->even parity shift in a state given an column with
    enacted districts and a column with proposed numberings. Returns a dictionary
    relabeling the proposed cols. Used in WI. Assumes that district labels are 1-indexed.

    Args:
        units: The units to optimize on. E.g. Census blocks.
        enacted_col: The column in the GeoDataFrame with the enacted districts.
        proposed_col: The column in the GeoDataFrame with the proposed districts.
        pop_col: The column in the GeoDataFrame with population counts.
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
                    numbering[district - 1, x - 1]
                    for x in range(1, len(districts) + 1)
                    if x % 2 == 0
                )
                == 1
            )

        extra_constraints(model, numbering, district, districts)

    return minimize_dispersion(
        units, enacted_col, proposed_col, pop_col, parity_constraint
    )


def calculate_dispersion(
    units: gpd.GeoDataFrame, enacted_col: str, proposed_col: str, pop_col: str
) -> int:
    """
    Calculates core dispersion in a state given an column with enacted districts
    and a column with proposed numberings. Used in WI.

    Args:
        units: The units to optimize on. E.g. Census blocks.
        enacted_col: The column in the GeoDataFrame with the enacted districts.
        proposed_col: The column in the GeoDataFrame with the proposed districts.
        pop_col: The column in the GeoDataFrame with population counts.

    Returns:
        An integer of the absolute number of people who changed districts.
    """
    if units[enacted_col].dtype != units[proposed_col].dtype:
        raise TypeError("Your enacted and proposed columns must have the same type!")

    return units[units[enacted_col] != units[proposed_col]][pop_col].sum()


def calculate_dispersion_per_district(
    units: gpd.GeoDataFrame, enacted_col: str, proposed_col: str, pop_col: str
) -> Dict[int, int]:
    """
    Calculates dispersion per district in a state given a column with enacted districts
    and a column with proposed districts. Used in GA.

    Args:
        units: The units to optimize on. E.g. census blocks.
        enacted_col: The column in the GeoDataFrame with the enacted districts.
        proposed_col: The column in the GeoDataFrame with the proposed districts.
        pop_col: The column in the GeoDataFrame with population counts.

    Returns:
        A dictionary with keys as districts, and values as the number of
        people displaced from the enacted plan to the proposed plan.
    """
    if units[enacted_col].dtype != units[proposed_col].dtype:
        raise TypeError("Your enacted and proposed columns must have the same type!")

    dispersion_dict = {
        district: sum(
            units[pop_col][
                (units[enacted_col] == district) & (units[proposed_col] != district)
            ]
        )
        for district in sorted(units[enacted_col].unique())
    }

    return dispersion_dict
