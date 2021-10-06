
from collections import namedtuple

"""
Creates a generalized schema for storing ensembles.
"""
ensemble_schema = namedtuple(
    "ensemble_schema",
    [
        "type", "num_districts", "epsilon", "chain_type", "pop_col", "metrics",
        "pov_party", "elections", "party_statewide_share"
    ]
)

assignment_schema = namedtuple(
    "assignment_schema",
    [
        "TOTPOP", "num_cut_edges", "seats", "contiguity", "population_assigned",
        "unassigned_units"
    ],
    defaults=(
        
    )
)
