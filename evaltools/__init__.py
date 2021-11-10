
"""
A package for processing districting plans, from retrieval to processing to
visualization.
"""
from .mapping import drawplan, drawgraph
from .colors import districtr, redblue
from .geography import dissolve, dualgraph
from .evaluation import (
    splits, pieces, deviations, ensemble_schema, assignment_schema,
    contiguous, unassigned_units, unassigned_population
)
from .numbering.optimize import minimize_dispersion, minimize_parity, minimize_dispersion_with_parity, calculate_dispersion

__all__ = [
    "districtr", "redblue", "drawplan", "dualgraph", "dissolve",
    "drawgraph", "Graph", "Partition", "splits", "pieces", "deviations",
    "ensemble_schema", "assignment_schema", "contiguous", "unassigned_units",
    "unassigned_population", "AssignmentCompressor",
    "minimize_dispersion", "minimize_parity", "minimize_dispersion_with_parity", "calculate_dispersion"
]
