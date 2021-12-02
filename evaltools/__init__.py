
"""

.. include:: ../docs/introduction.md
"""

from .plotting import drawplan, drawgraph, redblue, PlotSpecification
from .colors import districtr, redbluecmap
from .geography import dissolve, dualgraph
from .evaluation import (
    splits, pieces, deviations, contiguous, unassigned_units,
    unassigned_population
)
from .numbering import (
    minimize_dispersion, minimize_parity, minimize_dispersion_with_parity,
    calculate_dispersion, dispersion_updater_closure
)

"""
__all__ = [
    "districtr", "redblue", "drawplan", "dualgraph", "dissolve",
    "drawgraph", "Graph", "Partition", "splits", "pieces", "deviations",
    "ensemble_schema", "assignment_schema", "contiguous", "unassigned_units",
    "unassigned_population", "AssignmentCompressor",
    "minimize_dispersion", "minimize_parity", "minimize_dispersion_with_parity",
    "calculate_dispersion", "dispersion_updater_closure"
]
"""
