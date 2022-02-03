
"""

.. include:: ../docs/introduction.md
"""

from .plotting import (
    drawplan, drawgraph, districtr, flare, purples, redbluecmap, PlotSpecification
)
from .geometry import (
    dissolve, dualgraph, invert, unitmap, minimize_dispersion, minimize_parity,
    minimize_dispersion_with_parity, calculate_dispersion, dispersion_updater_closure
)
from .scores import (
    splits, pieces, deviations, contiguous, unassigned_units,
    unassigned_population
)
from .utilities import rename, JSONtoObject, objectify
from .data import acs5, cvap, census

