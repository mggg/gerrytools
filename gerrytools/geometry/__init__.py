"""
Provides ease-of-use functionality for geographic and geometric operations.
"""

from .dataframe import dataframe
from .dissolve import dissolve
from .dualgraph import dualgraph
from .optimize import (
    arealoverlap,
    calculate_dispersion,
    minimize_dispersion,
    minimize_dispersion_with_parity,
    minimize_parity,
    optimalrelabeling,
    populationoverlap,
)
from .unitmap import invert, unitmap
from .updater import dispersion_updater_closure

__all__ = [
    "minimize_dispersion",
    "minimize_dispersion_with_parity",
    "minimize_parity",
    "calculate_dispersion",
    "dispersion_updater_closure",
    "dissolve",
    "dualgraph",
    "unitmap",
    "invert",
    "dataframe",
    "populationoverlap",
    "optimalrelabeling",
    "arealoverlap",
]
