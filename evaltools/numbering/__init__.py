
"""
Various numbering optimization methods to minimize core dispersion
and address numbering parity concerns. Used in WI.
"""

from .optimize import (
    minimize_dispersion, minimize_dispersion_with_parity, minimize_parity,
    calculate_dispersion
)

from .updater import dispersion_updater_closure

__all__ = [
    "minimize_dispersion",
    "minimize_dispersion_with_parity",
    "minimize_parity",
    "calculate_dispersion",
    "dispersion_updater_closure"
]
