"""Facilities for processing data and districting plans in a standardized fashion."""

from . import uscensus
from .geometries import dualgraphs20, geometries20, vtds20

# The Census surface re-exports track uscensus.__all__ so the two packages never drift.
from .uscensus import *  # noqa: F403

__all__ = [
    # Lab-processed geometry downloads
    "vtds20",
    "dualgraphs20",
    "geometries20",
]
__all__ += uscensus.__all__
