
"""
Provides ease-of-use functionality for geographic and geometric operations.
"""

from .dissolve import dissolve, hierarchical_block_dissolve
from .dualgraph import dualgraph
from .unitmap import unitmap, invert
from .scores import reock
from ..processing import Graph, Partition

__all__ = [
    "dissolve",
    "hierarchical_block_dissolve",
    "dualgraph",
    "unitmap",
    "invert",
    "reock"
]
