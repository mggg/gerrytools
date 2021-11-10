
"""
Provides ease-of-use functionality for geographic and geometric operations.
"""

from .dissolve import dissolve
from .dualgraph import dualgraph
from .unitmap import unitmap, invert
from ..processing import Graph, Partition

__all__ = [
    "dissolve",
    "dualgraph",
    "unitmap",
    "invert"
]
