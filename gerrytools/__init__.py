"""
``gerrytools`` is a light package for redistricting analysis essentials,
making geographic data operations, map visualizations, plan
evaluation, and data retrieval simple.
"""

import geopandas

geopandas.options.use_pygeos = False


__version__ = "1.2.0"


try:
    from . import mgrp

    mgrp_available = True
except ImportError as e:
    print(f"Optional module 'mgrp' could not be imported: {e}")
    mgrp_available = False


__all__ = [
    "__version__",
    "mgrp_available",  # Add other public symbols as needed
]
