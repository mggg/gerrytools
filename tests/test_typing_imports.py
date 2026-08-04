"""Import-weight regression tests for the gerrytools.typing alias hub.

The hub serves dependency-light aliases (Color, CategoryKey, ...) to modules like the LaTeX
backends that never touch geopandas or matplotlib. The heavy aliases are built lazily via module
``__getattr__``, so importing the hub itself must not load the geo/plotting stack.
"""

import subprocess
import sys

HEAVY_MODULES = ("geopandas", "matplotlib", "pyproj", "pandas")


def _run_check(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True)


def test_importing_typing_does_not_load_the_geo_plotting_stack():
    check = (
        "import sys; import gerrytools.typing; "
        f"loaded = [m for m in sys.modules if m.startswith({HEAVY_MODULES!r})]; "
        "assert not loaded, loaded"
    )
    _run_check(check)


def test_light_aliases_are_usable_without_heavy_imports():
    check = (
        "import sys; from gerrytools.typing import Color, CategoryKey, TikzLineStyle; "
        f"loaded = [m for m in sys.modules if m.startswith({HEAVY_MODULES!r})]; "
        "assert not loaded, loaded"
    )
    _run_check(check)


def test_heavy_aliases_still_resolve_on_demand():
    check = (
        "from gerrytools.typing import BinsType, CRSLike, GeoColorMap, GeoSource, "
        "LegendHandle, NumericArrayLike; "
        "from matplotlib.artist import Artist; "
        "assert LegendHandle is Artist; "
        "import gerrytools.typing as t; "
        "assert t.GeoSource is t.GeoSource  # cached after first access"
    )
    _run_check(check)
