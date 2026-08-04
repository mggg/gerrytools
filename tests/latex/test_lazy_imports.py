"""Regression tests for the lazy ``gerrytools.latex`` public surface (PEP 562)."""

import subprocess
import sys

import pytest


def _run_in_subprocess(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True)


def test_importing_latex_package_loads_no_heavy_dependencies():
    # The package __init__ has no eager imports, so importing it costs nothing until a
    # name is touched.
    _run_in_subprocess(
        "import sys\n"
        "import gerrytools.latex\n"
        "heavy = [name for name in ('matplotlib', 'scipy', 'pandas', 'PyQt6')"
        " if name in sys.modules]\n"
        "assert heavy == [], heavy\n"
    )


def test_accessing_textable_skips_plot_stack_and_qt():
    # The table stack legitimately imports pandas, and gerrytools.colors' package __init__
    # (imported for color conversion) still drags matplotlib.pyplot and scipy in, so those
    # cannot be asserted absent here. What laziness buys on attribute access is that the
    # TikZ plot modules and the Qt preview stack stay unloaded.
    _run_in_subprocess(
        "import sys\n"
        "from gerrytools.latex import TexTable\n"
        "loaded = [name for name in ('gerrytools.latex.paintball',"
        " 'gerrytools.latex.seatsvotes', 'gerrytools.latex._tikz_plot_base', 'PyQt6')"
        " if name in sys.modules]\n"
        "assert loaded == [], loaded\n"
    )


def test_lazy_exports_resolve_cache_and_reject_unknown_names():
    import gerrytools.latex as latex_package
    from gerrytools.latex import UNSET, Unset

    assert isinstance(UNSET, Unset)
    assert latex_package.TexTable is not None
    assert "TexTable" in vars(latex_package)  # cached after first access
    assert set(latex_package.__all__) <= set(dir(latex_package))

    with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
        _ = latex_package.nonexistent


def test_tikz_plot_names_are_the_primary_public_exports():
    import gerrytools.latex as latex_package

    assert "TikzPaintballPlot" in latex_package.__all__
    assert "TikzSeatsVotesPlot" in latex_package.__all__
