"""Shared fixtures for plotting tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def testing_gdf():
    """Load the 12×12 grid GeoPackage used for geometry snapshot tests."""
    import geopandas as gpd

    return gpd.read_file(Path(__file__).parent.parent / "fixtures/testing_12x12.gpkg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test leaves open, so the suite never trips
    Matplotlib's more-than-20-open-figures warning."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
