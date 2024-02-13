import json
import math
import pkgutil
from random import choice
from string import hexdigits as hex
from typing import List, Tuple

import seaborn as sns

defaultGray = "#5c676f"
"""
Default gray plotting color; used in histograms, violin plots, and arrows.
"""

citizenBlue = "#4693b3"
"""
Citizen ensemble blue color; used in histograms, violin plots, and arrows. (Aka
Citizen Kane).
"""

overlays = ["gainsboro", "silver", "darkgray", "gray", "dimgrey"]
"""
Overlay colors for choropleth maps.
"""

latex = json.loads(pkgutil.get_data(__name__, "latexcolors.json"))
"""
A dictionary of nice LaTeX colors, [borrowed from here.](http://latexcolor.com/)
"""


def hexshift(color) -> str:
    """
    Randomly modifies the provided hexadecimal color.

    Args:
        color (str): A hexadecimal color string; e.g. `"#FFFF00"`.

    Returns:
        A hexadecimal color string.
    """
    # Choose a hexidecimal digit, first paring down the digits we'll use.
    h = hex.upper()[:-6]
    sub = choice(h)
    char = choice(color[1:])

    # Find the character we're going to replace that's *not* the same character
    # as the one we got from the hexadecimal string.
    while sub == char:
        sub = choice(h)

    # Return the subbed string.
    return color.replace(char, sub)


def districtr(N):
    colors = [
        "#0099cd",
        "#ffca5d",
        "#00cd99",
        "#99cd00",
        "#cd0099",
        "#9900cd",
        "#8dd3c7",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
        "#ffffb3",
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#b15928",
        "#64ffda",
        "#00B8D4",
        "#A1887F",
        "#76FF03",
        "#DCE775",
        "#B388FF",
        "#FF80AB",
        "#D81B60",
        "#26A69A",
        "#FFEA00",
        "#6200EA",
    ]

    repeats = math.ceil(N / len(colors))
    tail = [hexshift(c) for c in colors * (repeats - 1)]
    return (colors + (tail if tail else []))[:N]


def redbluecmap(n) -> List[Tuple]:
    """
    Generates a red/white/blue color palette in `n` colors with white at the
    `mid` th index.

    Args:
        n (int): The number of colors to generate.

    Returns:
        List of RGB tuples.
    """
    midpoint = math.ceil(n / 2)

    # To get the appropriately-toned blues and reds, we create a list of colors,
    # then select the first section of each color.
    blues = list(sns.color_palette("coolwarm", as_cmap=False, n_colors=n + 2))[
        :midpoint
    ]
    reds = list(sns.color_palette("coolwarm", as_cmap=False, n_colors=n + 2))[
        -midpoint:
    ]

    return list(reversed(reds)) + list(reversed(blues))


def flare(n) -> list:
    """
    Returns a list of colors based on the `flare` Matplotlib/seaborn colormap.

    Args:
        n (int): Number of colors to generate.

    Returns:
        List of RGB triples.
    """
    return list(sns.color_palette("flare", as_cmap=False, n_colors=n))


def purples(n) -> list:
    """
    Returns a list of colors based on the `Purples` Matplotlib/seaborn colormap.

    Args:
        n (int): Number of colors to generate.

    Returns:
        List of RGB triples.
    """
    return list(sns.color_palette("Purples", as_cmap=False, n_colors=n))


def greens(n) -> list:
    """
    Returns a list of colors based on the `Greens` Matplotlib/seaborn colormap.

    Args:
        n (int): Number of colors to generate.

    Returns:
        List of RGB triples.
    """
    return list(sns.color_palette("Greens", as_cmap=False, n_colors=n))
