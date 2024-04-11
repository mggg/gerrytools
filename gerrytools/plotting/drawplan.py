import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .colors import districtr
from .districtnumbers import districtnumbers


def drawplan(
    districts,
    assignment,
    overlays=[],
    colors=None,
    numbers=False,
    lw=1 / 2,
    fontsize=15,
    edgecolor="black",
) -> Axes:
    """
    Visualizes the districting plan defined by `assignment`.

    Args:
        districts (GeoDataFrame): Geometries for the districting plan. Assumes
            there is one geometry for each district.
        assignment (str): Column of `districts` which defines the districting plan.
        overlays (list, optional): A list of GeoDataFrames to be plotted over the
            districts.
        colors (str, optional): Column name which specifies colors for each district.
        numbers (bool, optional): If `True`, plots district names (as defined by
            `assignment`) at districts' centroids. Defaults to `False`.
        lw (float, optional): Line thickness if there are more than 20 districts.
        fontsize (float, optional): District-number font size; passed to
            `districtnumbers`.

    Returns:
        A `matplotlib` `Axes` object for the geometries attached to `districts`.
    """
    # Sort districts by their assignment and add a column specifying the color
    # index.
    districts = districts.dissolve(by=assignment).reset_index()
    N = len(districts)
    districts = districts.to_crs("epsg:3857")
    districts[assignment] = districts[assignment].astype(int)
    districts = districts.sort_values(by=assignment)
    if colors is None:
        # Assign colors.
        districts["color"] = districtr(N)

    # Plot the districts.
    base = districts.plot(
        color=districts[colors if colors else "color"],
        edgecolor="black",
        linewidth=lw if lw is not None else 1,
    )

    # If we have overlaid geometries, plot those too.
    if overlays:
        for overlay in overlays:
            overlay = overlay.to_crs(districts.crs)
            overlay.plot(color="None", edgecolor=edgecolor, linewidth=1 / 8, ax=base)

    # If the `numbers` flag is passed, plot the numbers for each district.
    if numbers:
        base = districtnumbers(
            base, districts, assignment=assignment, fontsize=fontsize
        )

    # Turn plot axes off.
    plt.axis("off")

    return base
