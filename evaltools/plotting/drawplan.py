
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .colors import districtr, redbluecmap

def drawplan(
        districts, assignment, overlays=None, colors=None, numbers=False, lw=1/2
    ) -> Axes:
    """
    Visualizes the districting plan defined by `assignment`.

    Args:
        districts (GeoDataFrame): Geometries for the districting plan. Assumes
            there is one geometry for each district.
        assignment (str): Column of `districts` which defines the districting plan.
        overlay (GeoDataFrame, optional): GeoDataFrame to be plotted over the
            districts. Often is a gdf of counties.
        colors (str, optional): Column name which specifies colors for each district.
        numbers (bool, optional): If `True`, plots district names (as defined by
            `assignment`) at districts' centroids. Defaults to `False`.
        lw (float, optional): Line thickness if there are more than 20 districts.
    
    Returns:
        A `matplotlib` `Axes` object for the geometries attached to `districts`.
    """
    # Sort districts by their assignment and add a column specifying the color
    # index.
    N = len(districts)
    districts = districts.to_crs("epsg:3857")
    districts[assignment] = districts[assignment].astype(int)
    districts = districts.sort_values(by=assignment)
    districts["colorindex"] = list(range(N))

    # Assign colors.
    districts["color"] = districtr(N)

    # Plot the districts.
    base = districts.plot(
        color=districts[colors if colors else "color"],
        edgecolor="black",
        linewidth=1 if N<=20 else lw
    )

    # If we have overlaid geometries, plot those too.
    if overlays:
        for overlay in overlays:
            overlay = overlay.to_crs(districts.crs)
            overlay.plot(color="None", edgecolor="black", linewidth=1/8, ax=base)
    
    # If the `numbers` flag is passed, plot the numbers for each district.
    if numbers:
        for district, identifier in zip(districts["geometry"], districts[assignment]):
            x, y = list(district.centroid.coords)[0]
            base.annotate(
                identifier, (x, y), xytext=(x,y), xycoords="data", fontsize=6,
                ha="center", va="center",
                bbox=dict(
                    boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=1
                )
            )

    # Turn plot axes off.
    plt.axis("off")

    return base
