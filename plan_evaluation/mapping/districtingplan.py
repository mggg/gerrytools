
import geopandas as gpd
import matplotlib.pyplot as plt
import math
from ..colors import districtr

def districtingplan(
        districts:gpd.GeoDataFrame, assignment:str, overlay=None, colors=None,
        numbers=False
    ) -> plt.Axes:
    """
    Visualizes the districting plan defined by `assignment`.

    :param districts: Geometries for the districting plan. Assumes there is one
    geometry for each district.
    :param assignment: Column of `districts` which defines the districting plan.
    :param overlay: Optional; geodataframe to be plotted over the districts.
    Often is a gdf of counties.
    :param colors: Optional; column name which specifies colors for each district.
    :param numbers: Optional; if true, plots district names (as defined by
    `assignment`) at districts' centroids.
    :returns: Matplotlib Axes object.
    """
    # Sort districts by their assignment and add a column specifying the color
    # index.
    N = len(districts)
    districts = districts.to_crs("epsg:3857")
    districts[assignment] = districts[assignment].astype(int)
    districts = districts.sort_values(by=assignment)
    districts["colorindex"] = list(range(N))

    # Assign colors.
    repeats = math.ceil(N/len(districtr))
    repeatedcolors = (districtr*repeats)[:N]
    districts["color"] = repeatedcolors

    # Plot the districts.
    base = districts.plot(
        color=districts[colors if colors else "color"],
        edgecolor="black",
        linewidth=1 if N<=20 else 1/2
    )

    # If we have overlaid geometries, plot those too.
    if overlay is not None:
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
