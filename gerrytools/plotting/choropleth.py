
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from .colors import overlays as overlaycolors
from .districtnumbers import districtnumbers


def choropleth(
    geometries, districts=None, assignment=None, demographic="BVAP",
    overlays=[], cmap="Purples", cbartitle=None, numbers=True, lw=1 / 8,
    fontsize=15, min=0, max=1, interval=1 / 10, colorbar=True
) -> Axes:
    r"""
    Visualization of population shares or totals in a state's map.

    Args:
        geometries (GeoDataFrame): Base geometries for the state. Population
            shares or totals will be drawn at this level (i.e. statistics are
            reported at this base geometric level).
        districts (GeoDataFrame, optional): Geometries for the districting plan.
            Assumes one geometry per district.
        assignment (str, optional): Required argument when `districts` are
            provided. Column of `districts` which defines the districing plan.
        demographic (str, optional): The string representing the demographic to
            be shown on the map. The string should specify a column in `geometries`.
            *This column must contain values in \\([0,1]\\).*
        overlays (list, optional): A list of GeoDataFrames desired to be overlaid on
            the map. Some options would include overlaying district assignments,
            blocks, VTDs, or counties. The first set of geometries in the list
            will be overlaid in the lightest color, and last will be overlaid in
            the darkest color.
            cmap (string/ListedColorMap, optional): Defines which colormap to use.
            Defaults to matplotlib's `Purples` colormap. Can be a string which
            specifies a named matplotlib colormap or a `ListedColormap` with the
            appropriate number of bins; by default, this is 10.
        cbartitle (string, optional): Title for the colorbar. Defaults to
            `demographic`.
        numbers (bool, optional): If `True`, plot district names (as defined by
            `assignment`) at districts' centroids. May only be `True` when
            `districts` is not `None`.
        lw (float, optional): The base geometries' line widths.
        min (float, optional): The lower limit of the data points; defaults to 0.
        max (float, optional): The upper limit of the data points; defaults to 1.
        interval (float, optional): The width of the interval; a bin.
        colorbar (bool, optional): Do we include the color bar?

    Returns:
        A matplotlib `Axes` object visualizing a choropleth map with the provided
        overlays.
    """
    # Get the figure and base axis sizes.
    fig, base = plt.subplots(1, 1, figsize=(10, 10))

    # Get the title for the colorbar.
    if cbartitle is None:
        cbartitle = demographic

    # Set minimum and maximum values.
    boundaries = np.arange(min, max + interval, interval)
    ticks = np.arange(min, max + interval, interval / 2)[:-1]
    labels = [
        f"{int(ticks[i-1]*100)} — {int(ticks[i+1]*100)}%" if i % 2 else ""
        for i in range(len(ticks))
    ]

    # Set the color map, based on user preference or default (purples). If we
    # can't find the correct colormap, then create our own.
    if isinstance(cmap, str):
        colorbarmap = plt.cm.get_cmap(cmap, len(boundaries))
    else:
        colorbarmap = cmap

    norm = mpl.colors.BoundaryNorm(boundaries, colorbarmap.N)

    # Plot geometries!
    geometries.plot(
        column=demographic,
        cmap=cmap,
        edgecolor="lightgray",
        linewidth=lw,
        vmin=min,
        vmax=max,
        ax=base
    )

    # Create and plot the colorbar on the right side of the figure.
    if colorbar:
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=colorbarmap, norm=norm), shrink=0.5,
            location="right", ax=base, ticks=ticks
        )
        cbar.ax.set_yticklabels(labels)
        cbar.ax.set_ylabel(cbartitle)
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.tick_params(size=0)

    # Plot each of the overlays, adjusting CRSes and applying colors as we go.
    for idx, geom in enumerate(overlays):
        geom = geom.to_crs(geometries.crs)
        geom.boundary.plot(edgecolor=overlaycolors[-(idx + 1)], linewidth=1 / 4, ax=base)

    # If district geometries are provided, plot them as well.
    if districts is not None:
        districts.plot(
            edgecolor="black",
            linewidth=3 / 2,
            ax=base,
            color="None"
        )

    # If district numbers are to be plotted, plot those too!
    if numbers:
        base = districtnumbers(base, districts, assignment=assignment, fontsize=fontsize)

    # Turn plot vertical/horizontal axes off and return base Axes.
    base.set_axis_off()

    return base, None if not colorbar else cbar
