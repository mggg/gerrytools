
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from math import ceil

from .colors import overlays as overlaycolors
from .districtnumbers import districtnumbers


def choropleth(
        geometries, districts=None, assignment=None, demographic="BVAP",
        overlays=[], cmap="Purples", cbartitle=None, numbers=True, lw=1/8
    ) -> Axes:
    """
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
            Note that if a demographic share is desired for plotting, the share
            should be added to the GeoDataFrame prior to calling this function. 
        overlays (list, optional): A list of GeoDataFrames desired to be overlaid on
            the map. Some options would include overlaying district assignments,
            blocks, VTDs, or counties. The first set of geometries in the list
            will be overlaid in the lightest color, and last will be overlaid in
            the darkest color.
	    cmap (string, optional): Defines which colormap to use. Defaults to
            matplotlib's `Purples` colormap.
        cbartitle (string, optional): Title for the colorbar. Defaults to
            `demographic`.
        numbers (bool, optional): If `True`, plot district names (as defined by
            `assignment`) at districts' centroids. May only be `True` when
            `districts` is not `None`.
        lw (float, optional): The base geometries' line widths.

    Returns:
        A matplotlib `Axes` object visualizing a choropleth map with the provided
        overlays.
    """
    # Get the figure and base axis sizes.
    fig, base = plt.subplots(1, 1, figsize=(10,10))
    
    # Dissolve the geometries to get the boundary of their union.
    boundary = geometries.dissolve()
    
    # Get the title for the colorbar.
    if cbartitle is None: cbartitle = demographic
    
    # Create an empty list to store ticks; if it's a proportion being plotted,
    # automatically create ticks and percentage labels for the color bar.
    colorbar_ticks = []
    if max(geometries[demographic]) <= 1: 
        colorbar_ticks = np.linspace(0, 1, 11)
        colorbar_labels = [f"{tick*100}%" for tick in colorbar_ticks]
        vmin = 0
        vmax = 1        
        bounds = colorbar_ticks * 100 
    else:
        # Decide the tick interval based on the average population (over geometries
        # with nonzero population).
        colorbar_center = ceil(np.mean(geometries[demographic][geometries[demographic] > 0])/10)*10
        colorbar_ticks = [i * 10 + 1 if i > 0 else i for i in range(int(colorbar_center/10))]
        max_pop = max(geometries[demographic])

        # Based on population conditions, decides the number of ticks.
        if max_pop > 200 and colorbar_center < 100: 
            colorbar_ticks.extend([i*100 + 1 for i in range(1, 3)])
        elif max_pop > 500 and colorbar_center > 100: 
            colorbar_ticks.extend([i*100 + 1 for i in range(round(colorbar_center/100)+1, 6)])
        else:
            colorbar_ticks.extend([i*100 + 1 for i in range(int(round(max_pop,-1)/10))])

        # Extend the color bar to capture the maximum population value.
        colorbar_ticks.append(max_pop)    
        colorbar_ticks = np.array(colorbar_ticks)

        # Create labels based on the population values.
        colorbar_labels = [f"{colorbar_ticks[i]}-{colorbar_ticks[i+1]}" for i in range(len(colorbar_ticks)-2)]
        colorbar_labels.extend([f"{colorbar_ticks[-2]}+", ""])

        # Set the plot bounds based on population values and set the normalizer
        # for the color map.
        bounds = colorbar_ticks
        vmin = 0 
        vmax = max_pop
    
    # Set the color map, based on user preference or default (purples).
    colorbarmap = plt.cm.get_cmap(cmap, len(colorbar_ticks))
    norm = mpl.colors.BoundaryNorm(bounds, colorbarmap.N)
        
    # Plot geometries!
    geometries.plot(
        column=demographic, 
        cmap=cmap,
        edgecolor="lightgray", 
        linewidth=lw,
        vmin=vmin, 
        vmax=vmax,
        ax=base
    )

    # Create and plot the colorbar on the right side of the figure.
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=colorbarmap, norm=norm), shrink=0.5,
        location="right",  ax=base, ticks=bounds
    )
    cbar.ax.set_yticklabels(colorbar_labels)
    cbar.ax.set_ylabel(cbartitle)
    cbar.ax.yaxis.set_label_position("left")
    
    # Plot each of the overlays, adjusting CRSes and applying colors as we go.
    for idx, geom in enumerate(overlays):
        geom = geom.to_crs(geometries.crs)
        geom.boundary.plot(edgecolor=overlaycolors[-(idx+1)], linewidth = 1/4, ax =base)
    
    # If district geometries are provided, plot them as well.
    if districts is not None:
        districts.boundary.plot( 
            edgecolor="black",
            linewidth=3/2, 
            ax=base
        )

    # If district numbers are to be plotted, plot those too!
    if numbers: base = districtnumbers( base, districts, assignment=assignment)
    
    # Plot the boundary and set its color to black.
    boundary.boundary.plot(color="black", ax=base)

    # Turn plot vertical/horizontal axes off and return base Axes.
    base.set_axis_off()
    return base
