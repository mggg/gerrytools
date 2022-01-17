import pandas as pd
import geopandas as gpd
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from math import ceil
import glob

overlay_colors = ["gainsboro", "silver", "darkgray", "gray", "dimgrey"]

def drawdemographics(state_shape, districts=None, assignment=None, demographic="BVAP", share=True,
                     overlay=[], cmap="Purples", colorbar_title="BVAP", numbers=True) -> Axes:
    """
    Visualization of population shares or totals in a state's map.
    
    Args:
        state_shape (GeoDataFrame): Geometries for the state. Population shares or totals will be drawn at this level. i.e.
            block or VTD level. 
        districts (GeoDataFrame, optional): Geometries for the districting plan. Assumes one geometry per district. 
        assignment (str, optional): Required argument when `districts` are provided. Column of `districts` which defines
            the districing plan. 
        demographic (str, optional): The string representing the demographic to be shown on the map. The string should specift
            a column in the state shapefile. Note that if a demographic share is desired for plotting, the share should be 
            added to the GeoDataFrame prior to calling this function. 
        overlay (list, optional): This should be a list of GeoDataFrames desired to be overlaid on the map. Some options would
            include overlaying district assignments, blocks, VTDs, or counties. Within the list, item [0] will be overlaid in the
            lightest color, and item[-1] will be overlaid in the darkest color.
        cmap (string, optional): Defines which colormap to use. Defaults to matplotlib's `Purples` colormap.
        colorbar_title(string, optional): Title for the colorbar. Defaults to demographic column.
        numbers (bool, optional): If `True`, plot district names (as defined by `assignment`) at districts' centroids. May only 
            be `True` when districts are provided.
        
    
    """
    fig, base = plt.subplots(1, 1, figsize=(10,10))
    
    state_shape.plot(
        column=demographic, 
        cmap=cmap, 
        edgecolor="lightgray", 
        linewidth = 1/8,
        ax=base)

    if districts is not None:
      districts.boundary.plot( 
         edgecolor="black",
         linewidth=3/4, 
         ax=base)
    
    state_shape_dissolve = state_shape.dissolve()
        
    colorbar_ticks = []
    if colorbar_title is None:
      colorbar_title = demographic
    
    if max(state_shape[demographic]) <= 1: 
        colorbar_ticks = np.linspace(0, 1, 10)
        colorbar_labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
        
        bounds = colorbar_ticks * 100 
    else:
        colorbar_center = ceil(np.mean(state_shape[demographic][state_shape[demographic] > 0])/10)*10
        colorbar_ticks = [i * 10 + 1 if i > 0 else i for i in range(int(colorbar_center/10))]
        max_pop = max(state_shape[demographic])
        if max_pop > 500: 
            colorbar_ticks.extend([i*100 + 1 for i in range(round(colorbar_center/100)+1, 6)])
        elif max_pop > 200: 
            colorbar_ticks.extend([i*100 + 1 for i in range(round(colorbar_center/100)+1, 3)])
        else:
            colorbar_ticks.extend([i*100 + 1 for i in range(int(round(max_pop,-1)/10))])
        colorbar_ticks.append(max_pop)
        
        colorbar_ticks = np.array(colorbar_ticks)
        colorbar_labels = [f"{colorbar_ticks[i]}-{colorbar_ticks[i+1]-1}" for i in range(len(colorbar_ticks)-2)]
        colorbar_labels.extend([f"{colorbar_ticks[-2]}+", ""])
        bounds = colorbar_ticks
    colorbarmap = plt.cm.get_cmap(cmap, len(colorbar_ticks))
    norm = mpl.colors.BoundaryNorm(bounds, colorbarmap.N)
     

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colorbarmap, norm=norm), shrink = 0.5, location="right",  ax=base, ticks=bounds)
    cbar.ax.set_yticklabels(colorbar_labels)
    cbar.ax.set_ylabel(colorbar_title)
    cbar.ax.yaxis.set_label_position("left")
    
    for idx, geom in enumerate(overlay):
        geom = geom.to_crs(state_shape.crs)
        geom.boundary.plot(edgecolor=overlay_colors[idx], linewidth = 1/(len(overlay)+1)*(idx+1), ax =base)
        
    if numbers: 
        for district, identifier in zip(districts["geometry"], districts[assignment]):
            x,y = list(district.centroid.coords)[0]
            base.annotate(
                identifier, (x,y), xytext=(x,y), xycoords="data", fontsize=6, 
                ha="center", va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="wheat", ec="black", alpha=1, 
                    linewidth=1/6
                )
            )
    
    state_shape_dissolve.boundary.plot(color="black", ax=base)

    base.set_axis_off()
    return base

