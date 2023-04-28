from matplotlib.axes import Axes

# If district numbers are to be plotted, plot those too!


def districtnumbers(
    base,
    districts,
    assignment="DISTRICTN",
    boxstyle="circle,pad=0.2",
    fc="wheat",
    ec="black",
    lw=1 / 6,
    fontsize=15,
) -> Axes:
    """
    Plots district numbers on top of overlaid district geometries.

    TODO: change (x,y) coordinate pairs to representative points rather than
    centroids.

    Args:
        base (Axes): Base `Axes` object for the plot.
        districts (GeoDataFrame): Geometries for the districting plan. Assumes
            there is one geometry for each district.
        assignment (str, optional): Column of `districts` which defines the
            districting plan.
        boxstyle (str, optional): Sets the box style for the district number
            markers. Defaults to circles with 0.2pt padding.
        fc (str, optional): District marker face color. Defaults to `"wheat"`.
        ec (str, optional): District marker edge color. Defaults to `"black"`.
        lw (float, optional): District marker edge width. Defaults to 1/6pt.
        fontsize (float, optional): District marker font size. Defaults to 15pt.

    Returns:
        Base axes object.
    """
    for district, identifier in zip(districts["geometry"], districts[assignment]):
        x, y = list(district.representative_point().coords)[0]
        base.annotate(
            identifier,
            (x, y),
            xytext=(x, y),
            xycoords="data",
            fontsize=fontsize,
            ha="center",
            va="center",
            bbox=dict(boxstyle=boxstyle, fc=fc, ec=ec, alpha=1, linewidth=lw),
        )

    return base
