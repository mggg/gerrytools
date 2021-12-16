
from shapely.geometry import box, MultiPolygon, Point
from random import uniform
from typing import Tuple
import geopandas as gpd
import json
from ..geography import dissolve


class PlotSpecification:
    """
    Specification for redblue plots. Rather than spending extra time re-computing
    bounding boxes, label locations, and manually editing titles, plot specifications
    store those data so plots can be re-created with speed.
    """

    webmercator: str = "epsg:3857"
    """
    WebMercator CRS, with coordinates in feet.
    """

    def __init__(self):
        self.bbox: dict = None
        """
        Plot bounding box; equivalent to matplotlib axis limits. Reported as a
        dictionary. If `None`, the plot's axis limits aren't modified.
        """

        self.context: bool = True
        """
        Asks whether "context" – the units surrounding the desired area to plot –
        should be included. If `True`, the axis limits are modified according to
        the calculated bounding box; if `False`, all other units are dropped and
        only designated units are included.
        """

        self.labels: dict = {}
        """
        Labels and their locations. If this is empty, no labels are plotted.
        """
        
    def computebbox(self, geometries, identifiers, idcolumn="COUNTYFP20", margin=1) -> Tuple:
        r"""
        Finds the minimal rectangle containing the geometries matching the provided
        identifier codes. Typically, the `geometries` are counties and `identifiers`
        are county FIPS codes.

        Args:
            geometries (gpd.GeoDataFrame): Set of geometries.
            identifiers (list): List of identifiers to whose boundaries we
                restrict the viewport.
            idcolumn (str, optional): Column on `geometries` where `identifiers`
                are stored.
            margin (float, optional): Bounding box margin measured in **miles**.
                For example, if the base width of the bounding box is \(w\), then
                the width of the margin-adjusted bounding box is \(w+2m\), where
                \(m\) is the margin; the height of the bounding box is adjusted
                similarly.

        Returns:
            A four-tuple of values: the first pair represents the bottom-left corner
            of the bounding box, the second pair the top-right corner.
        """
        # Set the geometries to the specified CRS, filter the geometries, and
        # dissolve them.
        geometries = geometries.to_crs(self.webmercator)
        geometries["dissolve"] = 1

        subgeometries = geometries[geometries[idcolumn].astype(str).isin(identifiers)]
        dissolved = dissolve(subgeometries, by="dissolve")

        # Find the bounding box and adjust.
        offset = margin*5280
        bounds = dissolved.bounds
        minx, maxx = bounds.minx.values[0], bounds.maxx.values[0]
        miny, maxy = bounds.miny.values[0], bounds.maxy.values[0]

        self.bbox = {
            "x": (minx-offset, maxx+offset),
            "y": (miny-offset, maxy+offset)
        }

        return minx-offset, miny-offset, maxx+offset, maxy+offset

    def computelabels(self, districts, assignment, geometrycolumn="geometry") -> dict:
        """
        Computes label locations for the provided district geometries.

        Args:
            districts (gpd.GeoDataFrame): Districting plan; assumes there is one
                geometry per district.
            assignment (str): Column of `districts` which defines the
                districting plan.
            geometrycolumn (str, optional): Column of `districts` which defines
                the geometry for each district.

        Returns:
            A dictionary mapping district labels to locations.
        """
        # Make sure we're in the same CRS.
        districts = districts.to_crs(self.webmercator)

        # Get the bounding box; if there is no bounding box, we don't need to
        # compute anything, and we simply assign labels to representative points.
        if self.bbox:
            minx, maxx = self.bbox["x"]
            miny, maxy = self.bbox["y"]
            bbox = box(minx, miny, maxx, maxy)

        locations = {}
        
        for geometry, district in zip(districts[geometrycolumn], districts[assignment]):
            # In the first condition, there *is* no bounding box for the plot;
            # set the location of the district's label to be a representative
            # point for the entire geometry.
            if not self.bbox:
                point = geometry.representative_point()
                locations[district] = point.coords[0]
            else:
                # If there *is* a bounding box, we encounter three conditions:
                # either the bounding box entirely contains the district, the
                # bounding box contains a part of the district, or the intersection
                # of the district and the bounding box is empty.

                # If the district is entirely contained, return a representative
                # point.
                if bbox.contains(geometry):
                    point = geometry.representative_point()
                    locations[district] = point.coords[0]

                # If the intersection is nonempty (and is greater than two square
                # miles), return a representative point of the *intersection*.
                elif bbox.intersection(geometry).area > 2*(5280**2):
                    # Find the intersection.
                    intersection = bbox.intersection(geometry)

                    # Check if the intersection is a MultiPolygon (i.e. there are
                    # multiple pieces of the same district), and label the largest
                    # one.
                    if isinstance(intersection, MultiPolygon):
                        largest = max(intersection, key=lambda p: p.area)
                    else:
                        largest = intersection

                    # Compute the representative point.
                    point = largest.representative_point()
                    locations[district] = point.coords[0]

                # Otherwise, continue.
                else: continue

        # Set labels to the `labels` property.
        self.labels = locations
        return locations
