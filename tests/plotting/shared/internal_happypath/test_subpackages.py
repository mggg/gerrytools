import inspect

from gerrytools import plotting
from gerrytools.plotting import data, geometry, mpl, other, plan

EXPECTED_PLOTTING_EXPORTS = [
    "ArrowPlacement",
    "ArrowTextStyle",
    "BandOptions",
    "BarPlot",
    "BarPlotOptions",
    "BoxPlot",
    "BoxPlotOptions",
    "BoxPlotStats",
    "ColormapLayer",
    "ColorbarOptions",
    "data",
    "geometry",
    "GeoPlot",
    "DotDensityPlot",
    "Histogram",
    "HistogramOptions",
    "LabelArrowOptions",
    "LabelArrowStyle",
    "LABEL_STYLES",
    "LabelBoxOptions",
    "LabelFontOptions",
    "LabelOptions",
    "LabelStyle",
    "LegendOptions",
    "LineOptions",
    "mpl",
    "other",
    "PaintballPlot",
    "plan",
    "PointMarkerOptions",
    "ScatterPlot",
    "SeaLevelPlot",
    "SeaLevelLineOptions",
    "SeatsVotesPlot",
    "SeatsVotesLineOptions",
    "SeatsVotesMarkerOptions",
    "SubwaySignOptions",
    "TextArrowOptions",
    "TextArrowStyle",
    "UNSET",
    "Unset",
    "ViolinPlot",
    "ViolinPlotOptions",
    "draw_graph",
    "draw_graph_components",
    "subway_signs",
]


def test_public_plotting_subpackages_are_visible():
    assert plotting.data is data
    assert plotting.geometry is geometry
    assert plotting.mpl is mpl
    assert plotting.other is other
    assert plotting.plan is plan


def test_public_export_surfaces_are_frozen():
    assert plotting.__all__ == EXPECTED_PLOTTING_EXPORTS
    assert data.__all__ == [
        "ArrowPlacement",
        "ArrowTextStyle",
        "BandOptions",
        "BarPlot",
        "BarPlotOptions",
        "BoxPlot",
        "BoxPlotOptions",
        "BoxPlotStats",
        "Histogram",
        "HistogramOptions",
        "LabelArrowOptions",
        "LabelArrowStyle",
        "LineOptions",
        "PaintballPlot",
        "ScatterPlot",
        "SeaLevelPlot",
        "SeaLevelLineOptions",
        "SeatsVotesPlot",
        "SeatsVotesLineOptions",
        "SeatsVotesMarkerOptions",
        "TextArrowOptions",
        "TextArrowStyle",
        "UNSET",
        "Unset",
        "ViolinPlot",
        "ViolinPlotOptions",
    ]
    assert geometry.__all__ == ["ColormapLayer", "DotDensityPlot", "GeoPlot", "LabelOptions"]
    assert mpl.__all__ == [
        "AxisLabelStyle",
        "ColorbarOptions",
        "FontFamily",
        "FontStretch",
        "FontStyle",
        "FontVariant",
        "FontWeight",
        "LABEL_STYLES",
        "LabelBoxOptions",
        "LabelFontOptions",
        "LabelStyle",
        "LegendAnchor",
        "LegendOptions",
        "PointMarkerOptions",
        "TickStyle",
        "TitleStyle",
    ]
    assert other.__all__ == ["SubwaySignOptions", "subway_signs"]
    assert plan.__all__ == ["draw_graph", "draw_graph_components"]


def test_internal_data_containers_are_not_exported():
    internal_names = {
        "ArrowData",
        "BandData",
        "BarSetData",
        "BoxPlotSetData",
        "HistogramData",
        "LineData",
        "PointSetData",
        "ScatterData",
        "SeaLevelSetData",
        "SeatsVotesData",
        "ViolinPlotSetData",
    }
    assert internal_names.isdisjoint(plotting.__all__)
    assert internal_names.isdisjoint(data.__all__)


def test_color_helpers_are_owned_by_colors_module():
    assert {"districtr", "flare", "latex", "purples", "redbluecmap"}.isdisjoint(plotting.__all__)


def test_public_mpl_options_imports():
    _ = mpl.PointMarkerOptions
    _ = mpl.TickStyle
    _ = mpl.LegendOptions
    _ = mpl.AxisLabelStyle
    _ = mpl.TitleStyle
    _ = mpl.LabelFontOptions
    _ = mpl.LabelBoxOptions
    _ = mpl.ColorbarOptions


def test_public_data_and_geometry_plot_imports():
    _ = data.PaintballPlot
    _ = data.SeatsVotesPlot
    _ = data.Histogram
    _ = data.BoxPlot
    _ = data.ViolinPlot
    _ = geometry.GeoPlot
    _ = geometry.DotDensityPlot


def test_rewritten_signatures_and_old_names_are_absent():
    choropleth = inspect.signature(geometry.GeoPlot.add_choropleth_layer).parameters
    density = inspect.signature(geometry.DotDensityPlot.add_density_layer).parameters

    assert choropleth["column"].default is inspect.Parameter.empty
    assert choropleth["geo_source"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density["refresh_cache"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "n_jobs" in density
    assert "n_cores" not in density

    assert not hasattr(data, "PaintBall")
    assert not hasattr(data, "SeaLevel")
    assert not hasattr(data, "SeatsVotes")
    assert not hasattr(data.Histogram, "add_histogram")

    for plot_class in (
        data.BarPlot,
        data.BoxPlot,
        data.Histogram,
        data.PaintballPlot,
        data.ScatterPlot,
        data.SeaLevelPlot,
        data.SeatsVotesPlot,
        data.ViolinPlot,
    ):
        assert (
            inspect.signature(plot_class).parameters["figure_size"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )


def test_boolean_display_toggles_support_both_states():
    for plot_class in (data.BarPlot, data.BoxPlot, data.ViolinPlot):
        plot = plot_class()
        plot.display_group_separators(True)
        assert plot._include_group_vlines is True
        plot.display_group_separators(False)
        assert plot._include_group_vlines is False

    for plot_class in (data.Histogram, data.SeaLevelPlot):
        plot = plot_class()
        plot.display_grid(True)
        assert plot.grid is True
        plot.display_grid(False)
        assert plot.grid is False

    histogram = data.Histogram()
    histogram.display_warnings(False)
    assert histogram._show_warnings is False
    histogram.display_warnings(True)
    assert histogram._show_warnings is True
