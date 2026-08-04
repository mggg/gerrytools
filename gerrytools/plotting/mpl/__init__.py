"""Public matplotlib option dataclasses for plotting APIs."""

from gerrytools.plotting.mpl.axis_title_style import AxisLabelStyle, TitleStyle
from gerrytools.plotting.mpl.geoplot_options import ColorbarOptions
from gerrytools.plotting.mpl.label_text_options import (
    LABEL_STYLES,
    FontFamily,
    FontStretch,
    FontStyle,
    FontVariant,
    FontWeight,
    LabelBoxOptions,
    LabelFontOptions,
    LabelStyle,
)
from gerrytools.plotting.mpl.legend_options import LegendAnchor, LegendOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.plotting.mpl.tick_style import TickStyle

__all__ = [
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
