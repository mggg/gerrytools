from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase

from gerrytools.plotting.mpl.legend_options import LegendOptions
from gerrytools.typing import LegendHandle, MplKwargs


def save_legend_handles(
    *,
    handles: Sequence[LegendHandle],
    legend_options: LegendOptions,
    filepath: str,
    outer_padding: float = 0.07,
    dpi: int | float = 300,
    **legend_kwargs: object,
) -> None:
    """Save legend handles to a standalone image.

    Args:
        handles (Sequence[LegendHandle]): Handles to render in the legend.
        legend_options (LegendOptions): Base legend options dataclass.
        filepath (str): Output image path.
        outer_padding (float, optional): Fractional expansion applied to the legend bounding box.
            Defaults to ``0.07``.
        dpi (int | float, optional): Save resolution. Defaults to ``300``.
        **legend_kwargs (object): Additional keyword arguments merged into the legend options.

    Raises:
        ValueError: If ``handles`` is empty.
    """
    if len(handles) == 0:
        raise ValueError("No legend handles to save.")

    legend_fig = plt.figure(dpi=dpi)
    try:
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")

        opts: MplKwargs = legend_options.to_dict() | dict(legend_kwargs)
        leg = legend_ax.legend(handles=handles, **cast("Any", opts))

        legend_fig.subplots_adjust(0, 0, 1, 1)

        canvas = legend_fig.canvas
        canvas.draw()
        # get_renderer is backend-specific (e.g. Agg), so it is absent from FigureCanvasBase.
        get_renderer_fn: Callable[[], RendererBase] | None = getattr(canvas, "get_renderer", None)
        if not callable(
            get_renderer_fn
        ):  # pragma: no cover - defensive guard for non-standard Matplotlib backends that omit get_renderer()
            raise RuntimeError("Matplotlib canvas does not expose get_renderer().")
        renderer = get_renderer_fn()

        bbox = leg.get_window_extent(renderer=renderer)
        bbox_inches = bbox.transformed(legend_fig.dpi_scale_trans.inverted())
        bbox_inches = bbox_inches.expanded(1.0 + outer_padding, 1.0 + outer_padding)

        legend_fig.savefig(
            filepath,
            bbox_inches=bbox_inches,
            pad_inches=0.0,
        )
    finally:
        plt.close(legend_fig)
