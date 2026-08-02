from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.registry import BackendFilter, backend_registry
from matplotlib.figure import Figure

from gerrytools._ipython import in_jupyter_kernel
from gerrytools.typing import MplKwargs


def _savefig_kwargs(fig: Figure, kwargs: dict[str, object]) -> dict[str, Any]:
    """``Figure.savefig`` kwargs with the gerrytools defaults filled in.

    The cast satisfies the type-checker: Matplotlib accepts a broad dynamic
    kwargs surface here, but static stubs are narrower.
    """
    savefig_kwargs: MplKwargs = dict(kwargs)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    savefig_kwargs.setdefault("dpi", fig.dpi)
    return cast("dict[str, Any]", savefig_kwargs)


def _is_gui_capable_backend(fig: Figure) -> bool:
    """Whether the active backend can open a GUI window for ``fig``.

    Builtin backend names are compared case-insensitively: ``matplotlib.get_backend()``
    may report canonical casing (e.g. ``"TkAgg"``) while the registry lists lowercase
    names. A third-party backend appears in neither builtin list, so it is classified
    by a capability signal instead: a live canvas manager means a window can be raised.
    """
    backend_name = matplotlib.get_backend().lower()
    interactive_names = {
        name.lower() for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    }
    if backend_name in interactive_names:
        return True
    non_interactive_names = {
        name.lower() for name in backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)
    }
    if backend_name in non_interactive_names:
        return False
    return fig.canvas.manager is not None


def show_figure(
    fig: Figure,
    *,
    non_gui_filename: str,
    non_gui_prefix: str,
    **kwargs: object,
) -> None:
    """Display a figure inline in notebooks or via GUI backend in scripts.

    If no GUI backend is available, the figure is written to a unique temporary file and a short
    message is printed. ``non_gui_filename`` supplies only the readable stem and image suffix.

    Args:
        fig (Figure): Matplotlib figure to display.
        non_gui_filename (str): Filepath used when no GUI backend is available.
        non_gui_prefix (str): Prefix used in the printed non-GUI status message.
        **kwargs (object): Additional keyword arguments passed to ``Figure.savefig``.
            Defaults: ``bbox_inches="tight"``, ``dpi=fig.dpi``.
    """
    savefig_kwargs = _savefig_kwargs(fig, dict(kwargs))

    # Notebook: display PNG inline. Display failures fall through to the
    # backend-based paths below, matching the previous broad try/except.
    if in_jupyter_kernel():  # pragma: no cover - only reachable inside a live Jupyter kernel
        try:
            from IPython.display import Image, display

            # The buffer render is always PNG; drop any caller-supplied "format".
            png_kwargs = {key: value for key, value in savefig_kwargs.items() if key != "format"}
            buf = BytesIO()
            fig.savefig(buf, format="png", **png_kwargs)
            buf.seek(0)
            display(Image(data=buf.getvalue()))
            return
        except Exception:
            pass

    if not _is_gui_capable_backend(fig):
        output_path = _save_temporary_figure(fig, non_gui_filename, savefig_kwargs)
        backend_name = matplotlib.get_backend()
        print(f"[{non_gui_prefix}] Non-GUI backend ({backend_name}); saved to {output_path}")
        return

    figure_number = getattr(fig, "number", None)
    if figure_number is None:
        # A GUI backend, but the figure is not pyplot-managed, so pyplot cannot raise a
        # window for it.
        output_path = _save_temporary_figure(fig, non_gui_filename, savefig_kwargs)
        print(f"[{non_gui_prefix}] Figure is not pyplot-managed; saved to {output_path}")
        return

    # Only reachable with a live interactive GUI backend; the test suite runs on Agg.
    plt.figure(figure_number)  # pragma: no cover
    plt.show(block=True)  # pragma: no cover


def _temporary_figure_path(filename: str) -> Path:
    hint = Path(filename)
    suffix = hint.suffix or ".png"
    with NamedTemporaryFile(prefix=f"{hint.stem}-", suffix=suffix, delete=False) as temporary:
        return Path(temporary.name)


def _save_temporary_figure(
    fig: Figure,
    filename: str,
    savefig_kwargs: dict[str, Any],
) -> Path:
    output_path = _temporary_figure_path(filename)
    try:
        fig.savefig(output_path, **savefig_kwargs)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return output_path


def save_figure(fig: Figure, filepath: str, **kwargs: object) -> None:
    """Save a Matplotlib figure with consistent defaults.

    Args:
        fig (Figure): Matplotlib figure to save.
        filepath (str): Output file path.
        **kwargs (object): Additional ``Figure.savefig`` keyword arguments. If omitted,
            defaults are ``bbox_inches="tight"`` and ``dpi=fig.dpi``.
    """
    fig.savefig(filepath, **_savefig_kwargs(fig, dict(kwargs)))
