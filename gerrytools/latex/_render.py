"""Compilation and preview backends for :class:`~gerrytools.latex.document.TexDocument`.

Everything that shells out lives here: TeX-engine compilation, PDF-to-PNG rasterization, and the
Qt/Jupyter preview windows. ``TexDocument`` keeps thin ``preview``/``save_*`` methods that
delegate to these functions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from gerrytools._ipython import in_jupyter_kernel as _in_jupyter_kernel

_SUBPROCESS_TIMEOUT_SECONDS = 120


def _which_any(names: Iterable[str]) -> Optional[str]:  # pragma: no cover
    """Checks for the first available executable in the provided list.

    Args:
        names (Iterable[str]): List of executable names to check.

    Returns:
        Optional[str]: The name of the first found executable, or None if none are found.
    """
    for n in names:
        if shutil.which(n):
            return n
    return None


def _compile_pdf(
    tex_source: str,
    *,
    tex_path: Path,
    pdf_path: Path,
    workdir: Path,
    engine_preference_order: Iterable[str],
    compile_passes: int,
    preferred_engine: Optional[str] = None,
) -> None:
    """Compile LaTeX source to a PDF in the working directory.

    Args:
        tex_source (str): Complete standalone LaTeX document source.
        tex_path (Path): Path the source is written to before compilation.
        pdf_path (Path): Path the engine writes the PDF to.
        workdir (Path): Output directory passed to the engine.
        engine_preference_order (Iterable[str]): Engines to try, in order, when
            ``preferred_engine`` is None.
        compile_passes (int): Number of passes for aux-file-dependent packages (e.g. nicematrix).
        preferred_engine (Optional[str], optional): Preferred TeX engine name. Defaults to None.

    Returns:
        None

    Raises:
        RuntimeError: If no TeX engine is found or LaTeX compilation fails.
    """
    engine_candidates = list(engine_preference_order)
    if preferred_engine is None:
        engine = _which_any(engine_candidates)
    else:  # pragma: no cover
        engine = _which_any([preferred_engine])

    if engine is None and preferred_engine is not None:  # pragma: no cover
        raise RuntimeError(f"TeX engine {preferred_engine} not found.")
    elif engine is None:  # pragma: no cover
        raise RuntimeError(
            f"No TeX engine found. Please install one of: [{', '.join(engine_candidates)}]"
        )

    tex_path.write_text(tex_source, encoding="utf-8")

    if engine == "tectonic":  # pragma: no cover
        cmd = [engine, str(tex_path), "--outdir", str(workdir)]
    else:
        cmd = [
            engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-output-directory",
            str(workdir),
            str(tex_path),
        ]

    # tectonic reruns itself until stable; other engines need explicit extra passes for aux-
    # file-dependent packages (e.g. nicematrix).
    passes = 1 if engine == "tectonic" else max(1, int(compile_passes))
    for _ in range(passes):
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 or not pdf_path.exists():
            raise RuntimeError(f"LaTeX compile failed with {engine}.\n\nLOG:\n{log}")


def _render_pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 250) -> None:
    """Render the first page of a PDF to a PNG without PyMuPDF.

    Preference order:
      1) pdftocairo (poppler)
      2) pdftoppm   (poppler)
      3) gs         (ghostscript)
      4) magick/convert (imagemagick)  [least preferred]

    Args:
        pdf_path (Path): Input PDF path.
        png_path (Path): Output PNG path.
        dpi (int, optional): Render resolution in dots-per-inch. Defaults to ``250``.

    Returns:
        None

    Raises:
        RuntimeError: If no supported renderer is available or rendering fails.
    """
    renderer = _which_any(["pdftocairo", "pdftoppm", "gs", "magick", "convert"])
    if renderer is None:
        raise RuntimeError(
            "No PDF renderer found. Install one of:\n"
            "  - poppler-utils (pdftocairo / pdftoppm)\n"
            "  - ghostscript (gs)\n"
            "  - imagemagick (magick/convert)\n"
        )

    out_base = png_path.with_suffix("")  # e.g. /tmp/xyz -> renderer appends .png

    if renderer == "pdftocairo":
        # Produces exactly <out_base>.png
        cmd = [
            "pdftocairo",
            "-png",
            "-r",
            str(dpi),
            "-f",
            "1",
            "-l",
            "1",
            "-singlefile",
            str(pdf_path),
            str(out_base),
        ]

    elif renderer == "pdftoppm":
        # Produces exactly <out_base>.png with -singlefile
        cmd = [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            "-f",
            "1",
            "-singlefile",
            str(pdf_path),
            str(out_base),
        ]

    elif renderer == "gs":
        # Safe-ish ghostscript invocation: first page only, transparent background
        cmd = [
            "gs",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pngalpha",
            f"-r{dpi}",
            "-dFirstPage=1",
            "-dLastPage=1",
            f"-sOutputFile={str(png_path)}",
            str(pdf_path),
        ]

    else:  # imagemagick convert/magick
        # Note: some distros lock down PDF conversion in ImageMagick policy.xml
        cmd = [
            renderer,
            "-density",
            str(dpi),
            f"{str(pdf_path)}[0]",
            "-quality",
            "100",
            str(png_path),
        ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError(f"PDF->PNG render failed using {renderer}.\n\nLOG:\n{log}")

    # Poppler outputs <out_base>.png; ensure final path exists where we expect.
    if renderer in {"pdftocairo", "pdftoppm"}:
        produced = out_base.with_suffix(".png")
        if produced != png_path:
            png_path.unlink(missing_ok=True)
            produced.replace(png_path)

    if not png_path.exists():
        raise RuntimeError(f"PDF->PNG renderer reported success but {png_path} not found.")


def _show_png_jupyter(png_path: Path) -> None:  # pragma: no cover
    """Displays a rendered PNG in a Jupyter notebook."""
    from IPython.display import Image, display

    display(Image(filename=str(png_path)))


def _show_png_qt(
    png_path: Path,
    *,
    title: str = "LaTeX Preview",
    max_size: tuple[int, int] = (1200, 800),
) -> None:  # pragma: no cover
    """Display a rendered PNG in a Qt window.

    Args:
        png_path (Path): Rendered PNG to display.
        title (str, optional): Window title text. Defaults to ``"LaTeX Preview"``.
        max_size (tuple[int, int], optional): Maximum ``(width, height)`` in pixels for
            the preview window. Defaults to ``(1200, 800)``.

    Returns:
        None

    Raises:
        RuntimeError: If ``PyQt6`` is unavailable or the PNG cannot be loaded.
    """
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyQt6 is required for non-Jupyter preview. Install PyQt6 or use save_png/save_pdf."
        ) from exc

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    pix = QtGui.QPixmap(str(png_path))
    if pix.isNull():
        raise RuntimeError("Failed to load PNG into QPixmap.")

    max_w, max_h = max_size
    scale = min(1.0, max_w / max(1, pix.width()), max_h / max(1, pix.height()))
    shown = (
        pix
        if scale >= 1.0
        else pix.scaled(
            int(pix.width() * scale),
            int(pix.height() * scale),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
    )

    label = QtWidgets.QLabel()
    label.setPixmap(shown)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(label)
    scroll.setWidgetResizable(True)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(title)
    win.setCentralWidget(scroll)
    win.resize(min(max_w, shown.width() + 30), min(max_h, shown.height() + 50))
    win.show()

    # In notebooks, calling app.exec() can hang; only do it when NOT in Jupyter.
    if not _in_jupyter_kernel():
        app.exec()
