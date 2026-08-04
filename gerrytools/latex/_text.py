"""Shared text helpers for LaTeX rendering."""

from __future__ import annotations

_LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}


def latex_escape(text: str) -> str:
    """Escape LaTeX-special characters in plain text.

    Args:
        text (str): Raw text to escape for safe LaTeX rendering.

    Returns:
        str: Escaped text with LaTeX control sequences for special characters.
    """
    return "".join(_LATEX_ESCAPE_MAP.get(ch, ch) for ch in text)
