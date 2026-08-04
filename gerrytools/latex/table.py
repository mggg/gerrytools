"""Plain-``tabular`` LaTeX tables.

:class:`TexTable` emits a conventional ``tabular`` environment: no extra package dependencies beyond
what its options require, a single compile pass, and source that pastes into any report or journal
template. For per-cell borders, free-form drawing, and cell geometry, use
:class:`gerrytools.latex.tikz_table.TikzTable`, which shares this builder API but emits a
``nicematrix`` table.
"""

import logging

from gerrytools.latex._colors import xcolor_args
from gerrytools.latex._table_base import (
    TableOptions,
    _TableBase,
)
from gerrytools.latex._table_layout import RenderedCell, RenderedTable, multicolumn_row
from gerrytools.logging import get_logger

__all__ = ["TableOptions", "TexTable"]

logger = get_logger(__name__)


class TexTable(_TableBase):
    """Class for generating LaTeX table code from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be converted to a LaTeX table
        use_defaults (bool, optional): Whether to initialize with default table options
            (bold headers, 4 decimal places, etc.). Defaults to True.

    Attributes:
        df (pd.DataFrame): The DataFrame to be converted to a LaTeX table"""

    # No `_setup_document` override: the plain ``tabular`` dialect needs no unconditional
    # packages. The document's preamble scan adds colortbl when \rowcolor/\cellcolor appear.

    def _generate_latex(self) -> str:
        """Generate the complete LaTeX table string."""
        layout = self._resolve_layout()
        return (
            self._generate_header(layout)
            + self._generate_body(layout)
            + self._generate_footer(layout)
        )

    def _multicolumn_format(self, layout: RenderedTable | None = None) -> str:
        r"""Generate the multicolumn header row (group headers).

        Divider rule: boundary dividers belong to the *preceding* cell; only the first emitted
        cell includes its left boundary, and every cell includes its right boundary.
        """
        group_row = (layout or self._resolve_layout()).group_row
        if group_row is None:
            return ""
        return multicolumn_row(group_row)

    def _generate_header(self, layout: RenderedTable | None = None) -> str:
        r"""Generate the LaTeX table header string.

        Generally includes the \multicolumn row (if applicable) and the column titles.

        Returns:
            str: LaTeX table header string.
        """
        layout = layout or self._resolve_layout()
        header_string = f"{{{self._column_format(layout)}}}"
        if layout.top_rule is not None:
            header_string += "\n" + layout.top_rule + "\n"
        for row in layout.rows:
            if row.kind == "data":
                break
            if row.kind == "group":
                header_string += f"\n{self._multicolumn_format(layout)}"
            else:
                header_string += "\n" + " & ".join(cell.text for cell in row.cells) + r" \\"

        header_string = rf"\begin{{tabular}}{header_string}" + "\n"
        logger.log(
            logging.DEBUG,
            "Generated LaTeX table header:\n%s",
            header_string,
            stacklevel=2,
        )
        return header_string

    @staticmethod
    def _cell_text(cell: RenderedCell) -> str:
        return cell.text if cell.fill is None else f"\\cellcolor{cell.fill}{cell.text}"

    def _generate_body(self, layout: RenderedTable | None = None) -> str:
        """Generate the LaTeX table body string.

        Returns:
            str: LaTeX table body string.
        """
        layout = layout or self._resolve_layout()
        body_string = ""
        for row in layout.rows:
            if row.kind != "data":
                continue
            body_string += layout.hrule * row.rules_before
            if row.rules_before:
                body_string += "\n"
            if row.fill is not None:
                body_string += "\\rowcolor" + xcolor_args(row.fill) + "\n"
            body_string += " & ".join(self._cell_text(cell) for cell in row.cells) + r" \\" + "\n"
        if layout.trailing_rules:
            body_string += layout.hrule * layout.trailing_rules + "\n"

        logger.log(logging.DEBUG, "Generated LaTeX table body:\n%s", body_string, stacklevel=2)
        return body_string

    def _generate_footer(self, layout: RenderedTable | None = None) -> str:
        """Generate the LaTeX table footer string."""
        layout = layout or self._resolve_layout()
        footer_str = ""
        if layout.bottom_rule is not None:
            footer_str += "\n" + layout.bottom_rule + "\n"
        footer_str += r"\end{tabular}"
        logger.log(logging.DEBUG, "Generated LaTeX table footer:\n%s", footer_str, stacklevel=2)
        return footer_str
