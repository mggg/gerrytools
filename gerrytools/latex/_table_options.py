"""Mutable builder options shared by both LaTeX table dialects."""

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Optional

from gerrytools.latex._colors import LatexColorSpec
from gerrytools.latex._table_layout import IndexColumn, TablePreamble
from gerrytools.latex.formatters import (
    CellWrapper,
    IndexCellWrapper,
)


@dataclass
class TableOptions:
    r"""Track the configurable state of a LaTeX table.

    Attributes:
        toprule_cmd (str | None): LaTeX command for the top rule of the table. Default is None.
        bottomrule_cmd (str | None): LaTeX command for the bottom rule of the table.
            Default is None.
        hrule_cmd (str): LaTeX command for horizontal rules in the table. Default is r"\hline".
        include_column_headers (bool): Whether to render the column-header row. Default is True.
        include_group_headers (bool): Whether to render the group-header row when groups are set.
            Default is True.
        include_index (bool): Whether to include the DataFrame index in the table.
            Default is False.
        index_name (Optional[str]): Name to use for the index column header.
            If None and include_index is True, uses df.index.name or "". Default is None.
        index_column (IndexColumn): Parsed index-column syntax.
        group_index (IndexColumn | None): Optional syntax for the group-header index cell.
            Defaults to None.
        nan_string (str): String to represent NaN values in the table. Default is "NaN".
        hrule_counts (list[int]): List of counts of horizontal rules at each boundary.
            Boundary 0 is after header, k is after row k-1. Default is empty list.
        preamble (TablePreamble): Data-column alignments and boundaries.
        group_preamble (TablePreamble | None): Explicit group-header syntax, excluding the index.
        row_highlight_colors (dict[int, LatexColorSpec]): Mapping from row index to its highlight
            color spec; unhighlighted rows are absent. Default is empty dict.
        number_fmt_fn (Optional[CellWrapper]): Formatter function for numerical values.
            Default is None.
        str_fmt_fn (Optional[CellWrapper]): Formatter function for string values.
            Default is None; string cells keep their escaped rendering.
        col_formatters (dict[Hashable, CellWrapper]): Dictionary mapping column labels to
            formatter functions. Default is empty dict.
        row_formatters (dict[int, CellWrapper]): Dictionary mapping row indices to formatter
            functions. Default is empty dict.
        index_fmt_fn (Optional[IndexCellWrapper]): Formatter function for index values.
            Default is None.
        groups_to_cols (dict[str, list[Hashable]]): Dictionary mapping group names to lists of
            raw column labels. Default is empty dict.
        bold_group_headers (bool): Whether to bold group headers. Default is True.
        italic_group_headers (bool): Whether to italicize group headers. Default is False.
        bold_column_headers (bool): Whether to bold column headers. Default is False.
        italic_column_headers (bool): Whether to italicize column headers. Default is False.
    """

    toprule_cmd: str | None = None
    bottomrule_cmd: str | None = None
    hrule_cmd: str = r"\hline"

    include_column_headers: bool = True
    include_group_headers: bool = True

    include_index: bool = False
    index_name: Optional[str] = None
    index_column: IndexColumn = IndexColumn()
    group_index: IndexColumn | None = None

    nan_string: str = "NaN"
    hrule_counts: list[int] = field(default_factory=list)

    preamble: TablePreamble = TablePreamble.plain(0)
    group_preamble: TablePreamble | None = None

    row_highlight_colors: dict[int, LatexColorSpec] = field(default_factory=dict)

    number_fmt_fn: Optional[CellWrapper] = None
    str_fmt_fn: Optional[CellWrapper] = None
    col_formatters: dict[Hashable, CellWrapper] = field(default_factory=dict)
    row_formatters: dict[int, CellWrapper] = field(default_factory=dict)
    index_fmt_fn: Optional[IndexCellWrapper] = None

    groups_to_cols: dict[str, list[Hashable]] = field(default_factory=dict)

    bold_group_headers: bool = True
    italic_group_headers: bool = False
    bold_column_headers: bool = False
    italic_column_headers: bool = False
