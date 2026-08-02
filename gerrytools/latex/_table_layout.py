"""Immutable preamble and rendered-layout values shared by both table dialects.

Alongside the value types, this module owns the pure layout-resolution functions: they read only
``(df, options)`` and mutate nothing, turning the mutable builder state into one immutable
:class:`RenderedTable` consumed by the syntax-only emitters.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Literal

import pandas as pd

from gerrytools.latex._colors import LatexColorSpec, split_cell_fill
from gerrytools.latex._table_preamble import (
    _consume_balanced,
    _infer_group_cell_align_from_data,
)
from gerrytools.latex._text import latex_escape

if TYPE_CHECKING:
    from gerrytools.latex._table_options import TableOptions
    from gerrytools.latex.formatters import TableIndexValue


@dataclass(frozen=True)
class TableBoundary:
    """Syntax attached to one boundary between rendered columns."""

    vrules: int = 0
    extra: str = ""

    def merged(self, other: TableBoundary) -> TableBoundary:
        """Combine index-owned syntax with the underlying data boundary."""
        return TableBoundary(self.vrules + other.vrules, self.extra + other.extra)

    def _parts(self, index: int) -> tuple[str, str, str]:
        """Return closing, between-column, and opening syntax."""
        closing: list[str] = []
        between: list[str] = []
        opening: list[str] = []
        position = 0
        while position < len(self.extra):
            kind = self.extra[position]
            if kind not in "@!<>":
                raise ValueError(
                    f"Unsupported boundary token {self.extra[position:]!r} "
                    f"at column boundary {index}."
                )
            try:
                group, next_position = _consume_balanced(self.extra, position + 1, "{", "}")
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported boundary token {self.extra[position:]!r} "
                    f"at column boundary {index}."
                ) from exc
            token = f"{kind}{{{group}}}"
            if kind == "<":
                closing.append(token)
            elif kind == ">":
                opening.append(token)
            else:
                between.append(token)
            position = next_position
        middle = "|" * self.vrules + "".join(between)
        return "".join(closing), middle, "".join(opening)

    def format(self, index: int) -> str:
        """Emit boundary syntax in the order required by LaTeX tabular preambles."""
        return "".join(self._parts(index))

    def before_column(self, index: int, *, include_boundary: bool) -> str:
        """Emit syntax owned by the following column."""
        _, middle, opening = self._parts(index)
        return (middle if include_boundary else "") + opening

    def after_column(self, index: int) -> str:
        """Emit syntax owned by the preceding column."""
        closing, middle, _ = self._parts(index)
        return closing + middle


@dataclass(frozen=True)
class TablePreamble:
    """Ordered columns and their adjacent boundaries."""

    alignments: tuple[str, ...]
    boundaries: tuple[TableBoundary, ...]

    @classmethod
    def plain(cls, count: int) -> TablePreamble:
        return cls(("c",) * count, (TableBoundary(),) * (count + 1))

    @classmethod
    def parsed(
        cls,
        alignments: list[str],
        vrules: list[int],
        extras: list[str],
    ) -> TablePreamble:
        return cls(
            tuple(alignments),
            tuple(TableBoundary(count, extra) for count, extra in zip(vrules, extras, strict=True)),
        )

    def with_rule(self, boundary: int, count: int) -> TablePreamble:
        boundaries = list(self.boundaries)
        current = boundaries[boundary]
        boundaries[boundary] = TableBoundary(current.vrules + count, current.extra)
        return TablePreamble(self.alignments, tuple(boundaries))

    def without_rules(self) -> TablePreamble:
        return TablePreamble(
            self.alignments,
            tuple(TableBoundary(extra=boundary.extra) for boundary in self.boundaries),
        )

    def reordered(self, order: list[int]) -> TablePreamble:
        """Move column specs while keeping separator syntax at rendered boundaries."""
        if order == list(range(len(self.alignments))):
            return self
        alignments = tuple(self.alignments[index] for index in order)
        boundaries: list[TableBoundary] = []
        for position in range(len(order) + 1):
            positional = self.boundaries[position]
            _, middle, _ = positional._parts(position)
            if position == 0:
                closing, _, _ = self.boundaries[0]._parts(0)
            else:
                closing, _, _ = self.boundaries[order[position - 1] + 1]._parts(
                    order[position - 1] + 1
                )
            if position == len(order):
                _, _, opening = self.boundaries[-1]._parts(len(order))
            else:
                _, _, opening = self.boundaries[order[position]]._parts(order[position])
            boundaries.append(
                TableBoundary(
                    positional.vrules,
                    closing + middle[positional.vrules :] + opening,
                )
            )
        return TablePreamble(alignments, tuple(boundaries))


@dataclass(frozen=True)
class IndexColumn:
    """Index colspec plus syntax owned by its left and right boundaries."""

    alignment: str = "c"
    left: TableBoundary = TableBoundary()
    right: TableBoundary = TableBoundary()

    def with_rule(self, side: Literal["left", "right"], count: int) -> IndexColumn:
        boundary = getattr(self, side)
        updated = TableBoundary(boundary.vrules + count, boundary.extra)
        if side == "left":
            return IndexColumn(self.alignment, updated, self.right)
        return IndexColumn(self.alignment, self.left, updated)

    def without_rules(self) -> IndexColumn:
        return IndexColumn(
            self.alignment,
            TableBoundary(extra=self.left.extra),
            TableBoundary(extra=self.right.extra),
        )


@dataclass(frozen=True)
class RenderedCell:
    """One resolved cell, optionally spanning columns in a group-header row."""

    text: str
    span: int = 1
    alignment: str | None = None
    left: TableBoundary | None = None
    right: TableBoundary | None = None
    fill: str | None = None


@dataclass(frozen=True)
class RenderedRow:
    """One semantic row with its rule event and optional background."""

    kind: Literal["group", "columns", "data"]
    cells: tuple[RenderedCell, ...]
    rules_before: int = 0
    fill: LatexColorSpec | None = None
    explicit_preamble: bool = False


@dataclass(frozen=True)
class RenderedTable:
    """Complete immutable table semantics consumed by syntax-only emitters."""

    preamble: TablePreamble
    rows: tuple[RenderedRow, ...]
    trailing_rules: int
    hrule: str
    top_rule: str | None
    bottom_rule: str | None

    @property
    def group_row(self) -> RenderedRow | None:
        return next((row for row in self.rows if row.kind == "group"), None)

    @property
    def data_start(self) -> int:
        return 1 + sum(row.kind != "data" for row in self.rows)


# ---- pure layout resolution over (df, options) ----


def column_format(preamble: TablePreamble) -> str:
    """Render a preamble as the tabular column format (e.g. ``'|c|p{2cm}||S[...]|'``)."""
    parts: list[str] = []
    for index, alignment in enumerate(preamble.alignments):
        parts.extend((preamble.boundaries[index].format(index), alignment))
    parts.append(preamble.boundaries[-1].format(len(preamble.alignments)))
    return "".join(parts)


def multicolumn_row(row: RenderedRow) -> str:
    r"""Render one resolved group-header row with tabular ``\multicolumn`` semantics."""
    parts: list[str] = []
    for index, cell in enumerate(row.cells):
        assert cell.alignment is not None and cell.left is not None and cell.right is not None
        left = cell.left.before_column(index, include_boundary=index == 0)
        right = cell.right.after_column(index + 1)
        parts.append(
            rf"\multicolumn{{{cell.span}}}"
            rf"{{{left}{cell.alignment}{right}}}{{{cell.text}}}"
        )
    return " & ".join(parts) + r" \\"


def has_groups(options: TableOptions) -> bool:
    """Whether any named header groups are set."""
    return set(options.groups_to_cols.keys()) != {""}


def column_ordering(df: pd.DataFrame, options: TableOptions) -> list[Hashable]:
    """Data columns in render order (group order when groups are set)."""
    if has_groups(options):
        return [column for columns in options.groups_to_cols.values() for column in columns]
    return list(df.columns)


def _styled_header_text(text: str, *, bold: bool, italic: bool) -> str:
    """Wrap header text in bold/italic commands per the options."""
    if bold:
        text = rf"\textbf{{{text}}}"
    if italic:
        text = rf"\textit{{{text}}}"
    return text


def _resolved_index_name(df: pd.DataFrame, options: TableOptions) -> str:
    """Index column header text: the configured name, else the frame's index name, else ''."""
    if options.index_name is not None:
        return options.index_name
    index_name = df.index.name
    return str(index_name) if index_name is not None else ""


def _column_header_cells(df: pd.DataFrame, options: TableOptions) -> list[str]:
    """Styled column-header cells in render order (index cell first when present)."""
    header_texts: list[str] = []
    if options.include_index:
        header_texts.append(_resolved_index_name(df, options))
    header_texts.extend(str(column) for column in column_ordering(df, options))
    return [
        _styled_header_text(
            latex_escape(text),
            bold=options.bold_column_headers,
            italic=options.italic_column_headers,
        )
        for text in header_texts
    ]


def _format_cell_value(
    options: TableOptions, row_idx: int, col: Hashable, cell_value: object
) -> str:
    """Render one cell through the NaN, column, row, and default formatters."""
    # pd.isna returns an array for array-valued cells; bool() reproduces the historical
    # ambiguous-truth ValueError for those, while scalar cells stay a plain bool.
    cell_is_na = pd.isna(cell_value)
    if not isinstance(cell_is_na, bool):
        cell_is_na = bool(cell_is_na)
    if cell_is_na:
        return options.nan_string
    escaped = latex_escape(str(cell_value))
    if col in options.col_formatters:
        return options.col_formatters[col](cell_value, escaped)[1]
    if row_idx in options.row_formatters:
        return options.row_formatters[row_idx](cell_value, escaped)[1]
    if pd.api.types.is_bool(cell_value):
        return escaped
    if isinstance(cell_value, Real) and options.number_fmt_fn is not None:
        return options.number_fmt_fn(cell_value, escaped)[1]
    if isinstance(cell_value, str) and options.str_fmt_fn is not None:
        return options.str_fmt_fn(cell_value, escaped)[1]
    return escaped


def _format_index_cell(options: TableOptions, df_row_idx: TableIndexValue) -> str:
    """Render one index cell through the index formatter (escaped fallback)."""
    escaped = latex_escape(str(df_row_idx))
    if options.index_fmt_fn is not None:
        return options.index_fmt_fn(df_row_idx, escaped)[1]
    return escaped


def resolved_preamble(df: pd.DataFrame, options: TableOptions) -> TablePreamble:
    """Return the complete rendered preamble, adding the index without mutating data syntax."""
    preamble = options.preamble
    expected = len(df.columns)
    if len(preamble.alignments) != expected or len(preamble.boundaries) != expected + 1:
        raise ValueError(
            "Current tabular format does not match DataFrame columns. Got "
            f"{len(preamble.alignments)} colspecs but expected {expected}."
        )
    ordering = column_ordering(df, options)
    positions = {column: index for index, column in enumerate(df.columns)}
    preamble = preamble.reordered([positions[column] for column in ordering])
    if not options.include_index:
        return preamble
    index = options.index_column
    boundaries = (
        index.left,
        index.right.merged(preamble.boundaries[0]),
        *preamble.boundaries[1:],
    )
    return TablePreamble((index.alignment, *preamble.alignments), boundaries)


def _group_cell(
    options: TableOptions,
    name: str,
    span: int,
    alignment: str,
    left: TableBoundary,
    right: TableBoundary,
) -> RenderedCell:
    text = latex_escape(str(name))
    if text:
        text = _styled_header_text(
            text,
            bold=options.bold_group_headers,
            italic=options.italic_group_headers,
        )
    return RenderedCell(text, span, alignment, left, right)


def _group_header_cells(options: TableOptions, preamble: TablePreamble) -> tuple[RenderedCell, ...]:
    """Resolve group spans, alignments, and boundary ownership once for both dialects."""
    groups = [(name, len(columns)) for name, columns in options.groups_to_cols.items() if columns]
    explicit = options.group_preamble
    cells: list[RenderedCell] = []

    if explicit is not None:
        expected = len(groups)
        if len(explicit.alignments) != expected or len(explicit.boundaries) != expected + 1:
            raise ValueError(
                f"Group-header preamble has {len(explicit.alignments)} cells but expected "
                f"{expected}."
            )
        cell_index = 0
        if options.include_index:
            index = options.group_index or options.index_column
            cells.append(
                RenderedCell(
                    "",
                    alignment=index.alignment,
                    left=index.left,
                    right=index.right.merged(explicit.boundaries[0]),
                )
            )
        for name, span in groups:
            cells.append(
                _group_cell(
                    options,
                    name,
                    span,
                    explicit.alignments[cell_index],
                    explicit.boundaries[cell_index],
                    explicit.boundaries[cell_index + 1],
                )
            )
            cell_index += 1
        return tuple(cells)

    data_start = int(options.include_index)
    if options.include_index:
        cells.append(
            RenderedCell(
                "",
                alignment=preamble.alignments[0],
                left=preamble.boundaries[0],
                right=preamble.boundaries[1],
            )
        )
    offset = 0
    for name, span in groups:
        left_index = data_start + offset
        right_index = left_index + span
        offset += span
        cells.append(
            _group_cell(
                options,
                name,
                span,
                _infer_group_cell_align_from_data(
                    list(preamble.alignments), left_index, right_index
                ),
                preamble.boundaries[left_index],
                preamble.boundaries[right_index],
            )
        )
    return tuple(cells)


def resolve_layout(df: pd.DataFrame, options: TableOptions) -> RenderedTable:
    """Resolve mutable builder options and frame values into one immutable render model."""
    preamble = resolved_preamble(df, options)
    rows: list[RenderedRow] = []
    if has_groups(options) and options.include_group_headers:
        rows.append(
            RenderedRow(
                "group",
                _group_header_cells(options, preamble),
                explicit_preamble=options.group_preamble is not None,
            )
        )
    if options.include_column_headers:
        rows.append(
            RenderedRow(
                "columns",
                tuple(RenderedCell(text) for text in _column_header_cells(df, options)),
            )
        )

    ordering = column_ordering(df, options)
    for row_index, frame_index in enumerate(df.index):
        cells: list[RenderedCell] = []
        if options.include_index:
            fill, text = split_cell_fill(_format_index_cell(options, frame_index))
            cells.append(RenderedCell(text, fill=fill))
        for column in ordering:
            fill, text = split_cell_fill(
                _format_cell_value(options, row_index, column, df[column].iloc[row_index])
            )
            cells.append(RenderedCell(text, fill=fill))
        rules = options.hrule_counts[row_index] if row_index < len(options.hrule_counts) else 0
        rows.append(
            RenderedRow(
                "data",
                tuple(cells),
                rules_before=rules,
                fill=options.row_highlight_colors.get(row_index),
            )
        )

    trailing = options.hrule_counts[len(df)] if len(options.hrule_counts) > len(df) else 0
    return RenderedTable(
        preamble=preamble,
        rows=tuple(rows),
        trailing_rules=trailing,
        hrule=options.hrule_cmd,
        top_rule=options.toprule_cmd,
        bottom_rule=options.bottomrule_cmd,
    )
