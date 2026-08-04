"""TikZ-augmented table renderer from a pandas DataFrame.

Mirrors the :class:`TexTable` interface but generates a ``nicematrix`` ``NiceTabular`` environment
instead of a plain ``tabular``.  LaTeX's own tabular engine does the layout — so rules, spacing, and
double rules are pixel-identical to :class:`TexTable` by construction — while nicematrix exposes
every cell as a PGF/TikZ node, so you can still:

* Add arbitrary ``\\draw`` commands referencing cell nodes such as ``(table-2-3.north west)`` or
  the boundary lattice ``(row-i)``/``(col-j)``.
* Control individual cell borders (top/right/bottom/left) per cell.
* Fill rows and cells with background colours.

The generated LaTeX requires two compile passes (nicematrix stores node positions in the aux file);
:attr:`document` configures that automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from gerrytools.latex._colors import xcolor_args
from gerrytools.latex._table_base import (
    TableOptions,
    _TableBase,
)
from gerrytools.latex._table_layout import RenderedRow, RenderedTable, multicolumn_row

_BORDER_SIDES = frozenset({"top", "bottom", "left", "right"})
_TABLE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_DIMENSION_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:bp|cc|cm|dd|em|ex|in|mm|pc|pt|sp)")


# ---------------------------------------------------------------------------
# Options dataclass
# ---------------------------------------------------------------------------


@dataclass
class TikzTableOptions(TableOptions):
    r"""Options for a TikZ-based LaTeX table.

    Extends :class:`TableOptions` with the ``NiceTabular``-specific settings; every shared field
    keeps the same semantics.

    Attributes:
        cell_space_limits (str): Minimum vertical space around cell content as a LaTeX dimension.
            Defaults to ``"1pt"``.
        table_name (str): PGF node-name prefix for the table. Defaults to ``"table"``.
        extra_draws (list[str]): Raw TikZ commands emitted after the table. Defaults to an empty
            list.
        cell_borders (dict[tuple[int, int], set[str]]): Border sides keyed by one-based
            ``(row, column)`` cell coordinates. Defaults to an empty mapping.
    """

    # --- NiceTabular-specific ---
    cell_space_limits: str = "1pt"
    table_name: str = "table"
    extra_draws: list[str] = field(default_factory=list)

    # Per-cell border control: maps (tikz_row_1based, tikz_col_1based) to a set of sides.  Valid
    # sides: "top", "bottom", "left", "right".
    cell_borders: dict[tuple[int, int], set[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TikzTable(_TableBase):
    """Generate a ``nicematrix`` ``NiceTabular`` table from a pandas DataFrame.

    The builder API is identical to :class:`TexTable`; both classes share one implementation and
    differ only in the TeX dialect they emit. The default output matches the visual appearance of
    ``TexTable``: no cell borders, horizontal rules where ``\\hline`` would appear, and vertical
    rules where ``|`` would appear in the tabular preamble. TikZ-specific extras let you control
    cell geometry, inject raw ``\\draw`` commands, and set per-cell borders.

    The generated document requires the ``nicematrix`` and ``tikz`` packages and two compiler passes
    (registered automatically at construction), because ``nicematrix`` records cell positions on the
    first pass and draws on the second.

    Args:
        df (pd.DataFrame): Source data.
        use_defaults (bool): When ``True``, bold column headers, 4 decimal places, and a double
            rule above the first data row are applied (the same defaults as :class:`TexTable`).
    """

    _options_cls: type[TableOptions] = TikzTableOptions
    _options: TikzTableOptions

    def _setup_document(self) -> None:
        """Register the ``nicematrix`` dialect's packages and two-pass compile."""
        self._document.add_packages(["nicematrix", "tikz"])
        self._document.add_command("\\usetikzlibrary{calc}")
        self._document.compile_passes = 2

    # ==================================================================
    #   NiceTabular-specific setters
    # ==================================================================

    def set_cell_space_limits(self, limit: str) -> None:
        """Set nicematrix's minimal vertical space around cell content.

        Args:
            limit (str): Literal LaTeX dimension (e.g. ``"2pt"``).

        Raises:
            ValueError: If ``limit`` is not a safe LaTeX dimension.
        """
        if not isinstance(limit, str) or _DIMENSION_RE.fullmatch(limit) is None:
            raise ValueError("Cell-space limit must be a LaTeX dimension such as '2pt'.")
        self._options.cell_space_limits = limit

    def set_table_name(self, name: str) -> None:
        """Set the nicematrix name used to address this table's cells in TikZ draws.

        Cells become addressable as ``(<name>-<row>-<col>)``. The default is ``"table"``;
        give each table a distinct name when pasting several ``TikzTable`` bodies into one
        document, since duplicate nicematrix names collide.

        Args:
            name (str): PGF node-name prefix for the ``NiceTabular`` environment.

        Raises:
            ValueError: If ``name`` is not a safe PGF node-name prefix.
        """
        if not isinstance(name, str) or _TABLE_NAME_RE.fullmatch(name) is None:
            raise ValueError(
                "Table name must start with an ASCII letter and contain only ASCII letters, "
                "digits, and '-'."
            )
        self._options.table_name = name

    def add_draw(self, draw_cmd: str) -> None:
        r"""Append a raw TikZ command to the table's ``\\CodeAfter`` block.

        Cells are addressable as ``(table-<row>-<col>)`` with the usual TikZ anchors, and the
        boundary lattice as ``(row-<i>)`` / ``(col-<j>)``.

        Args:
            draw_cmd (str): A complete TikZ command, including the trailing semicolon.
        """
        self._options.extra_draws.append(draw_cmd)

    def clear_extra_draws(self) -> None:
        """Remove every command added with :meth:`add_draw`."""
        self._options.extra_draws = []

    def set_cell_border(
        self,
        row: int | list[int],
        col: int | list[int],
        sides: str | Iterable[str],
    ) -> None:
        """Specify which borders to draw on individual cells.

        The (row, col) indices are 1-based TikZ matrix coordinates — row 1 is the first rendered row
        (group-header or column-header), and column 1 is the leftmost column.

        Args:
            row (int | list[int]): TikZ row index (or list of indices).
            col (int | list[int]): TikZ column index (or list of indices).
            sides (str | Iterable[str]): ``"top"``, ``"bottom"``, ``"left"``,
                ``"right"`` — or an iterable of them, or ``"all"``.

        Raises:
            ValueError: If a side is unsupported or a row or column index is out of bounds.
        """
        rows = [row] if isinstance(row, int) else list(row)
        cols = [col] if isinstance(col, int) else list(col)
        if isinstance(sides, str):
            side_set = set(_BORDER_SIDES) if sides == "all" else {sides}
        else:
            side_set = set(sides)
        invalid_sides = side_set - _BORDER_SIDES
        if invalid_sides:
            raise ValueError(f"Unsupported cell border side(s): {sorted(invalid_sides)!r}")

        layout = self._resolve_layout()
        row_count = len(layout.rows)
        col_count = len(layout.preamble.alignments)
        for label, indices, limit in (("Row", rows, row_count), ("Column", cols, col_count)):
            for index in indices:
                if index < 1 or index > limit:
                    raise ValueError(
                        f"{label} index {index} is out of bounds; TikZ matrix coordinates are "
                        f"1-based, so this table allows 1 up to {limit}."
                    )
        for r in rows:
            for c in cols:
                self._options.cell_borders.setdefault((r, c), set()).update(side_set)

    def clear_cell_borders(self) -> None:
        """Remove all per-cell border specifications."""
        self._options.cell_borders = {}

    # ==================================================================
    #   Rendering
    # ==================================================================

    def _header_rule_metrics(self) -> tuple[str, str]:
        """Line width and inter-rule step for the pushed-up header rule stack.

        The values are LaTeX length expressions matching the active ``hrule_cmd`` so the drawn rules
        look like real stacked rules: booktabs ``\\midrule`` uses ``\\lightrulewidth``,
        ``\\toprule``/``\\bottomrule`` use ``\\heavyrulewidth``, and ``\\hline`` (or anything else)
        uses ``\\arrayrulewidth``. The step is ``\\doublerulesep`` throughout, which reproduces a
        tabular double rule and reads as a clean multi-rule.
        """
        cmd = self._options.hrule_cmd
        if cmd == r"\midrule":
            return r"\lightrulewidth", r"\doublerulesep"
        if cmd in (r"\toprule", r"\bottomrule"):
            return r"\heavyrulewidth", r"\doublerulesep"
        return r"\arrayrulewidth", r"\doublerulesep"

    def _header_rule_strut(self, rule_count: int) -> str:
        """Zero-width depth strut giving the last header row room for the stack.

        Depth ``(count - 1) * step + width`` reaches just past the topmost drawn rule so it sits in
        page-coloured space inside the header rather than bleeding into the first data row's colour
        band.
        """
        width, step = self._header_rule_metrics()
        extra = rule_count - 1
        return rf"{{\rule[-\dimexpr{extra}{step}+{width}\relax]{{0pt}}{{0pt}}}}"

    def _code_before(self, layout: RenderedTable) -> list[str]:
        r"""Build the ``\CodeBefore`` block for row and cell backgrounds."""
        entries: list[str] = []
        data_index = 0
        for row in layout.rows:
            if row.kind != "data":
                continue
            nicerow = layout.data_start + data_index
            data_index += 1
            if row.fill is not None:
                entries.append(f"\\rowcolor{xcolor_args(row.fill)}{{{nicerow}}}")
            for column, cell in enumerate(row.cells, start=1):
                if cell.fill is not None:
                    entries.append(f"\\cellcolor{cell.fill}{{{nicerow}-{column}}}")
        if not entries:
            return []
        return [r"\CodeBefore", *(f"  {entry}" for entry in entries), r"\Body"]

    def _group_header_row(
        self,
        row: RenderedRow,
        *,
        push_header_double: bool,
        rule_count: int,
        has_column_row: bool,
    ) -> str:
        r"""Translate the group header with the same ``\multicolumn`` semantics as TexTable."""
        strut = (
            self._header_rule_strut(rule_count) if push_header_double and not has_column_row else ""
        )
        return multicolumn_row(row).removesuffix(" \\\\") + strut + r" \\"

    def _data_rows_with_rules(
        self,
        layout: RenderedTable,
        *,
        push_header_double: bool,
    ) -> list[str]:
        """Translate resolved data rows and their rule events."""
        lines: list[str] = []
        data_index = 0
        for row in layout.rows:
            if row.kind != "data":
                continue
            count = row.rules_before
            if data_index == 0 and push_header_double:
                count = 1
            lines.extend([layout.hrule] * count)
            lines.append(" & ".join(cell.text for cell in row.cells) + r" \\")
            data_index += 1
        lines.extend([layout.hrule] * layout.trailing_rules)
        return lines

    def _pushed_header_rule_draws(self, layout: RenderedTable, rule_count: int) -> list[str]:
        """Draw the extra header rules above the one emitted in the body."""
        ncols = len(layout.preamble.alignments)
        width, step = self._header_rule_metrics()
        return [
            rf"\draw[line width={width}] "
            f"([yshift={k}{step}]row-{layout.data_start}-|col-1) -- "
            f"([yshift={k}{step}]row-{layout.data_start}-|col-{ncols + 1});"
            for k in range(1, rule_count)
        ]

    def _cell_border_draws(self) -> list[str]:
        """Canonicalize cell sides into unique horizontal and vertical segments."""
        horiz_edges: set[tuple[int, int]] = set()
        vert_edges: set[tuple[int, int]] = set()
        for (r, c), sides in self._options.cell_borders.items():
            for side in sides:
                if side == "top":
                    horiz_edges.add((c, r))
                elif side == "bottom":
                    horiz_edges.add((c, r + 1))
                elif side == "left":
                    vert_edges.add((r, c))
                else:  # ``set_cell_border`` validates the four-side vocabulary.
                    vert_edges.add((r, c + 1))
        return [
            *(
                f"\\draw (row-{row_boundary}-|col-{c}) -- (row-{row_boundary}-|col-{c + 1});"
                for c, row_boundary in sorted(horiz_edges)
            ),
            *(
                f"\\draw (row-{r}-|col-{col_boundary}) -- (row-{r + 1}-|col-{col_boundary});"
                for r, col_boundary in sorted(vert_edges)
            ),
        ]

    def _code_after(
        self,
        *,
        layout: RenderedTable,
        push_header_double: bool,
        rule_count: int,
    ) -> list[str]:
        r"""Build the ``\CodeAfter`` block for rules, cell borders, and user draws."""
        draws: list[str] = []
        if push_header_double:
            draws.extend(self._pushed_header_rule_draws(layout, rule_count))
        draws.extend(self._cell_border_draws())
        draws.extend(self._options.extra_draws)
        if not draws:
            return []
        return [
            r"\CodeAfter",
            r"\begin{tikzpicture}",
            *(f"  {entry}" for entry in draws),
            r"\end{tikzpicture}",
        ]

    def _generate_latex(self) -> str:
        """Build the complete ``NiceTabular`` string from its rendering phases."""
        layout = self._resolve_layout()
        first_data_row = next((row for row in layout.rows if row.kind == "data"), None)
        rule_count = first_data_row.rules_before if first_data_row is not None else 0
        has_headers = any(row.kind != "data" for row in layout.rows)
        push_header_double = rule_count >= 2 and has_headers
        env_options = (
            f"[name={self._options.table_name}, "
            f"cell-space-limits={self._options.cell_space_limits}]"
        )
        lines = [
            f"\\begin{{NiceTabular}}{{{self._column_format(layout)}}}{env_options}",
            *self._code_before(layout),
        ]
        if layout.top_rule is not None:
            lines.append(layout.top_rule)
        for row in layout.rows:
            if row.kind == "data":
                break
            if row.kind == "group":
                lines.append(
                    self._group_header_row(
                        row,
                        push_header_double=push_header_double,
                        rule_count=rule_count,
                        has_column_row=any(item.kind == "columns" for item in layout.rows),
                    )
                )
            else:
                strut = self._header_rule_strut(rule_count) if push_header_double else ""
                lines.append(" & ".join(cell.text for cell in row.cells) + strut + r" \\")
        lines.extend(
            self._data_rows_with_rules(
                layout,
                push_header_double=push_header_double,
            )
        )
        if layout.bottom_rule is not None:
            lines.append(layout.bottom_rule)
        lines.extend(
            self._code_after(
                layout=layout,
                push_header_double=push_header_double,
                rule_count=rule_count,
            )
        )
        lines.append(r"\end{NiceTabular}")
        return "\n".join(lines)
