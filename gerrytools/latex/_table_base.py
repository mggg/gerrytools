"""Shared builder machinery for the LaTeX table classes.

:class:`TexTable` and :class:`TikzTable` are two emitters over one builder API: the same options,
setters, formatters, and document plumbing, differing only in the TeX dialect they generate.
:class:`_TableBase` owns everything shared so the dialects cannot drift; each subclass supplies
``_setup_document()`` and ``_generate_latex()``.
"""

import inspect
from collections.abc import Hashable
from numbers import Real
from typing import Callable, Iterable, Literal, Optional, cast

import pandas as pd

from gerrytools.latex._colors import to_latex_xcolor_or_html_spec
from gerrytools.latex._table_layout import (
    IndexColumn,
    RenderedTable,
    TableBoundary,
    TablePreamble,
    column_format,
    resolve_layout,
    resolved_preamble,
)
from gerrytools.latex._table_options import TableOptions
from gerrytools.latex._table_preamble import _parse_tabular_preamble
from gerrytools.latex.document import TexDocument
from gerrytools.latex.formatters import (
    CellWrapper,
    IndexCellWrapper,
    TableCellValue,
    TableIndexValue,
    latex_commands_for,
    round_decimals,
)
from gerrytools.typing import Color


class _Unset:
    pass


_UNSET = _Unset()


def _validate_rule_count(count: int) -> None:
    """Reject rule counts below 1: ``"\\hline" * -3 == ""`` would silently cancel additions."""
    if count < 1:
        raise ValueError(f"Rule count must be at least 1, got {count}.")


class _TableBase:
    """Shared builder API for :class:`TexTable` and :class:`TikzTable`.

    The two public classes differ only in the TeX dialect they emit (a plain ``tabular`` versus a
    ``nicematrix`` ``NiceTabular``) and in the TikZ-only extras that dialect makes possible. Every
    option setter, formatter, and document affordance lives here so the two dialects cannot drift.
    """

    _options_cls: type[TableOptions] = TableOptions

    def __init__(self, df: pd.DataFrame, *, use_defaults: bool = True) -> None:
        """Initialize the table builder from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to convert to a LaTeX table. A copy is
                stored, so later edits to the original frame do not affect the table.
            use_defaults (bool, optional): Whether to seed the report-style defaults
                (bold headers, a double rule below the header, 4 decimal places).
                Defaults to True.
        """
        self.df = df
        self._document = TexDocument()
        self._setup_document()
        # Commands added on behalf of formatters, so clear_options can drop them again without
        # touching dialect- or user-registered commands.
        self._formatter_commands: list[str] = []
        self._options = self._fresh_options()
        if use_defaults:
            self._options.bold_column_headers = True
            self.add_hrule_above(0, 2)
            self.set_decimal_count(4)

    @property
    def df(self) -> pd.DataFrame:
        """The table's DataFrame.

        Replacement may change cell values and the index, but must preserve the original row
        count and column labels because layout options are sized and keyed at construction.
        """
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        if value.shape[1] == 0:
            raise ValueError("DataFrame must contain at least one column.")
        if not value.columns.is_unique:
            raise ValueError("DataFrame column labels must be unique.")
        if hasattr(self, "_df") and (
            len(value) != len(self._df) or not value.columns.equals(self._df.columns)
        ):
            raise ValueError(
                "A replacement DataFrame must have the same rows and columns as the original."
            )
        self._df = value.copy()

    def _fresh_options(self) -> TableOptions:
        """Build a plain-default options object with every vector sized from ``df.shape``.

        Eager sizing means the setters and emitters never need to seed or extend the boundary
        and rule vectors lazily.
        """
        n_rows, n_cols = self.df.shape
        return self._options_cls(
            groups_to_cols={"": list(self.df.columns)},
            preamble=TablePreamble.plain(n_cols),
            hrule_counts=[0] * (n_rows + 1),
        )

    def _setup_document(self) -> None:
        """Register the packages and commands this table dialect always needs."""

    def _generate_latex(self) -> str:
        """Render the table environment for this dialect."""
        raise NotImplementedError  # pragma: no cover

    def clear_options(self) -> None:
        """Reset all table options to plain defaults.

        Command definitions registered on behalf of formatters are dropped with the formatters
        they served; dialect- and user-registered commands are untouched.
        """
        for command in self._formatter_commands:
            if command in self._document.command_list:
                self._document.command_list.remove(command)
        self._formatter_commands = []
        self._options = self._fresh_options()

    def __repr__(self) -> str:  # pragma: no cover
        return self.document.to_tex()

    def __str__(self) -> str:  # pragma: no cover
        return self.document.to_tex()

    @property
    def document(self) -> TexDocument:
        """TexDocument: The LaTeX document associated with this table."""
        self._document.body_string = self._generate_latex()
        return self._document

    def print_table(self) -> None:
        """Print the table environment alone, without the surrounding document.

        The counterpart of ``print(table)``, which prints the full standalone document: use this
        when pasting the table into an existing report.
        """
        print(self._generate_latex())

    def preview(self) -> None:  # pragma: no cover
        """Render and preview the table through its ``TexDocument``.

        Returns:
            None
        """
        self.document.preview()

    def _register_formatter_latex_commands(self, fmt_fn: Callable) -> None:
        """Add any LaTeX preamble commands required by a formatter.

        Identically built formatters carry identical command names and bodies, so registration
        dedupes on the full command text.

        Args:
            fmt_fn (Callable): Formatter callable that may carry LaTeX command metadata.

        Returns:
            None
        """
        for command in latex_commands_for(fmt_fn):
            if command not in self._document.command_list:
                self._document.add_command(command)
                self._formatter_commands.append(command)

    def _register_colspec_packages(self, colspecs: list[str], extras: list[str]) -> None:
        """Register the packages implied by parsed colspec tokens.

        Registration happens at parse time because the document's macro scan cannot reliably
        recognize colspec tokens (``S[...]``, ``D{}{}{}``, ``!{...}``) inside an emitted
        ``\\begin{tabular}{...}`` line.
        """
        packages: list[str] = []
        if any(spec.startswith("S") for spec in colspecs):
            packages.append("siunitx")
        if any(spec.startswith("D") for spec in colspecs):
            packages.append("dcolumn")
        array_tokens = ("m{", "b{")
        array_extras = (">{", "<{", "!{")
        if any(spec.startswith(array_tokens) for spec in colspecs) or any(
            token in extra for extra in extras for token in array_extras
        ):
            packages.append("array")
        if packages:
            self._document.add_packages(packages)

    def include_index(
        self,
        name: Optional[str] | _Unset = _UNSET,
        alignment: str | _Unset = _UNSET,
        include: bool = True,
    ) -> None:
        """Add or remove the DataFrame index column.

        The alignment may be a rich preamble fragment (for example ``">{\\bfseries}c|"``).
        Its parsed syntax is stored separately from the data preamble, so removal requires no
        mutation reversal.

        Args:
            name (Optional[str]): Name to give the index in the table. If None, uses
                ``df.index.name`` or an empty header. When omitted, retains the configured name.
            alignment (str): Alignment spec for the index column. When omitted, retains the
                configured alignment, which defaults to "c".
            include (bool): Whether to include the index column. Defaults to True.

        Raises:
            ValueError: If the current tabular format is inconsistent with the
                DataFrame shape, or the alignment spec implies more than one column.
        """
        if name is not _UNSET:
            self._options.index_name = cast(Optional[str], name)
        if alignment is not _UNSET:
            specs, vrules, extras = _parse_tabular_preamble(cast(str, alignment))
            if len(specs) != 1:
                raise ValueError(
                    f"Index alignment must specify exactly 1 column, got {len(specs)}."
                )
            self._register_colspec_packages(specs, extras)
            self._options.index_column = IndexColumn(
                specs[0],
                TableBoundary(vrules[0], extras[0]),
                TableBoundary(vrules[1], extras[1]),
            )
        self._options.include_index = include

    def remove_index(self) -> None:
        """Remove the index from the generated latex table (if it exists)"""
        self._options.include_index = False

    # ---- shared rendering helpers (thin delegations to the pure functions) ----

    def _column_format(self, layout: RenderedTable | None = None) -> str:
        """Generate the tabular preamble (e.g. ``'|c|p{2cm}||S[...]|'``)."""
        return column_format((layout or self._resolve_layout()).preamble)

    def _resolved_preamble(self) -> TablePreamble:
        """Return the complete rendered preamble, adding the index without mutating data syntax."""
        return resolved_preamble(self.df, self._options)

    def _resolve_layout(self) -> RenderedTable:
        """Resolve mutable builder options and frame values into one immutable render model."""
        return resolve_layout(self.df, self._options)

    def add_hrule_above(self, index: int | list[int], count: int = 1) -> None:
        """Add horizontal rules above specified rows in the LaTeX table.

        Note: Row index 0 corresponds to the first row of data in the dataframe, so it will appear
        below the header row in the LaTeX table.

        Args:
            index (int | list[int]): Row index or list of row indices above which to add horizontal
                rules.
            count (int, optional): Number of horizontal rules to add. Defaults to 1.

        Raises:
            ValueError: If ``count`` is below 1 or any row index is out of bounds.
        """
        _validate_rule_count(count)
        if isinstance(index, int):
            indices = [index]
        else:
            indices = index

        for idx in indices:
            if idx < 0 or idx > len(self.df):
                raise ValueError(
                    f"Row index {idx} is out of bounds for DataFrame with {len(self.df)} rows."
                )
            self._options.hrule_counts[idx] += count

    def add_toprule(self, *, cmd: Optional[str] = None) -> None:
        """Add a rule to the top of the LaTeX table.

        Args:
            cmd (Optional[str], optional): LaTeX command for the top rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        self.set_toprule_command(cmd)

    def remove_toprule(self) -> None:
        """Remove the rule at the top of the LaTeX table."""
        self._options.toprule_cmd = None

    def add_bottomrule(self, *, cmd: Optional[str] = None) -> None:
        """Add a rule to the bottom of the LaTeX table.

        Args:
            cmd (Optional[str], optional): LaTeX command for the bottom rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        self.set_bottomrule_command(cmd)

    def remove_bottomrule(self) -> None:
        """Remove the rule at the bottom of the LaTeX table."""
        self._options.bottomrule_cmd = None

    def add_hrule_above_all(self, count: int = 1) -> None:
        """Add horizontal rules above all rows in the LaTeX table.

        Args:
            count (int, optional): Number of horizontal rules to add above each row. Defaults to 1.

        Raises:
            ValueError: If ``count`` is below 1.
        """
        _validate_rule_count(count)
        for idx in range(len(self.df)):
            self._options.hrule_counts[idx] += count

    def clear_all_hrule(self) -> None:
        """Remove all horizontal rules from the LaTeX table."""
        self._options.toprule_cmd = None
        self._options.bottomrule_cmd = None
        self._options.hrule_counts = [0] * (len(self.df) + 1)

    def add_vrule_left_of(self, col_idx: int | list[int], count: int = 1) -> None:
        """Add vertical rules to the left of specified columns in the LaTeX table.

        Note: Column index 0 corresponds to the first column in the dataframe which will include the
        index column if include_index is set to True in options.

        Args:
            col_idx (int | list[int]): Column index or list of column indices to the left of which
                to add vertical rules.
            count (int, optional): Number of vertical rules to add. Defaults to 1.

        Raises:
            ValueError: If ``count`` is below 1 or any column index is out of bounds.
        """
        self._add_vrules(col_idx, count, side="left")

    def _add_vrules(
        self,
        col_idx: int | list[int],
        count: int,
        *,
        side: Literal["left", "right"],
    ) -> None:
        _validate_rule_count(count)
        cols = [col_idx] if isinstance(col_idx, int) else col_idx
        include_index_offset = 1 if self._options.include_index else 0
        boundary_offset = int(side == "right")
        max_index = len(self.df.columns) + include_index_offset - boundary_offset

        for cidx in cols:
            if cidx < 0 or cidx > max_index:
                raise ValueError(
                    f"Column index {cidx} is out of bounds for DataFrame with "
                    f"{len(self.df.columns)} columns. "
                    f"You may add vrules to the {side} of index 0 up to index "
                    f"{max_index} for this dataframe."
                )
            self._add_vrule_boundary(cidx + boundary_offset, count)

    def _add_vrule_boundary(self, boundary: int, count: int) -> None:
        """Add rules to one rendered boundary without merging index and data state."""
        if not self._options.include_index:
            self._options.preamble = self._options.preamble.with_rule(boundary, count)
        elif boundary == 0:
            self._options.index_column = self._options.index_column.with_rule("left", count)
        elif boundary == 1:
            self._options.index_column = self._options.index_column.with_rule("right", count)
        else:
            self._options.preamble = self._options.preamble.with_rule(boundary - 1, count)

    def clear_all_vrule(self) -> None:
        """Remove all vertical rules from the LaTeX table."""
        self._options.preamble = self._options.preamble.without_rules()
        self._options.index_column = self._options.index_column.without_rules()
        if self._options.group_preamble is not None:
            self._options.group_preamble = self._options.group_preamble.without_rules()
        if self._options.group_index is not None:
            self._options.group_index = self._options.group_index.without_rules()

    def add_vrule_right_of(self, col_idx: int | list[int], count: int = 1) -> None:
        """Add vertical rules to the right of specified columns in the LaTeX table.

        Note: Column index 0 corresponds to the first column in the dataframe which will include the
        index column if include_index is set to True in options.

        Args:
            col_idx (int | list[int]): Column index or list of column indices to the right of which
                to add vertical rules.
            count (int, optional): Number of vertical rules to add. Defaults to 1.

        Raises:
            ValueError: If ``count`` is below 1 or any column index is out of bounds.
        """
        self._add_vrules(col_idx, count, side="right")

    def add_vrule_all(self, count: int = 1) -> None:
        """Add vertical rules around all columns in the LaTeX table.

        Args:
            count (int, optional): Number of vertical rules to add between each column.
                Defaults to 1.

        Raises:
            ValueError: If ``count`` is below 1.
        """
        _validate_rule_count(count)
        if not self._options.include_index:
            self._options.index_column = self._options.index_column.with_rule("left", count)
        total_cols = len(self.df.columns) + int(self._options.include_index)

        for idx in range(total_cols + 1):
            self._add_vrule_boundary(idx, count)

    def highlight_rows(self, rows: int | Iterable[int], color: Color = "yellow") -> None:
        """Highlight specified rows in the LaTeX table.

        Args:
            rows (int | Iterable[int]): Row index or iterable of row indices to highlight.
            color (Color, optional): Color to use for highlighting. Supports valid xcolor
                expressions (for example ``"amber!80!gray"``), hex color strings, RGB tuples,
                and other names parseable by GerryTools/Matplotlib (which are converted to
                HTML color form). Defaults to ``"yellow"``.
        """
        row_indices: list[int]

        if isinstance(rows, int):
            row_indices = [rows]
        else:
            row_indices = list(rows)

        color_spec = to_latex_xcolor_or_html_spec(color)

        for ridx in row_indices:
            if ridx < 0 or ridx >= len(self.df):
                raise ValueError(
                    f"Row index {ridx} is out of bounds for DataFrame with {len(self.df)} rows."
                )
            self._options.row_highlight_colors[ridx] = color_spec

    def remove_column_headers(self) -> None:
        """Remove the column headers from the LaTeX table."""
        self._options.include_column_headers = False

    def remove_group_headers(self) -> None:
        """Remove the group headers from the LaTeX table."""
        self._options.include_group_headers = False

    def remove_all_headers(self) -> None:
        """Remove all headers (both column and group) from the LaTeX table."""
        self.remove_column_headers()
        self.remove_group_headers()

    def include_column_headers(self) -> None:
        """Include the column headers in the LaTeX table."""
        self._options.include_column_headers = True

    def include_group_headers(self) -> None:
        """Include the group headers in the LaTeX table."""
        self._options.include_group_headers = True

    def include_all_headers(self) -> None:
        """Include all headers (both column and group) in the LaTeX table."""
        self.include_column_headers()
        self.include_group_headers()

    def set_column_headers_text_format(self, bold: bool = True, italic: bool = False) -> None:
        """Set column-header emphasis styling.

        Args:
            bold (bool, optional): Whether to apply bold styling. Defaults to True.
            italic (bool, optional): Whether to apply italic styling. Defaults to False.

        Returns:
            None
        """
        self._options.bold_column_headers = bold
        self._options.italic_column_headers = italic

    def set_group_headers_text_format(self, bold: bool = True, italic: bool = False) -> None:
        """Set group-header emphasis styling.

        Args:
            bold (bool, optional): Whether to apply bold styling. Defaults to True.
            italic (bool, optional): Whether to apply italic styling. Defaults to False.

        Returns:
            None
        """
        self._options.bold_group_headers = bold
        self._options.italic_group_headers = italic

    def set_decimal_count(self, count: int) -> None:
        """Set the number of decimal places to round float values to in the LaTeX table.

        Args:
            count (int): Number of decimal places to round to.
        """
        if count < 0:
            raise ValueError("Decimal count must be non-negative")

        self._options.number_fmt_fn = round_decimals(count)

    def set_hrule_command(self, cmd: str) -> None:
        r"""Set the LaTeX command for horizontal rules in the table.

        Args:
            cmd (str): LaTeX command for horizontal rules (e.g., r"\hline").
        """
        self._options.hrule_cmd = cmd

    def set_toprule_command(self, cmd: str | None = None) -> None:
        """Set the LaTeX command for the top rule in the table.

        Args:
            cmd (str | None, optional): LaTeX command for the top rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        if cmd is None:
            self._options.toprule_cmd = self._options.hrule_cmd
        else:
            self._options.toprule_cmd = cmd

    def set_bottomrule_command(self, cmd: str | None = None) -> None:
        """Set the LaTeX command for the bottom rule in the table.

        Args:
            cmd (str | None, optional): LaTeX command for the bottom rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        if cmd is None:
            self._options.bottomrule_cmd = self._options.hrule_cmd
        else:
            self._options.bottomrule_cmd = cmd

    def set_all_hrule(self, count: int) -> None:
        """Set the number of horizontal rules above all rows in the LaTeX table.

        Args:
            count (int): Number of horizontal rules to add above each row. Zero clears the
                interior rules.

        Raises:
            ValueError: If ``count`` is negative.
        """
        if count < 0:
            raise ValueError(f"Rule count must be non-negative, got {count}.")
        self._options.hrule_counts = [count] * len(self.df) + [0]

    def set_nan_string(self, nan_str: str) -> None:
        """Set the string to represent NaN values in the LaTeX table.

        The value is inserted into the table as raw LaTeX (not escaped), so commands such as
        ``"---"`` or ``"\\textemdash"`` render as written; escape any literal special
        characters yourself.

        Args:
            nan_str (str): String to represent NaN values.
        """
        self._options.nan_string = nan_str

    def set_tabular_format(self, fmt: str) -> None:
        """Set the table-row tabular preamble.

        Supports simple and rich specifications (for example ``"|l|p{2cm}||S[table-
        format=1.3]|>{...}p{..}<{...}|"``) and preserves top-level boundary tokens for multicolumn
        rendering. Packages the colspecs require (``siunitx``, ``dcolumn``, ``array``) are
        registered on the document automatically.

        Note: within each column boundary, the ordering of ``|`` relative to ``@{...}``/``!{...}``
        tokens is normalized to a canonical emission order (``<{...}``, then ``|`` rules, then
        ``@{...}``/``!{...}``, then ``>{...}``), which can differ from the source format's spacing
        semantics.

        Args:
            fmt (str): Raw LaTeX tabular preamble string.

        Returns:
            None

        Raises:
            ValueError: If the parsed column count does not match the table width.
        """
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)

        expected_cols = self.df.shape[1] + (1 if self._options.include_index else 0)
        if len(colspecs) != expected_cols:
            raise ValueError(
                f"Format implies {len(colspecs)} columns but expected {expected_cols} "
                f"({'with' if self._options.include_index else 'without'} index)."
            )

        self._register_colspec_packages(colspecs, extras)
        parsed = TablePreamble.parsed(colspecs, vrules, extras)
        if not self._options.include_index:
            self._options.preamble = parsed
            return
        self._options.index_column = IndexColumn(
            parsed.alignments[0],
            parsed.boundaries[0],
            parsed.boundaries[1],
        )
        self._options.preamble = TablePreamble(
            parsed.alignments[1:],
            (TableBoundary(), *parsed.boundaries[2:]),
        )

    def set_group_tabular_format(self, fmt: str) -> None:
        """Set the group-header-row tabular preamble.

        Args:
            fmt (str): Raw LaTeX tabular preamble for group-header cells.

        Returns:
            None

        Raises:
            ValueError: If parsed group-header cell count is incompatible with current grouping.
        """
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)

        data_group_cells = sum(bool(columns) for columns in self._options.groups_to_cols.values())
        group_cells = data_group_cells + int(self._options.include_index)
        index_omitted = self._options.include_index and len(colspecs) == data_group_cells

        if len(colspecs) + int(index_omitted) != group_cells:
            index_note = (
                " A format one cell short is accepted as the data-group format and receives "
                "a leading centered index cell."
                if self._options.include_index
                else ""
            )
            raise ValueError(
                f"Group-header format implies {len(colspecs)} cells but expected {group_cells} "
                f"({'with' if self._options.include_index else 'without'} index).{index_note}"
            )

        self._register_colspec_packages(colspecs, extras)
        parsed = TablePreamble.parsed(colspecs, vrules, extras)
        if self._options.include_index:
            if index_omitted:
                index = self._options.index_column
                leading = parsed.boundaries[0]
                right = TableBoundary() if leading != TableBoundary() else index.right
                self._options.group_index = IndexColumn("c", index.left, right)
            else:
                self._options.group_index = IndexColumn(
                    parsed.alignments[0],
                    parsed.boundaries[0],
                    parsed.boundaries[1],
                )
                parsed = TablePreamble(
                    parsed.alignments[1:],
                    (TableBoundary(), *parsed.boundaries[2:]),
                )
        else:
            self._options.group_index = None
        self._options.group_preamble = parsed

    def clear_header_groups(self) -> None:
        """Clear any header groups set for the LaTeX table."""
        self._options.groups_to_cols = {"": list(self.df.columns)}

        self._options.group_preamble = None
        self._options.group_index = None

    def set_header_groups(
        self,
        groups_to_columns: dict[str, Iterable[Hashable]],
    ):
        """Set header groups for the LaTeX table.

        Column labels are matched against the DataFrame's columns as-is (no stringification), so
        non-string labels such as ints keep working with column formatters and cell lookup.

        Example:
            If the table has columns ["Col1", "Col2", "Col3"], then calling
            set_header_groups({"GroupA": ["Col1", "Col2"], "GroupB": ["Col3"]},
            "GroupA" spanning "Col1" and "Col2" with center alignment, and "GroupB"
            spanning "Col3" with left alignment.

        Args:
            groups_to_columns (dict[str, Iterable[Hashable]]): Mapping of group names to column
                labels.

        Raises:
            ValueError: If columns in groups_to_columns do not exist in the DataFrame or appear
                in multiple groups.
        """
        df_cols = list(self.df.columns)

        groups_to_cols: dict[str, list[Hashable]] = {
            str(grp): list(cols) for grp, cols in groups_to_columns.items()
        }

        all_listed: list[Hashable] = []
        for cols in groups_to_cols.values():
            all_listed.extend(cols)

        unknown = set(all_listed) - set(df_cols)
        if unknown:
            raise ValueError(f"Unknown columns in groups_to_columns: {sorted(unknown, key=repr)}")

        seen: set[Hashable] = set()
        duplicates: set[Hashable] = set()
        for column in all_listed:
            if column in seen:
                duplicates.add(column)
            seen.add(column)
        if duplicates:
            raise ValueError(
                f"Columns may appear in only one header group: {sorted(duplicates, key=repr)}"
            )

        # preserve DF column order: any unlisted columns go into the "" group (at the end)
        listed = set(all_listed)
        missing_cols = [column for column in df_cols if column not in listed]

        if "" in groups_to_cols:
            groups_to_cols[""].extend(missing_cols)
        elif len(missing_cols) > 0:
            groups_to_cols[""] = missing_cols

        self._options.groups_to_cols = groups_to_cols

        self._options.group_preamble = None
        self._options.group_index = None

    def _adapt(
        self,
        fmt_fn: Callable,
        *,
        one_arg_input: Literal["raw", "escaped"] = "raw",
    ) -> CellWrapper:
        """Register a formatter's LaTeX commands and adapt it to ``CellWrapper`` form.

        One-argument formatters are wrapped into the two-argument ``(value, prev) -> (value,
        rendered)`` shape. ``one_arg_input`` selects what the one-argument form receives: the raw
        cell value or the escaped rendering. Two-argument formatters are wrapped so a return that
        is not a 2-tuple raises instead of being silently mis-unpacked (a 2-character string
        unpacks like a pair, truncating the cell).

        Raises:
            TypeError: If the formatter cannot accept either supported call shape, or its
                signature cannot be introspected. Arity is never sniffed by trial call: that
                would swallow a genuine formatter's own TypeError and double-execute it.
        """
        self._register_formatter_latex_commands(fmt_fn)

        try:
            signature = inspect.signature(fmt_fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Cannot introspect the formatter's signature (functools.partial over C callables "
                "and some builtins do this). Wrap the formatter in a def or lambda with an "
                "explicit one- or two-parameter signature."
            ) from exc

        try:
            signature.bind(object())
        except TypeError:
            accepts_one = False
        else:
            accepts_one = True

        if accepts_one:
            one_arg = cast(Callable[[TableCellValue], str], fmt_fn)

            def _wrapped_one_arg(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
                return v, one_arg(s if one_arg_input == "escaped" else v)

            return _wrapped_one_arg

        try:
            signature.bind(object(), object())
        except TypeError as exc:
            raise TypeError(
                "Formatters must accept one positional argument (the value) or two "
                "(value, escaped text)."
            ) from exc

        two_arg = cast(CellWrapper, fmt_fn)

        def _checked_two_arg(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
            result = two_arg(v, s)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "Two-argument formatters must return a (value, rendered_text) 2-tuple; "
                    f"got {result!r}."
                )
            return result

        return _checked_two_arg

    def set_index_formatter(
        self, fmt_fn: IndexCellWrapper | Callable[[TableIndexValue], str]
    ) -> None:
        """Set a formatter function for the index column.

        A one-argument formatter receives the raw index value. A two-argument formatter
        receives ``(raw_value, escaped_text)`` and must return a ``(value, rendered_text)``
        2-tuple. Formatter output is inserted into the LaTeX table verbatim; escape raw content
        with :func:`gerrytools.latex.latex_escape`.

        Args:
            fmt_fn (IndexCellWrapper | Callable[[TableIndexValue], str]): Either a full
                two-argument formatter or a one-argument formatter over the raw index value.

        Returns:
            None
        """
        self._options.index_fmt_fn = self._adapt(fmt_fn)

    def set_number_formatter(self, fmt_fn: CellWrapper | Callable[[Real], str]) -> None:
        """Set the number formatter function for the LaTeX table.

        Used as the default formatter for all real-valued cells in the table.

        A one-argument formatter receives the raw numeric value. A two-argument formatter
        receives ``(raw_value, escaped_text)`` and must return a ``(value, rendered_text)``
        2-tuple. Formatter output is inserted into the LaTeX table verbatim; escape raw content
        with :func:`gerrytools.latex.latex_escape`.

        Args:
            fmt_fn (CellWrapper | Callable[[Real], str]): Formatter function for real-valued cells.
        """
        self._options.number_fmt_fn = self._adapt(fmt_fn)

    def set_string_formatter(self, fmt_fn: CellWrapper | Callable[[str], str]) -> None:
        """Set the string formatter function for the LaTeX table.

        Used as the default formatter for all string values in the table.

        A one-argument formatter receives the LaTeX-escaped cell text (not the raw value), so
        plain transformations such as ``str.upper`` compose with escaping. A two-argument
        formatter receives ``(raw_value, escaped_text)`` and must return a
        ``(value, rendered_text)`` 2-tuple. Formatter output is inserted into the LaTeX table
        verbatim; escape raw content with :func:`gerrytools.latex.latex_escape`.

        Args:
            fmt_fn (CellWrapper | Callable[[str], str]): Formatter function for string values.
        """
        self._options.str_fmt_fn = self._adapt(fmt_fn, one_arg_input="escaped")

    def set_column_formatter(
        self, col: Hashable | list[Hashable], fmt_fn: CellWrapper | Callable[[TableCellValue], str]
    ) -> None:
        """Set a specific column formatter function for the LaTeX table.

        A one-argument formatter receives the raw cell value. A two-argument formatter
        receives ``(raw_value, escaped_text)`` and must return a ``(value, rendered_text)``
        2-tuple. Formatter output is inserted into the LaTeX table verbatim; escape raw content
        with :func:`gerrytools.latex.latex_escape`. A cell covered by both a column and a row
        formatter uses the column formatter.

        Args:
            col (Hashable | list[Hashable]): Column label or list of column labels to set the
                formatter for.
            fmt_fn (CellWrapper | Callable[[TableCellValue], str]): Formatter function for the
                specified column(s).

        Raises:
            ValueError: If any of the specified columns do not exist in the DataFrame.
        """
        columns = col if isinstance(col, list) else [col]
        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' does not exist in DataFrame.")
            self._options.col_formatters[column] = self._adapt(fmt_fn)

    def set_row_formatter(
        self, row_idx: int | list[int], fmt_fn: CellWrapper | Callable[[TableCellValue], str]
    ) -> None:
        """Set a specific row formatter function for the LaTeX table.

        A one-argument formatter receives the raw cell value. A two-argument formatter
        receives ``(raw_value, escaped_text)`` and must return a ``(value, rendered_text)``
        2-tuple. Formatter output is inserted into the LaTeX table verbatim; escape raw content
        with :func:`gerrytools.latex.latex_escape`. A cell covered by both a column and a row
        formatter uses the column formatter.

        Args:
            row_idx (int | list[int]): Row index or list of row indices to set formatters for.
            fmt_fn (CellWrapper | Callable[[TableCellValue], str]): Formatter function for the
                specified row.

        Raises:
            ValueError: If any specified row index is out of bounds.
        """
        row_indices = row_idx if isinstance(row_idx, list) else [row_idx]
        for ridx in row_indices:
            if ridx < 0 or ridx >= len(self.df):
                raise ValueError(
                    f"Row index {ridx} is out of bounds for DataFrame with {len(self.df)} rows."
                )
            self._options.row_formatters[ridx] = self._adapt(fmt_fn)
