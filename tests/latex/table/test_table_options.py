"""Tests for TexTable column format and multicolumn format generation."""

import pandas as pd
import pytest

from gerrytools.latex._table_layout import IndexColumn, TablePreamble, column_format
from gerrytools.latex.table import TableOptions, TexTable


def options(
    *,
    tabular_alignments: list[str],
    vrule_counts: list[int],
    boundary_extras: list[str] | None = None,
    group_tabular_alignments: list[str] | None = None,
    group_vrule_counts: list[int] | None = None,
    group_boundary_extras: list[str] | None = None,
    index_alignment: str = "c",
    **kwargs,
) -> TableOptions:
    """Build options through the canonical preamble values used by production."""
    preamble = TablePreamble.parsed(
        tabular_alignments,
        vrule_counts,
        boundary_extras or [""] * len(vrule_counts),
    )
    group_preamble = None
    if group_tabular_alignments is not None:
        assert group_vrule_counts is not None
        group_preamble = TablePreamble.parsed(
            group_tabular_alignments,
            group_vrule_counts,
            group_boundary_extras or [""] * len(group_vrule_counts),
        )
    return TableOptions(
        preamble=preamble,
        group_preamble=group_preamble,
        index_column=IndexColumn(index_alignment),
        **kwargs,
    )


def table_for_options(opts: TableOptions) -> TexTable:
    columns = [
        column for group_columns in opts.groups_to_cols.values() for column in group_columns
    ] or [f"c{index}" for index in range(len(opts.preamble.alignments))]
    table = TexTable(pd.DataFrame(columns=pd.Index(columns)), use_defaults=False)
    table._options = opts
    return table


def render_column_format(opts: TableOptions) -> str:
    # The pure layout function; no throwaway table needed (no index column in these cases).
    return column_format(opts.preamble)


def render_multicolumn_format(opts: TableOptions) -> str:
    table = table_for_options(opts)
    return table._multicolumn_format()


# ===================
# == COLUMN FORMAT ==
# ===================
class TestColumnFormat:
    def test_column_format_basic_empty_boundary_extras(self):
        opts = options(
            tabular_alignments=["l", "c", "r"],
            vrule_counts=[1, 0, 2, 0],
            boundary_extras=["", "", "", ""],
        )

        expected = "|lc||r"
        assert render_column_format(opts) == expected

    def test_column_format_with_boundary_extras_and_vrules(self):
        opts = options(
            tabular_alignments=["c", "c"],
            vrule_counts=[1, 0, 1],  # 3 boundaries
            boundary_extras=[r"@{}", r"!{\quad}", r"<{\arraybackslash}"],
        )

        expected = r"|@{}c!{\quad}c<{\arraybackslash}|"
        assert render_column_format(opts) == expected


# ===============================
# == MULTICOLUMN FORMAT MODE A ==
# ===============================
class TestMulticolumnFormatModeA:
    def test_multicolumn_format_mode_a_simple_no_index(self):
        opts = options(
            tabular_alignments=["l"],
            vrule_counts=[1, 0],
            groups_to_cols={"Group1": ["col1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[1, 0],
            group_boundary_extras=["", ""],
        )

        out = render_multicolumn_format(opts)
        assert out == r"\multicolumn{1}{|c}{\textbf{Group1}} \\"

    def test_multicolumn_format_mode_a_with_index(self):
        opts = options(
            include_index=True,
            index_alignment="l",
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["col1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[1, 0],
            group_boundary_extras=["", ""],
        )

        out = render_multicolumn_format(opts)

        expected = r"\multicolumn{1}{l|}{}" r" & " r"\multicolumn{1}{c}{\textbf{Group1}} \\"
        assert out == expected

    def test_multicolumn_format_mode_a_bold_and_italic_group_headers(self):
        opts = options(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["c1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["", ""],
            bold_group_headers=True,
            italic_group_headers=True,
        )

        out = render_multicolumn_format(opts)

        assert r"\textit{\textbf{Group1}}" in out

    def test_multicolumn_format_mode_a_no_bold_no_italic(self):
        opts = options(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["c1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["", ""],
            bold_group_headers=False,
            italic_group_headers=False,
        )

        out = render_multicolumn_format(opts)

        assert "Group1" in out
        assert r"\textbf{Group1}" not in out
        assert r"\textit{Group1}" not in out

    def test_multicolumn_format_mode_a_span_zero_group_is_skipped(self):
        opts = options(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Empty": [], "Full": ["c1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["", ""],
        )

        out = render_multicolumn_format(opts)

        assert "Empty" not in out
        assert r"\textbf{Full}" in out
        assert out.count(r"\multicolumn{1}") == 1

    def test_multicolumn_format_mode_a_italic_only_group_headers(self):
        opts = options(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["c1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["", ""],
            bold_group_headers=False,
            italic_group_headers=True,
        )

        out = render_multicolumn_format(opts)

        assert r"\textit{Group1}" in out
        assert r"\textbf{Group1}" not in out

    def test_multicolumn_format_mode_a_group_header_cell_count_mismatch_raises(self):
        opts = options(
            tabular_alignments=["l", "c"],
            vrule_counts=[0, 0, 0],
            groups_to_cols={"G1": ["c1"], "G2": ["c2"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["", ""],
        )

        with pytest.raises(
            ValueError,
            match=r"Group-header preamble has 1 cells but expected 2\.",
        ):
            _ = render_multicolumn_format(opts)


# ===============================
# == MULTICOLUMN FORMAT MODE B ==
# ===============================
class TestMulticolumnFormatModeB:
    def test_multicolumn_format_mode_b_simple_no_index(self):
        opts = options(
            tabular_alignments=["l", "c", "r"],
            vrule_counts=[1, 0, 0, 0],
            groups_to_cols={"G1": ["c1", "c2"], "G2": ["c3"]},
        )

        out = render_multicolumn_format(opts)

        expected = r"\multicolumn{2}{|c}{\textbf{G1}}" r" & " r"\multicolumn{1}{r}{\textbf{G2}} \\"

        assert out == expected

    def test_multicolumn_format_mode_b_italic_only_group_headers(self):
        opts = options(
            tabular_alignments=["c"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["c1"]},
            bold_group_headers=False,
            italic_group_headers=True,
        )

        out = render_multicolumn_format(opts)

        expected = r"\multicolumn{1}{c}{\textit{Group1}} \\"
        assert out == expected

    def test_multicolumn_format_mode_b_bold_and_italic_group_headers(self):
        opts = options(
            tabular_alignments=["c"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["c1"]},
            bold_group_headers=True,
            italic_group_headers=True,
        )

        out = render_multicolumn_format(opts)

        expected = r"\multicolumn{1}{c}{\textit{\textbf{Group1}}} \\"
        assert out == expected

    def test_multicolumn_format_mode_b_span_zero_group_is_skipped(self):
        opts = options(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Empty": [], "Used": ["c1"]},
        )

        out = render_multicolumn_format(opts)

        assert "Empty" not in out
        assert r"\textbf{Used}" in out
        assert out.count(r"\multicolumn{1}") == 1
