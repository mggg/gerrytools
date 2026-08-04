"""Tests for TexTable construction, formatting, rules, and LaTeX generation."""

from dataclasses import dataclass

import pandas as pd
import pytest

from gerrytools.latex._table_layout import TablePreamble
from gerrytools.latex.table import TexTable
from gerrytools.latex.tikz_table import TikzTable


@pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
def test_table_dialects_share_ordered_boundary_emission(table_cls):
    table = table_cls(pd.DataFrame({"a": [1], "b": [2]}), use_defaults=False)
    table.set_tabular_format(r"c>{\bfseries}@{}<{\arraybackslash}|!{\hspace{1pt}}c")

    assert table._column_format() == (r"c<{\arraybackslash}|@{}!{\hspace{1pt}}>{\bfseries}c")


def test_table_dialects_share_one_resolved_layout():
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    tex = TexTable(df, use_defaults=False)
    tikz = TikzTable(df, use_defaults=False)

    for table in (tex, tikz):
        table.include_index(name="ID")
        table.set_header_groups({"Numbers": ["A"], "Labels": ["B"]})
        table.add_vrule_left_of([0, 2])
        table.add_hrule_above(1, count=2)
        table.highlight_rows(0, color="amber")
        table.set_column_headers_text_format(bold=True)

    assert tex._resolve_layout() == tikz._resolve_layout()


# ===========================
# == TEXTABLE CONSTRUCTION ==
# ===========================
class TestTexTableConstruction:
    def test_default_and_clear_options_are_the_same(self, df):
        table_1 = TexTable(df)
        table_1.clear_options()
        table_2 = TexTable(df, use_defaults=False)

        assert str(table_1) == str(table_2)

    def test_include_index_does_not_mutate_data_preamble(self, table_defaults):
        table = table_defaults
        before = table._options.preamble

        table.include_index(include=True)

        assert table._options.preamble is before
        assert len(table._resolved_preamble().alignments) == len(table.df.columns) + 1

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_remove_index_preserves_configuration(self, table_cls):
        table = table_cls(pd.DataFrame({"value": [1]}), use_defaults=False)
        table.include_index(name="ID", alignment="|r|")

        table.remove_index()
        table.include_index()

        assert table._options.index_name == "ID"
        assert table._column_format() == "|r|c"

    def test_string_ops_idempotent(self, table_defaults):
        orig_str = str(table_defaults)
        table_defaults.include_index(include=True)
        table_defaults.remove_index()
        assert str(table_defaults) == orig_str

    def test_document_property_updates_body_string(self, table_defaults):
        table = table_defaults

        document = table.document

        assert document.body_string in str(table)

    def test_mixed_numeric_columns_preserve_cell_dtypes(self):
        table = TexTable(pd.DataFrame({"count": [1, 2], "share": [0.5, 1.5]}), use_defaults=False)

        assert "1 & 0.5" in table._generate_latex()


# ================================
# == TEXTABLE VALIDATION ERRORS ==
# ================================
class TestTexTableValidationErrors:
    def test_mismatched_preamble_rejected_during_resolution(self, table_defaults):
        table = table_defaults
        n_data_cols = table.df.shape[1]
        table._options.preamble = TablePreamble.plain(n_data_cols + 1)

        with pytest.raises(
            ValueError,
            match=r"Current tabular format does not match DataFrame columns\.",
        ):
            table._resolve_layout()

    def test_add_hrule_above_negative_index_raises(self, table_defaults):
        table = table_defaults
        with pytest.raises(
            ValueError,
            match=r"Row index -1 is out of bounds for DataFrame with \d+ rows\.",
        ):
            table.add_hrule_above(-1)

    def test_add_hrule_above_index_too_large_raises(self, table_defaults):
        table = table_defaults
        too_big = len(table.df) + 1
        with pytest.raises(
            ValueError,
            match=rf"Row index {too_big} is out of bounds for DataFrame with {len(table.df)} rows\.",
        ):
            table.add_hrule_above(too_big)

    def test_add_vrule_left_of_negative_index_raises(self, table_defaults):
        table = table_defaults
        with pytest.raises(
            ValueError,
            match=r"Column index -1 is out of bounds for DataFrame with \d+ columns\.",
        ):
            table.add_vrule_left_of(-1)

    def test_add_vrule_left_of_index_too_large_raises(self, table_defaults):
        table = table_defaults
        include_index_offset = int(table._options.include_index)
        too_big = len(table.df.columns) + include_index_offset + 1

        with pytest.raises(
            ValueError,
            match=rf"Column index {too_big} is out of bounds for DataFrame with "
            rf"{len(table.df.columns)} columns\.",
        ):
            table.add_vrule_left_of(too_big)

    @pytest.mark.parametrize("index", [-2, -1])
    def test_add_vrule_right_of_negative_index_raises(self, table_defaults, index):
        table = table_defaults
        with pytest.raises(
            ValueError,
            match=rf"Column index {index} is out of bounds for DataFrame with \d+ columns\.",
        ):
            table.add_vrule_right_of(index)

    def test_add_vrule_right_of_index_too_large_raises(self, table_defaults):
        table = table_defaults
        include_index_offset = int(table._options.include_index)
        too_big = len(table.df.columns) + include_index_offset

        with pytest.raises(
            ValueError,
            match=rf"Column index {too_big} is out of bounds for DataFrame with "
            rf"{len(table.df.columns)} columns\.",
        ):
            table.add_vrule_right_of(too_big)

    def test_highlight_rows_invalid_rgb_range_raises(self, table_defaults):
        table = table_defaults
        bad_color = (-1, 0, 0)

        with pytest.raises(
            ValueError,
            match=r"RGB color components must be in the range \[0\.0, 1\.0\] or \[0, 255\]",
        ):
            table.highlight_rows(0, color=bad_color)

    def test_highlight_rows_invalid_color_spec_raises(self, table_defaults):
        table = table_defaults
        with pytest.raises(
            ValueError,
            match=r"must be a LaTeX color name, HEX string, or RGB tuple",
        ):
            table.highlight_rows(0, color=(1, 2))

    def test_highlight_rows_rejects_none_before_rowcolor_emission(self, table_defaults):
        with pytest.raises(ValueError, match="cannot be emitted"):
            table_defaults.highlight_rows(0, color="none")

    @pytest.mark.parametrize("count", [0, -3])
    def test_add_hrule_above_rejects_non_positive_count(self, table_defaults, count):
        # "\hline" * -3 == "" would silently cancel later additions.
        with pytest.raises(ValueError, match=r"Rule count must be at least 1"):
            table_defaults.add_hrule_above(0, count=count)

    @pytest.mark.parametrize("count", [0, -3])
    def test_add_hrule_above_all_rejects_non_positive_count(self, table_defaults, count):
        with pytest.raises(ValueError, match=r"Rule count must be at least 1"):
            table_defaults.add_hrule_above_all(count=count)

    @pytest.mark.parametrize("count", [0, -3])
    def test_add_vrule_left_of_rejects_non_positive_count(self, table_defaults, count):
        with pytest.raises(ValueError, match=r"Rule count must be at least 1"):
            table_defaults.add_vrule_left_of(0, count=count)

    @pytest.mark.parametrize("count", [0, -3])
    def test_add_vrule_right_of_rejects_non_positive_count(self, table_defaults, count):
        with pytest.raises(ValueError, match=r"Rule count must be at least 1"):
            table_defaults.add_vrule_right_of(0, count=count)

    @pytest.mark.parametrize("count", [0, -3])
    def test_add_vrule_all_rejects_non_positive_count(self, table_defaults, count):
        with pytest.raises(ValueError, match=r"Rule count must be at least 1"):
            table_defaults.add_vrule_all(count=count)

    def test_set_all_hrule_rejects_negative_count(self, table_defaults):
        # Zero stays legal for set_all_hrule: it clears the interior rules.
        with pytest.raises(ValueError, match=r"Rule count must be non-negative"):
            table_defaults.set_all_hrule(-1)

    def test_set_decimal_count_negative_raises(self, table_defaults):
        table = table_defaults
        with pytest.raises(
            ValueError,
            match=r"Decimal count must be non-negative",
        ):
            table.set_decimal_count(-1)

    def test_set_tabular_format_column_count_mismatch_raises(self, table_defaults):
        table = table_defaults
        fmt = "c" * max(1, table.df.shape[1] - 1)

        with pytest.raises(
            ValueError,
            match=r"Format implies \d+ columns but expected \d+ ",
        ):
            table.set_tabular_format(fmt)

    def test_set_group_tabular_format_cell_count_mismatch_raises(self, table_defaults):
        table = table_defaults
        fmt = "cc"

        with pytest.raises(
            ValueError,
            match=r"Group-header format implies \d+ cells but expected \d+",
        ):
            table.set_group_tabular_format(fmt)

    def test_set_header_groups_unknown_columns_raises(self, table_defaults):
        table = table_defaults

        with pytest.raises(
            ValueError,
            match=r"Unknown columns in groups_to_columns: \['not_a_col'\]",
        ):
            table.set_header_groups({"GroupA": ["not_a_col"]})

    def test_set_column_formatter_unknown_column_raises(self, table_defaults):
        table = table_defaults

        def fmt(v, s):
            return v, s

        with pytest.raises(
            ValueError,
            match=r"Column 'not_a_col' does not exist in DataFrame\.",
        ):
            table.set_column_formatter("not_a_col", fmt)

    def test_set_row_formatter_row_too_large_raises(self, table_defaults):
        table = table_defaults

        def fmt(v, s):
            return v, s

        bad_idx = len(table.df)

        with pytest.raises(
            ValueError,
            match=rf"Row index {bad_idx} is out of bounds for DataFrame with {len(table.df)} rows\.",
        ):
            table.set_row_formatter(bad_idx, fmt)

    def test_set_row_formatter_negative_index_raises(self, table_defaults):
        table = table_defaults

        def fmt(v, s):
            return v, s

        with pytest.raises(
            ValueError,
            match=r"Row index -1 is out of bounds for DataFrame with \d+ rows\.",
        ):
            table.set_row_formatter(-1, fmt)


# ======================
# == TEXTABLE HEADERS ==
# ======================
class TestTexTableHeaders:
    def test_highlight_rows_accepts_iterable_indices(self, table_defaults):
        table = table_defaults

        table.highlight_rows([0, 2], color="amber")

        assert table._options.row_highlight_colors[0] == ("NAME", "amber")
        assert table._options.row_highlight_colors[2] == ("NAME", "amber")

    def test_header_toggle_helpers_flip_all_header_flags(self, table_defaults):
        table = table_defaults

        table.remove_column_headers()
        assert table._options.include_column_headers is False

        table.remove_group_headers()
        assert table._options.include_group_headers is False

        table.include_column_headers()
        assert table._options.include_column_headers is True

        table.include_group_headers()
        assert table._options.include_group_headers is True

        table.remove_all_headers()
        assert table._options.include_column_headers is False
        assert table._options.include_group_headers is False

        table.include_all_headers()
        assert table._options.include_column_headers is True
        assert table._options.include_group_headers is True

    def test_generate_header_can_italicize_index_name(self, table_defaults):
        table = table_defaults
        table.include_index(include=True, name="Index")
        table.set_column_headers_text_format(bold=False, italic=True)

        header = table._generate_header()

        assert r"\textit{Index}" in header

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_index_header_name_is_escaped_once(self, table_cls):
        table = table_cls(pd.DataFrame({"a": [1]}), use_defaults=False)
        table.include_index(name="A_B & C")

        header = table.document.body_string

        assert r"A\_B \& C" in header
        assert r"\textbackslash" not in header

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_column_headers_escape_special_characters_once(self, table_cls):
        table = table_cls(pd.DataFrame({"Vote %": [1], "Win & Loss_Rate": [2]}), use_defaults=False)

        body = table.document.body_string

        assert r"Vote \%" in body
        assert r"Win \& Loss\_Rate" in body
        assert r"\textbackslash" not in body

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_group_headers_escape_special_characters_once(self, table_cls):
        table = table_cls(pd.DataFrame({"a": [1], "b": [2]}), use_defaults=False)
        table.set_header_groups({"Pct % & Rank_1": ["a", "b"]})

        body = table.document.body_string

        assert r"Pct \% \& Rank\_1" in body
        assert r"\textbackslash" not in body

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_unnamed_index_header_falls_back_to_dataframe_name(self, table_cls):
        frame = pd.DataFrame({"a": [1]})
        frame.index.name = "Row_ID"
        table = table_cls(frame, use_defaults=False)
        table.include_index()

        assert r"Row\_ID" in table.document.body_string

    def test_generate_header_with_groups_and_index(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        table = TexTable(df)

        table.include_index(name="idx", include=True)
        table.set_header_groups({"G1": ["A"], "G2": ["B"]})
        table.add_toprule()

        header = table._generate_header()

        assert header.startswith("\\begin{tabular}{ccc}\n\\hline\n")
        assert (
            r"\multicolumn{1}{c}{} & \multicolumn{1}{c}{\textbf{G1}} & "
            r"\multicolumn{1}{c}{\textbf{G2}} \\" in header
        )
        assert r"\textbf{idx} & \textbf{A} & \textbf{B} \\" in header

    def test_set_column_and_group_header_text_format(self, table_defaults):
        table = table_defaults

        table.set_column_headers_text_format(bold=True, italic=True)
        table.set_group_headers_text_format(bold=False, italic=True)

        header = table._generate_header()
        assert r"\textbf{" in header
        assert r"\textit{" in header

    def test_set_header_groups_with_missing_cols_and_blank_group_key(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        table = TexTable(df)

        table.set_header_groups(
            {
                "G1": ["a"],
                "": ["b"],
            }
        )
        g2c = table._options.groups_to_cols
        assert g2c[""][-1] == "c"

    def test_set_header_groups_with_missing_cols_and_no_blank_group(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        table = TexTable(df)

        table.set_header_groups({"G1": ["a"]})
        g2c = table._options.groups_to_cols

        assert g2c["G1"] == ["a"]
        assert g2c[""] == ["b", "c"]

    def test_set_header_groups_covering_all_cols_no_missing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        table = TexTable(df)

        table.set_header_groups({"G1": ["a"], "G2": ["b"]})
        g2c = table._options.groups_to_cols

        assert set(g2c.keys()) == {"G1", "G2"}
        assert "" not in g2c

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_header_group_reordering_moves_column_preamble(self, table_cls):
        table = table_cls(pd.DataFrame({"A": [1], "B": [2], "C": [3]}), use_defaults=False)
        table.set_tabular_format("lcr")
        table.add_vrule_right_of(0)
        table.set_header_groups({"G1": ["C"], "G2": ["A", "B"]})

        layout = table._resolve_layout()

        assert layout.preamble.alignments == ("r", "l", "c")
        assert [boundary.vrules for boundary in layout.preamble.boundaries] == [0, 1, 0, 0]

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_set_header_groups_keeps_raw_non_string_column_labels(self, table_cls):
        # Regression: the setter stringified labels, so rendering raised KeyError: '1' and
        # int-keyed column formatters silently unlinked.
        df = pd.DataFrame({1: [0.5], 2: [0.25], "name": ["x"]})
        table = table_cls(df, use_defaults=False)
        table.set_column_formatter(1, lambda value, rendered: (value, f"FMT={rendered}"))
        table.set_header_groups({"Numbers": [1, 2]})

        body = table.document.body_string

        assert "FMT=0.5" in body
        assert table._options.groups_to_cols["Numbers"] == [1, 2]
        assert table._options.groups_to_cols[""] == ["name"]

    def test_set_header_groups_rejects_columns_in_multiple_groups(self):
        table = TexTable(pd.DataFrame({"a": [1], "b": [2]}))

        with pytest.raises(ValueError, match=r"only one header group.*\['a'\]"):
            table.set_header_groups({"G1": ["a"], "G2": ["a", "b"]})

    def test_clear_header_groups_resets_state(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        table = TexTable(df)

        table.set_header_groups({"G1": ["a"], "G2": ["b"]})
        table.set_group_tabular_format("cc")
        table.clear_header_groups()

        assert table._options.groups_to_cols == {"": ["a", "b"]}
        assert table._options.group_preamble is None
        assert table._options.group_index is None


# ====================
# == TEXTABLE RULES ==
# ====================
class TestTexTableRules:
    def test_add_hrule_above_writes_into_the_eagerly_sized_vector(self, table_defaults):
        table = table_defaults

        table.clear_all_hrule()
        assert table._options.hrule_counts == [0] * (len(table.df) + 1)

        table.add_hrule_above(0, count=2)

        assert table._options.hrule_counts[0] == 2

    def test_add_hrule_above_bottom_boundary_emits_trailing_rule(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        table = TexTable(df, use_defaults=False)

        table.add_hrule_above(len(df), count=1)
        body = table._generate_body()

        assert body.rstrip().endswith(r"\hline")

    def test_add_hrule_above_all_increments_every_row(self, table_defaults):
        table = table_defaults

        table.clear_all_hrule()
        table.add_hrule_above_all(count=3)

        assert table._options.hrule_counts == [3] * len(table.df) + [0]

    def test_clear_all_hrule(self, table_defaults):
        table = table_defaults

        table.add_toprule()
        table.add_bottomrule()
        table.add_hrule_above_all(count=1)
        assert any(c > 0 for c in table._options.hrule_counts)

        table.clear_all_hrule()

        assert table._options.hrule_counts == [0] * (len(table.df) + 1)
        assert table._options.toprule_cmd is None
        assert table._options.bottomrule_cmd is None
        assert table._options.hrule_cmd not in table._generate_latex()

    def test_add_toprule_default_and_remove(self, table_defaults):
        table = table_defaults

        table._options.hrule_cmd = r"\midrule"
        table._options.toprule_cmd = None

        table.add_toprule()
        assert table._options.toprule_cmd == r"\midrule"

        header = table._generate_header()
        assert r"\midrule" in header

        table.remove_toprule()
        assert table._options.toprule_cmd is None

    def test_add_bottomrule_default_and_remove(self, table_defaults):
        table = table_defaults

        table._options.hrule_cmd = r"\midrule"
        table._options.bottomrule_cmd = None

        table.add_bottomrule()
        assert table._options.bottomrule_cmd == r"\midrule"

        footer = table._generate_footer()
        assert r"\midrule" in footer

        table.remove_bottomrule()
        assert table._options.bottomrule_cmd is None

    def test_add_vrule_left_of_updates_boundary(self, table_defaults):
        table = table_defaults
        ncols = len(table.df.columns)
        include_index_offset = int(table._options.include_index)

        table.add_vrule_left_of(0, count=2)

        preamble = table._resolved_preamble()
        assert len(preamble.boundaries) == ncols + 1 + include_index_offset
        assert preamble.boundaries[0].vrules == 2

    def test_clear_all_vrule_sets_correct_length(self, table_defaults):
        table = table_defaults

        table.include_index(include=True)
        table.clear_all_vrule()

        expected_len = len(table.df.columns) + 1 + int(table._options.include_index)
        boundaries = table._resolved_preamble().boundaries
        assert len(boundaries) == expected_len
        assert all(boundary.vrules == 0 for boundary in boundaries)

    def test_add_vrule_right_of_updates_boundary(self, table_defaults):
        table = table_defaults
        ncols = len(table.df.columns)
        include_index_offset = int(table._options.include_index)

        table.add_vrule_right_of(0, count=3)

        preamble = table._resolved_preamble()
        assert len(preamble.boundaries) == ncols + 1 + include_index_offset
        assert preamble.boundaries[1].vrules == 3

    def test_add_vrule_all_increments_every_boundary(self, table_defaults):
        table = table_defaults

        table.add_vrule_all(count=1)

        total_cols = len(table.df.columns) + int(table._options.include_index)
        assert [boundary.vrules for boundary in table._resolved_preamble().boundaries] == [1] * (
            total_cols + 1
        )

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_add_vrule_all_before_include_index_keeps_outer_rule(self, table_cls):
        table = table_cls(pd.DataFrame({"a": [1], "b": [2]}), use_defaults=False)

        table.add_vrule_all()
        table.include_index()

        assert table._column_format() == "|c|c|c|"

    def test_add_rule_methods_accept_list_inputs(self, table_defaults):
        table = table_defaults

        table.add_hrule_above([1, 3], count=2)
        table.add_vrule_left_of([1, 3], count=2)
        table.add_vrule_right_of([0, 2], count=2)

        assert table._options.hrule_counts[1] == 2
        assert table._options.hrule_counts[3] == 2
        boundaries = table._resolved_preamble().boundaries
        assert boundaries[1].vrules == 4
        assert boundaries[3].vrules == 4

    def test_toprule_and_bottomrule_default_to_hrule_command(self, table_defaults):
        table = table_defaults
        table.set_hrule_command(r"\specialrule")

        table.set_toprule_command()
        table.set_bottomrule_command()

        assert table._options.toprule_cmd == r"\specialrule"
        assert table._options.bottomrule_cmd == r"\specialrule"

    def test_set_hrule_and_all_hrule(self, table_defaults):
        table = table_defaults

        table.set_hrule_command(r"\hline")
        table.set_all_hrule(2)

        assert table._options.hrule_cmd == r"\hline"
        assert table._options.hrule_counts == [2] * len(table.df) + [0]


# ===========================
# == TEXTABLE HIGHLIGHTING ==
# ===========================
class TestTexTableHighlighting:
    def test_highlight_rows_name_color_and_generate_body(self, table_defaults):
        table = table_defaults
        table.highlight_rows(0, color="yellow")

        body = table._generate_body()
        assert r"\rowcolor{yellow}" in body

    def test_highlight_rows_xcolor_expression_preserved(self, table_defaults):
        table = table_defaults
        table.highlight_rows(0, color="amber!80!gray")

        assert table._options.row_highlight_colors[0] == ("NAME", "amber!80!gray")

        body = table._generate_body()
        assert r"\rowcolor{amber!80!gray}" in body

    def test_highlight_rows_non_latex_named_color_converts_to_html(self, table_defaults):
        table = table_defaults
        table.highlight_rows(0, color="tab:blue")

        color_type, color_value = table._options.row_highlight_colors[0]
        assert color_type == "HTML"
        assert isinstance(color_value, str)
        assert len(color_value) == 6

        body = table._generate_body()
        assert r"\rowcolor[HTML]{" in body

    def test_highlight_rows_stores_only_highlighted_rows(self):
        df = pd.DataFrame({"a": [1, 2]})
        table = TexTable(df)

        table.highlight_rows(1, color="yellow")

        assert table._options.row_highlight_colors == {1: ("NAME", "yellow")}

    def test_highlight_rows_out_of_bounds_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        table = TexTable(df)

        with pytest.raises(ValueError, match="out of bounds"):
            table.highlight_rows(2, color="yellow")

    def test_highlight_rows_hex_color_and_generate_body(self, table_defaults):
        table = table_defaults
        table.highlight_rows(0, color="#abcdef")

        body = table._generate_body()
        assert r"\rowcolor[HTML]{ABCDEF}" in body

    def test_highlight_rows_rgb_float_and_generate_body(self):
        # Use our own one-row df for clean expectations
        df = pd.DataFrame({"a": [1.0]})
        table = TexTable(df)

        table.highlight_rows(0, color=(0.1, 0.2, 0.3))

        body = table._generate_body()
        # formatted to 3 decimals
        assert r"\rowcolor[rgb]{0.100,0.200,0.300}" in body

    def test_highlight_rows_RGB_int_and_generate_body(self):
        df = pd.DataFrame({"a": [1.0]})
        table = TexTable(df)

        table.highlight_rows(0, color=(10, 20, 30))

        body = table._generate_body()
        assert r"\rowcolor[RGB]{10,20,30}" in body


# =========================
# == TEXTABLE GENERATION ==
# =========================
class TestTexTableGeneration:
    def test_reassigning_dataframe_with_different_shape_fails_fast(self):
        table = TexTable(pd.DataFrame({"value": [1, 2]}))
        table.add_hrule_above_all()

        with pytest.raises(ValueError, match="same rows and columns"):
            table.df = pd.DataFrame({"value": [1, 2, 3, 4]})

    def test_generate_body_uses_group_ordering_and_raw_index_when_no_index_formatter(self, df):
        table = TexTable(df)
        table.include_index(include=True)
        table.set_number_formatter(lambda v: f"{v:.3f}")
        table.set_header_groups(
            {
                "Group A": ["Column 3", "Column 1"],
                "Group B": ["Column 2", "Column 4", "Column 5", "Names"],
            }
        )

        first_row = next(
            line for line in table._generate_body().splitlines() if not line.startswith(r"\hline")
        )

        assert first_row.startswith("0 & 0.369 & 0.803")

    def test_generate_body_with_hrule_and_various_paths(self):
        df = pd.DataFrame(
            {
                "f": [1.23, 4.56],
                "s": ["x&y", "z"],
            }
        )
        table = TexTable(df)

        table.set_all_hrule(0)
        table.add_hrule_above(0, count=1)

        table.set_column_formatter("f", lambda v, s: (v, f"F={s}"))
        table.set_row_formatter(1, lambda v, s: (v, f"R={s}"))
        table.set_string_formatter(lambda s: s.replace("&", r"\&"))

        body = table._generate_body()

        assert "F=1.23" in body or "F=1.2300" in body
        assert "R=z" in body
        assert r"\&" in body

    def test_generate_footer_and_full_latex(self, table_defaults):
        table = table_defaults
        table.add_bottomrule()

        footer = table._generate_footer()
        assert r"\end{tabular}" in footer

        full = table._generate_latex()
        assert full.startswith(r"\begin{tabular}")
        assert full.strip().endswith(r"\end{tabular}")


# =====================
# == DEGENERATE SHAPES ==
# =====================
class TestDegenerateDataFrameShapes:
    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_zero_row_dataframe_resolves_and_renders(self, table_cls):
        table = table_cls(pd.DataFrame({"a": [], "b": []}))

        layout = table._resolve_layout()
        assert [row.kind for row in layout.rows] == ["columns"]
        assert len(layout.preamble.alignments) == 2

        body = table.document.body_string
        assert "a" in body and "b" in body

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_zero_column_dataframe_is_rejected(self, table_cls):
        with pytest.raises(ValueError, match="at least one column"):
            table_cls(pd.DataFrame(index=pd.Index([0, 1])))

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    def test_duplicate_dataframe_columns_are_rejected(self, table_cls):
        frame = pd.DataFrame([[1, 2]], columns=pd.Index(["a", "a"]))
        with pytest.raises(ValueError, match="unique"):
            table_cls(frame)


# ===================================
# == COLSPEC PACKAGE REGISTRATION ==
# ===================================
class TestColspecPackageRegistration:
    """Rich colspec tokens register their packages at set_tabular_format parse time.

    The document's macro scan cannot recognize these tokens inside an emitted
    ``\\begin{tabular}{...}`` line, so the setters register them directly.
    """

    @pytest.mark.parametrize("table_cls", [TexTable, TikzTable])
    @pytest.mark.parametrize(
        "fmt, package",
        [
            (r"S[table-format=1.4]c", "siunitx"),
            (r"D{.}{.}{-1}c", "dcolumn"),
            (r"c!{\hspace{4pt}}c", "array"),
            (r"c<{\hspace{2pt}}c", "array"),
            (r"m{1cm}c", "array"),
        ],
    )
    def test_set_tabular_format_registers_required_packages(self, table_cls, fmt, package):
        table = table_cls(pd.DataFrame({"a": [1.25], "b": [2.5]}), use_defaults=False)
        table.set_tabular_format(fmt)

        assert package in table.document.preamble

    def test_include_index_alignment_registers_required_packages(self):
        table = TexTable(pd.DataFrame({"a": [1.25]}), use_defaults=False)
        table.include_index(alignment=r"!{\hspace{4pt}}c")

        assert "array" in table.document.preamble

    def test_set_group_tabular_format_registers_required_packages(self):
        table = TexTable(pd.DataFrame({"a": [1.25], "b": [2.5]}), use_defaults=False)
        table.set_header_groups({"G": ["a", "b"]})
        table.set_group_tabular_format(r"!{\hspace{4pt}}c")

        assert "array" in table.document.preamble

    @pytest.mark.latex
    @pytest.mark.parametrize(
        "fmt",
        [r"S[table-format=1.4]c", r"D{.}{.}{-1}c", r"c!{\hspace{4pt}}c"],
    )
    def test_rich_colspec_documents_compile(self, fmt):
        table = TexTable(pd.DataFrame({"a": [1.25, 3.5], "b": [2.5, 4.75]}), use_defaults=False)
        table.set_tabular_format(fmt)
        # S and D columns parse cell content as numbers; plain text headers would need braces.
        table.remove_column_headers()

        table.document._compile_pdf()

    @pytest.mark.latex
    def test_tikz_header_rule_strut_compiles_in_siunitx_column(self):
        table = TikzTable(pd.DataFrame({"label": ["x"], "2": [2.5]}))
        table.set_tabular_format(r"cS[table-format=1.2]")

        table.document._compile_pdf()


# ==================================
# —————— OBJECT-CELL ESCAPING ——————
# ==================================
@dataclass(frozen=True)
class _ObjectCell:
    text: str

    def __str__(self) -> str:
        return self.text


class TestTexTableObjectCellEscaping:
    def test_object_cells_escape_special_characters_once(self):
        # Regression: the fallback branch escaped, then fed the escaped string to the default
        # string formatter, which escaped again (& -> \& -> \textbackslash{}\&).
        df = pd.DataFrame({"col": [_ObjectCell("A & B_C")]})
        latex = str(TexTable(df))
        assert r"A \& B\_C" in latex
        assert r"\textbackslash" not in latex

    def test_object_cells_escape_once_without_string_formatter(self):
        df = pd.DataFrame({"col": [_ObjectCell("A & B_C")]})
        table = TexTable(df)
        table._options.str_fmt_fn = None
        latex = str(table)
        assert r"A \& B\_C" in latex
        assert r"\textbackslash" not in latex
