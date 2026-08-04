"""Tests for the nicematrix-based TikzTable LaTeX generation."""

import pandas as pd
import pytest

from gerrytools.latex import TikzTable
from gerrytools.latex._table_layout import TableBoundary, TablePreamble
from gerrytools.latex.formatters import (
    compose_formatters,
    diverging_gradient_formatter,
    highlight_ge,
    round_decimals,
    wrap_with_tex_command,
)


class TestTikzTableNiceTabularGeneration:
    def test_emits_nicetabular_with_tabular_dialect_rules(self, df):
        table = TikzTable(df)
        table.add_vrule_all()
        table.add_hrule_above_all()
        table.add_toprule()
        table.add_bottomrule()

        latex = table.document.body_string

        # Real tabular column spec: rules live in the preamble, so they meet
        # horizontal rules by construction (the old matrix-of-nodes emitter
        # drew rules between node anchors and left visible gaps).
        assert r"\begin{NiceTabular}{|c|c|c|c|c|c|}[name=table" in latex
        assert latex.count(r"\hline") >= len(df)  # one interior rule per data row
        assert latex.strip().endswith(r"\end{NiceTabular}")

    def test_add_hrule_above_bottom_boundary_emits_trailing_rule(self):
        # Counterpart of the TexTable trailing-rule test: the bottom-boundary rule count must
        # survive into the NiceTabular body too.
        frame = pd.DataFrame({"a": [1, 2, 3]})
        table = TikzTable(frame, use_defaults=False)

        table.add_hrule_above(len(frame), count=2)
        lines = table.document.body_string.splitlines()

        assert lines[-1] == r"\end{NiceTabular}"
        assert lines[-3:-1] == [r"\hline", r"\hline"]

    def test_header_double_rule_pushed_into_header(self, df):
        # The header double rule (count == 2) is split: a single \hline at the
        # header/data boundary plus the extra rule drawn in \CodeAfter, with a
        # depth strut on the header row supplying the \doublerulesep gap. This
        # keeps the first data row's colour band the same height as the others
        # (a plain \hline\hline absorbs the gap into the first shaded band). The
        # rule width/step are emitted as length registers so they track the
        # document's \arrayrulewidth / \doublerulesep.
        table = TikzTable(df)  # use_defaults=True -> header double rule

        latex = str(table)

        # No verbatim double rule; the pair is reconstructed instead.
        assert "\\hline\n\\hline" not in latex
        # Header depth strut sized for one extra \hline-width rule.
        assert r"{\rule[-\dimexpr1\doublerulesep+\arrayrulewidth\relax]{0pt}{0pt}} \\" in latex
        # Upper rule of the pair, drawn above the boundary across the full width
        # (df has 6 columns -> col-7 is the right boundary; data starts at row 2).
        assert (
            r"\draw[line width=\arrayrulewidth] ([yshift=1\doublerulesep]row-2-|col-1) -- "
            r"([yshift=1\doublerulesep]row-2-|col-7);" in latex
        )
        assert r"\fill[white]" not in latex

    def test_rules_never_perturbed_by_fills(self, df):
        # The header rule geometry is identical whether or not rows are coloured,
        # so the (row-i) lattice stays stable for custom \draw commands.
        table = TikzTable(df)
        table.highlight_rows(0, color="lightblue")

        latex = str(table)

        assert r"{\rule[-\dimexpr1\doublerulesep+\arrayrulewidth\relax]{0pt}{0pt}} \\" in latex
        assert (
            r"\draw[line width=\arrayrulewidth] ([yshift=1\doublerulesep]row-2-|col-1) -- "
            r"([yshift=1\doublerulesep]row-2-|col-7);" in latex
        )
        assert r"\fill[white]" not in latex

    def test_header_rule_stack_pushed_for_any_command_and_count(self, df):
        # The push generalises beyond \hline/count-2: stacking the default header
        # rule and an explicit add_hrule_above with a \midrule command yields a
        # triple rule (count 3). It must still emit ONE rule at the boundary and
        # draw the remaining two inside the header, using the booktabs
        # \lightrulewidth so the stack reads like real \midrule rows. Otherwise
        # nicematrix absorbs the gaps into the first shaded data row.
        table = TikzTable(df)  # use_defaults -> header rule count 2
        table.set_hrule_command(r"\midrule")
        table.add_hrule_above(0)  # -> count 3

        latex = str(table)

        assert "\\midrule\n\\midrule" not in latex  # not emitted verbatim
        # Strut sized for two extra \lightrulewidth rules.
        assert r"{\rule[-\dimexpr2\doublerulesep+\lightrulewidth\relax]{0pt}{0pt}} \\" in latex
        # Two extra rules drawn at 1x and 2x the step above the boundary (row 2).
        assert r"\draw[line width=\lightrulewidth] ([yshift=1\doublerulesep]row-2-|col-1)" in latex
        assert r"\draw[line width=\lightrulewidth] ([yshift=2\doublerulesep]row-2-|col-1)" in latex

    def test_single_and_custom_rules_emitted_verbatim(self, df):
        table = TikzTable(df, use_defaults=False)
        table.add_hrule_above_all()  # count 1 everywhere
        table.set_hrule_command(r"\midrule")

        latex = str(table)

        assert r"\midrule" in latex
        assert r"\Hline[tikz=" not in latex

    def test_row_and_cell_fills_emitted_in_code_before(self, df):
        table = TikzTable(df, use_defaults=False)
        table.highlight_rows([0], color="gray!50")
        table.set_number_formatter(
            compose_formatters(
                diverging_gradient_formatter(
                    color_lo="steelblue",
                    color_mid="white",
                    color_hi="firebrick",
                    command_name=None,
                ),
                round_decimals(3),
            )
        )

        latex = str(table)

        assert r"\CodeBefore" in latex
        assert r"\Body" in latex
        # Row highlight targets the first data row (row 2: one header row).
        assert r"\rowcolor{gray!50}{2}" in latex
        # Gradient cell colours are emitted inline as \cellcolor[HTML]{...}.
        assert r"\cellcolor[HTML]{" in latex
        # Regression: NO body-level \definecolor. nicematrix reserves spurious
        # horizontal space for every \definecolor in the document body, which
        # shifted gradient tables far to the right.
        assert r"\definecolor" not in latex

    def test_named_color_cell_fills_route_to_code_before(self, df):
        # A literal-path highlighter with an xcolor name lands in \CodeBefore as \cellcolor{name},
        # and the cell text keeps no \cellcolor prefix.
        table = TikzTable(df, use_defaults=False)
        table.set_column_formatter("Column 1", highlight_ge(0.0, color="teal"))

        latex = str(table)

        assert r"\cellcolor{teal}{2-1}" in latex
        body_lines = [line for line in latex.splitlines() if line.endswith(r" \\")]
        assert all(r"\cellcolor" not in line for line in body_lines)

    def test_wrapper_composed_outside_fill_formatter_still_routes_to_code_before(self, df):
        # Regression: the wrapper used to swallow the CellFillText, stranding \cellcolor inline
        # in the body instead of routing the fill to \CodeBefore.
        table = TikzTable(df, use_defaults=False)
        table.set_column_formatter(
            "Column 1",
            compose_formatters(
                wrap_with_tex_command("textbf"),
                highlight_ge(0.0, color="teal"),
            ),
        )

        latex = str(table)

        assert r"\cellcolor{teal}{2-1}" in latex
        body_lines = [line for line in latex.splitlines() if line.endswith(r" \\")]
        assert all(r"\cellcolor" not in line for line in body_lines)
        assert any(r"\textbf{0.8" in line for line in body_lines)

    def test_fill_formatter_composed_outside_wrapper_still_routes_to_code_before(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_column_formatter(
            "Column 1",
            compose_formatters(
                highlight_ge(0.0, color="teal"),
                wrap_with_tex_command("textbf"),
            ),
        )

        latex = str(table)

        assert r"\cellcolor{teal}{2-1}" in latex
        body_lines = [line for line in latex.splitlines() if line.endswith(r" \\")]
        assert all(r"\cellcolor" not in line for line in body_lines)
        assert any(r"\textbf{0.8" in line for line in body_lines)

    def test_hand_written_literal_cellcolor_string_stays_inline(self, df):
        # A user formatter returning a literal \cellcolor prefix as a plain string is not
        # re-parsed; only CellFillText results are routed to \CodeBefore.
        table = TikzTable(df, use_defaults=False)
        table.set_column_formatter("Column 1", lambda v, s: (v, r"\cellcolor{teal}" + s))

        latex = str(table)

        assert r"\CodeBefore" not in latex
        body_lines = [line for line in latex.splitlines() if line.endswith(r" \\")]
        assert any(r"\cellcolor{teal}" in line for line in body_lines)

    def test_no_body_definecolor_for_hex_row_highlight(self, df):
        # Same regression guard for hex-coloured row highlights.
        table = TikzTable(df, use_defaults=False)
        table.highlight_rows([1, 3], color="#F6E8C3")

        latex = str(table)

        assert r"\rowcolor[HTML]{F6E8C3}{3}" in latex
        assert r"\definecolor" not in latex

    def test_group_tabular_format_uses_multicolumn_rules(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_header_groups(
            {"Group A": ["Column 1", "Column 2", "Column 3"], "Group B": ["Column 4", "Column 5"]}
        )
        table.set_group_tabular_format("|c|c|c|")

        latex = str(table)

        assert r"\multicolumn{3}{|c|}{\textbf{Group A}}" in latex
        assert r"\multicolumn{2}{c|}{\textbf{Group B}}" in latex
        assert r"\multicolumn{1}{c|}{}" in latex

    def test_group_headers_use_multicolumn_spans(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_header_groups(
            {"Scores": ["Column 1", "Column 2", "Column 3", "Column 4", "Column 5"]}
        )

        latex = str(table)

        assert r"\multicolumn{5}{c}{\textbf{Scores}}" in latex
        assert r"\multicolumn{1}{c}{}" in latex

    def test_cell_borders_draw_on_boundary_lattice(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_cell_border(3, 2, "all")

        latex = str(table)

        assert r"\CodeAfter" in latex
        # Four sides of cell (3,2) on nicematrix's (row-i)/(col-j) lattice.
        assert r"\draw (row-3-|col-2) -- (row-3-|col-3);" in latex  # top
        assert r"\draw (row-4-|col-2) -- (row-4-|col-3);" in latex  # bottom
        assert r"\draw (row-3-|col-2) -- (row-4-|col-2);" in latex  # left
        assert r"\draw (row-3-|col-3) -- (row-4-|col-3);" in latex  # right

    def test_adjacent_cell_borders_share_edges(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_cell_border(3, 2, "right")
        table.set_cell_border(3, 3, "left")  # same physical edge

        latex = str(table)

        assert latex.count(r"\draw (row-3-|col-3) -- (row-4-|col-3);") == 1

    @pytest.mark.parametrize(
        ("row", "col", "message"),
        [
            (99, 1, "Row index 99"),
            (0, 1, "Row index 0"),
            (-1, 1, "Row index -1"),
            (1, 99, "Column index 99"),
            (1, 0, "Column index 0"),
        ],
    )
    def test_cell_borders_reject_out_of_range_indices(self, df, row, col, message):
        table = TikzTable(df, use_defaults=False)

        with pytest.raises(ValueError, match=message):
            table.set_cell_border(row, col, "all")

    def test_cell_borders_reject_unknown_sides(self, df):
        table = TikzTable(df, use_defaults=False)

        with pytest.raises(ValueError, match="diagonal"):
            table.set_cell_border(3, 2, "diagonal")

    def test_extra_draws_render_inside_code_after(self, df):
        table = TikzTable(df, use_defaults=False)
        draw = r"\draw[red] (table-2-1.north west) rectangle (table-2-1.south east);"
        table.add_draw(draw)

        latex = str(table)

        assert r"\CodeAfter" in latex
        assert r"\begin{tikzpicture}" in latex
        assert draw in latex

    def test_set_cell_space_limits_updates_nicematrix_option(self, df):
        table = TikzTable(df, use_defaults=False)

        table.set_cell_space_limits("3pt")

        assert "cell-space-limits=3pt" in str(table)

    @pytest.mark.parametrize(
        "limit", ["", "1", "1px", "1pt]", r"\smallskipamount", r"\input", r"\end"]
    )
    def test_set_cell_space_limits_rejects_unsafe_value(self, df, limit):
        with pytest.raises(ValueError, match="LaTeX dimension"):
            TikzTable(df, use_defaults=False).set_cell_space_limits(limit)

    def test_set_table_name_changes_nicematrix_name(self, df):
        # Two tables in one document need distinct nicematrix names to avoid node collisions.
        table = TikzTable(df, use_defaults=False)

        table.set_table_name("scores")

        assert "[name=scores, cell-space-limits=1pt]" in str(table)

    @pytest.mark.parametrize("name", ["", "1table", "bad_name", "bad,name", "bad]name"])
    def test_set_table_name_rejects_unsafe_name(self, df, name):
        with pytest.raises(ValueError, match="Table name"):
            TikzTable(df, use_defaults=False).set_table_name(name)

    def test_clear_extra_draws_removes_added_commands(self, df):
        table = TikzTable(df, use_defaults=False)
        draw = r"\draw[red] (table-2-1) -- (table-2-2);"
        table.add_draw(draw)

        table.clear_extra_draws()

        assert draw not in str(table)

    def test_clear_cell_borders_removes_added_borders(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_cell_border(3, 2, "right")

        table.clear_cell_borders()

        assert r"\draw (row-3-|col-3) -- (row-4-|col-3);" not in str(table)

    def test_boundary_extras_reconstruct_valid_preamble_ordering(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_tabular_format(r">{\bfseries}c|c c c c l")
        table.add_vrule_left_of(0)

        latex = str(table)

        # ``>{...}`` opens its column after any vrule at the same boundary.
        assert r"\begin{NiceTabular}{|>{\bfseries}c|c" in latex

    def test_at_and_bang_decorators_survive_into_colspec(self, df):
        # Regression: @{...} and !{...} boundary decorators were silently dropped from the
        # NiceTabular preamble; nicematrix accepts all four standard decorators.
        table = TikzTable(df, use_defaults=False)
        table.set_tabular_format(r">{\bfseries}c|!{\quad}c@{}c c c l")

        latex = str(table)

        assert r"\begin{NiceTabular}{>{\bfseries}c|!{\quad}c@{}cccl}" in latex

    def test_group_alignment_reaches_multicolumn(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_header_groups(
            {"Group A": ["Column 1", "Column 2", "Column 3"], "Group B": ["Column 4", "Column 5"]}
        )
        table.set_group_tabular_format("|l|r|c|")  # "Names" falls into the trailing "" group

        latex = str(table)

        assert r"\multicolumn{3}{|l|}{\textbf{Group A}}" in latex
        assert r"\multicolumn{2}{r|}{\textbf{Group B}}" in latex

    def test_rich_group_alignment_is_preserved(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_header_groups(
            {"Group A": ["Column 1", "Column 2", "Column 3"], "Group B": ["Column 4", "Column 5"]}
        )
        table.set_group_tabular_format("p{2cm}c c")

        latex = str(table)

        assert r"\multicolumn{3}{p{2cm}}{\textbf{Group A}}" in latex

    def test_group_boundary_extra_is_preserved(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_header_groups(
            {"Group A": ["Column 1", "Column 2", "Column 3"], "Group B": ["Column 4", "Column 5"]}
        )
        table.set_group_tabular_format(r">{\bfseries}c c c")

        assert r"\multicolumn{3}{>{\bfseries}c}{\textbf{Group A}}" in str(table)

    def test_unrecognized_boundary_extra_token_raises_at_generate_time(self, df):
        table = TikzTable(df, use_defaults=False)
        preamble = table._options.preamble
        table._options.preamble = TablePreamble(
            preamble.alignments,
            (TableBoundary(extra="E0"), *preamble.boundaries[1:]),
        )

        with pytest.raises(ValueError, match="Unsupported boundary token 'E0'"):
            str(table)

    def test_document_requires_nicematrix_and_two_passes(self, df):
        table = TikzTable(df, use_defaults=False)

        doc = table.document

        assert "nicematrix" in doc.package_list
        assert doc.compile_passes == 2


class TestTikzTableObjectCellEscaping:
    def test_object_cells_escape_special_characters_once(self):
        # Regression: same double-escape as TexTable's fallback branch (& -> \& ->
        # \textbackslash{}\&).
        import pandas as pd

        class _ObjectCell:
            def __str__(self) -> str:
                return "A & B_C"

        df = pd.DataFrame({"col": [_ObjectCell()]})
        latex = TikzTable(df).document.body_string
        assert r"A \& B\_C" in latex
        assert r"\textbackslash" not in latex


class TestTikzTableIncludeIndexInverse:
    def test_include_then_remove_index_restores_boundary_state_exactly(self, df):
        # Regression: remove_index popped boundary 0 only, leaving the right rule and extras
        # that include_index had merged into boundary 1 behind as phantoms.
        table = TikzTable(df)
        table.add_vrule_all()
        preamble = table._options.preamble
        table._options.preamble = TablePreamble(
            preamble.alignments,
            (
                TableBoundary(preamble.boundaries[0].vrules, "E0"),
                TableBoundary(preamble.boundaries[1].vrules, "E1"),
                *preamble.boundaries[2:],
            ),
        )
        before = table._options.preamble

        table.include_index(alignment=r">{\bfseries}c<{X}|")

        # Resolution merges index-owned syntax without changing the stored data boundary.
        resolved = table._resolved_preamble()
        assert resolved.boundaries[1].vrules == before.boundaries[0].vrules + 1
        assert resolved.boundaries[1].extra == "<{X}" + before.boundaries[0].extra
        assert table._options.preamble is before

        table.remove_index()

        assert table._options.preamble is before
        assert table._resolved_preamble() == before
