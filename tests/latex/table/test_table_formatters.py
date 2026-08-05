"""Tests for TexTable value and cell formatter behavior."""

import math
import re
from dataclasses import dataclass
from typing import cast

import pandas as pd
import pytest

from gerrytools.latex.formatters import (
    CellWrapper,
    compose_formatters,
    diverging_gradient_formatter,
    highlight_ge,
    latex_commands_for,
    wrap_with_tex_command,
)
from gerrytools.latex.table import TexTable


def command_name_of(formatter) -> str:
    """Extract the generated command name from a formatter's single preamble command."""
    (command,) = latex_commands_for(formatter)
    match = re.search(r"\\newcommand\{\\([A-Za-z]+)\}", command)
    assert match is not None
    return match.group(1)


@dataclass(frozen=True)
class _DemoIndexValue:
    label: str
    number: int


# =========================
# == TEXTABLE FORMATTERS ==
# =========================
class TestTexTableFormatters:
    def test_set_decimal_count_positive_path_and_used_in_body(self):
        df = pd.DataFrame({"a": [math.pi]})
        table = TexTable(df)

        table.set_decimal_count(2)
        body = table._generate_body()
        # Rounded to 2 decimals
        assert "3.14" in body

    def test_set_nan_string_used_in_body(self):
        df = pd.DataFrame({"a": [1.0, float("nan")]})
        table = TexTable(df)

        table.set_nan_string("NA")
        body = table._generate_body()
        assert "NA" in body

    def test_set_nan_string_is_inserted_as_raw_latex(self):
        # Contract: the NaN string is raw LaTeX; commands must not be escaped.
        df = pd.DataFrame({"a": [float("nan")]})
        table = TexTable(df)

        table.set_nan_string("\\textemdash")
        body = table._generate_body()

        assert "\\textemdash" in body
        assert "\\textbackslash" not in body

    def test_set_tabular_format_success_without_index(self, table_defaults):
        table = table_defaults
        ncols = len(table.df.columns)

        fmt = "c" * ncols
        table.set_tabular_format(fmt)

        assert table._options.preamble.alignments == ("c",) * ncols
        assert len(table._options.preamble.boundaries) == ncols + 1

    def test_set_tabular_format_success_with_index(self):
        df = pd.DataFrame({"a": [1, 2]})
        table = TexTable(df)
        table.include_index(include=True)

        fmt = "cc"
        table.set_tabular_format(fmt)

        assert table._resolved_preamble().alignments == ("c", "c")

    def test_set_group_tabular_format_prepends_omitted_index_cell(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        table = TexTable(df)

        table.include_index()
        table.set_tabular_format("|l||cc|")
        table.set_header_groups({"G1": ["a"], "G2": ["b"]})
        table.set_group_tabular_format("lr")

        assert table._options.group_preamble is not None
        assert table._options.group_index is not None
        assert table._options.group_index.alignment == "c"
        assert table._options.group_index.left == table._options.index_column.left
        assert table._options.group_index.right == table._options.index_column.right
        assert table._options.group_preamble.alignments == ("l", "r")
        assert len(table._options.group_preamble.boundaries) == 3

    def test_set_group_tabular_format_counts_only_nonempty_groups(self):
        table = TexTable(pd.DataFrame({"a": [1]}))
        table.set_header_groups({"Empty": [], "Full": ["a"]})

        table.set_group_tabular_format("r")

        assert table._options.group_preamble is not None
        assert table._options.group_preamble.alignments == ("r",)
        assert "Empty" not in table._multicolumn_format()

    def test_omitted_group_index_does_not_duplicate_shared_boundary(self):
        table = TexTable(pd.DataFrame({"a": [1], "b": [2]}), use_defaults=False)
        table.include_index()
        table.set_tabular_format("c|cc")
        table.set_header_groups({"G1": ["a"], "G2": ["b"]})

        table.set_group_tabular_format("|cc")

        assert r"\multicolumn{1}{c|}{}" in table.document.body_string
        assert r"\multicolumn{1}{c||}{}" not in table.document.body_string

    def test_omitted_group_index_preserves_first_group_opening_modifier(self):
        table = TexTable(pd.DataFrame({"a": [1], "b": [2]}), use_defaults=False)
        table.include_index()
        table.set_header_groups({"G1": ["a"], "G2": ["b"]})

        table.set_group_tabular_format(r">{\itshape}cc")

        assert r"\multicolumn{1}{>{\itshape}c}{\textbf{G1}}" in table.document.body_string

    def test_set_number_formatter_one_arg_and_used(self):
        df = pd.DataFrame({"a": [2]})
        table = TexTable(df)

        table.set_number_formatter(lambda x: f"{x:.1f}")
        body = table._generate_body()
        assert "2.0" in body

    def test_formatter_with_defaulted_second_parameter_uses_one_argument_form(self):
        table = TexTable(pd.DataFrame({"a": [2]}))

        def fmt(value, prefix="value="):
            return f"{prefix}{value}"

        table.set_number_formatter(fmt)

        assert "value=2" in table._generate_body()

    def test_set_number_formatter_two_arg_and_used(self):
        df = pd.DataFrame({"a": [2.0]})
        table = TexTable(df)

        def fmt(v, s):
            return v, f"VAL={s}"

        table.set_number_formatter(fmt)
        body = table._generate_body()
        assert "VAL=" in body

    def test_set_number_formatter_registers_required_latex_commands(self):
        df = pd.DataFrame({"a": [0.75]})
        table = TexTable(df)

        formatter = diverging_gradient_formatter(precision=3)
        table.set_number_formatter(formatter)

        doc = str(table.document)
        name = command_name_of(formatter)

        assert rf"\newcommand{{\{name}}}[1]{{%" in doc
        assert rf"\{name}{{0.750}}" in doc
        assert r"\cellcolor[HTML]" not in doc

    def test_differing_formatter_parameters_get_distinct_command_names(self):
        df = pd.DataFrame({"a": [0.25], "b": [0.75]})
        table = TexTable(df)

        formatter_a = diverging_gradient_formatter(
            color_lo="steelblue", color_hi="firebrick", precision=2
        )
        formatter_b = diverging_gradient_formatter(
            color_lo="darkpastelgreen",
            color_hi="richlavender",
            precision=2,
        )
        table.set_column_formatter("a", formatter_a)
        table.set_column_formatter("b", formatter_b)

        doc = str(table.document)
        name_a = command_name_of(formatter_a)
        name_b = command_name_of(formatter_b)

        assert name_a != name_b
        assert rf"\newcommand{{\{name_a}}}[1]{{%" in doc
        assert rf"\newcommand{{\{name_b}}}[1]{{%" in doc
        assert rf"\{name_a}{{0.25}}" in doc
        assert rf"\{name_b}{{0.75}}" in doc

    def test_differing_highlight_colors_get_distinct_command_names(self):
        df = pd.DataFrame({"a": [0.8], "b": [0.9]})
        table = TexTable(df)

        formatter_teal = highlight_ge(0.7, color="teal", command_prefix="ge")
        formatter_salmon = highlight_ge(0.7, color="salmon", command_prefix="ge")
        table.set_column_formatter("a", formatter_teal)
        table.set_column_formatter("b", formatter_salmon)

        doc = str(table.document)
        name_teal = command_name_of(formatter_teal)
        name_salmon = command_name_of(formatter_salmon)

        assert name_teal != name_salmon
        assert rf"\newcommand{{\{name_teal}}}[1]{{\cellcolor{{teal}}#1}}" in doc
        assert rf"\newcommand{{\{name_salmon}}}[1]{{\cellcolor{{salmon}}#1}}" in doc
        assert rf"\{name_teal}{{0.8}}" in doc
        assert rf"\{name_salmon}{{0.9}}" in doc

    def test_one_formatter_reused_in_two_tables_keeps_one_stable_name(self):
        # Regression: the command name was once mutable shared state, so registering one
        # formatter in a second table renamed the first table's command out from under it.
        formatter = highlight_ge(0.7, color="teal", command_prefix="ge")
        name = command_name_of(formatter)

        table_one = TexTable(pd.DataFrame({"a": [0.8]}))
        table_one.set_column_formatter("a", formatter)
        doc_one = str(table_one.document)

        table_two = TexTable(pd.DataFrame({"b": [0.9]}))
        table_two.set_column_formatter("b", formatter)
        doc_two = str(table_two.document)

        # The second registration must not have renamed the first table's command.
        assert str(table_one.document) == doc_one
        for doc in (doc_one, doc_two):
            assert doc.count(rf"\newcommand{{\{name}}}[1]") == 1
            assert rf"\{name}{{" in doc

    def test_identically_built_formatters_dedupe_instead_of_renaming(self):
        table = TexTable(pd.DataFrame({"a": [0.8], "b": [0.9]}))
        table.set_column_formatter("a", highlight_ge(0.7, color="teal", command_prefix="ge"))
        table.set_column_formatter("b", highlight_ge(0.7, color="teal", command_prefix="ge"))

        doc = str(table.document)
        name = command_name_of(highlight_ge(0.7, color="teal", command_prefix="ge"))

        assert doc.count(rf"\newcommand{{\{name}}}[1]") == 1

    def test_clear_options_drops_formatter_registered_commands(self):
        # Regression: formatter \newcommand definitions accumulated forever, so clear_options
        # never returned the document to its initial state.
        df = pd.DataFrame({"a": [0.8]})
        table = TexTable(df)
        table.set_number_formatter(diverging_gradient_formatter())
        assert any(r"\newcommand" in command for command in table._document.command_list)

        table.clear_options()

        assert table._document.command_list == []
        assert str(table) == str(TexTable(df, use_defaults=False))

    def test_clear_options_keeps_user_registered_commands(self):
        table = TexTable(pd.DataFrame({"a": [0.8]}))
        user_command = r"\newcommand{\mycmd}[1]{\textbf{#1}}"
        table.document.add_command(user_command)
        table.set_number_formatter(diverging_gradient_formatter())

        table.clear_options()

        assert table._document.command_list == [user_command]

    def test_unrelated_formatter_creation_order_does_not_change_names(self):
        first = highlight_ge(0.7, color="teal", command_prefix="ge")
        # Unrelated formatter construction in between must not perturb later names.
        highlight_ge(0.1, color="salmon", command_prefix="ge")
        diverging_gradient_formatter(precision=2)
        second = highlight_ge(0.7, color="teal", command_prefix="ge")

        assert command_name_of(first) == command_name_of(second)
        assert latex_commands_for(first) == latex_commands_for(second)

    def test_literal_path_attaches_no_command_metadata(self):
        formatter = highlight_ge(0.7, color="teal", command_prefix=None)

        assert latex_commands_for(formatter) == ()

        table = TexTable(pd.DataFrame({"a": [0.8]}))
        table.set_column_formatter("a", formatter)
        body = table._generate_body()

        assert r"\cellcolor{teal}0.8" in body
        assert r"\newcommand" not in str(table.document.command_list)

    def test_wrapper_composed_outside_fill_formatter_keeps_prefix_outermost(self):
        # Regression: the wrapper used to swallow the CellFillText, burying \cellcolor inside
        # \textbf instead of leaving it as an outermost prefix on the wrapped text.
        table = TexTable(pd.DataFrame({"a": [0.8]}))
        table.set_column_formatter(
            "a",
            compose_formatters(
                wrap_with_tex_command("textbf"),
                highlight_ge(0.7, color="teal"),
            ),
        )

        assert r"\cellcolor{teal}\textbf{0.8}" in table._generate_body()

    def test_fill_formatter_composed_outside_wrapper_keeps_prefix_outermost(self):
        table = TexTable(pd.DataFrame({"a": [0.8]}))
        table.set_column_formatter(
            "a",
            compose_formatters(
                highlight_ge(0.7, color="teal", command_prefix=None),
                wrap_with_tex_command("textbf"),
            ),
        )

        assert r"\cellcolor{teal}\textbf{0.8}" in table._generate_body()

    def test_set_string_formatter_one_arg_and_used(self):
        df = pd.DataFrame({"a": ["hello"]})
        table = TexTable(df)

        table.set_string_formatter(lambda s: s.upper())
        body = table._generate_body()
        assert "HELLO" in body

    def test_set_string_formatter_two_arg_and_used(self):
        df = pd.DataFrame({"a": ["hello"]})
        table = TexTable(df)

        def fmt(v, s):
            return v, s + "!"

        table.set_string_formatter(fmt)
        body = table._generate_body()
        assert "hello!" in body

    def test_string_formatter_does_not_format_numbers_without_number_formatter(self):
        table = TexTable(pd.DataFrame({"number": [1.5], "text": ["hello"]}), use_defaults=False)
        table.set_string_formatter(lambda value: f"<{value}>")

        body = table._generate_body()

        assert "1.5 & <hello>" in body
        assert "<1.5>" not in body

    def test_string_formatter_does_not_format_numpy_booleans(self):
        table = TexTable(
            pd.DataFrame({"left": [True, False], "right": [False, True]}),
            use_defaults=False,
        )
        table.set_string_formatter(lambda value: f"<{value}>")

        body = table._generate_body()

        assert "<True>" not in body
        assert "<False>" not in body
        assert body.count("True") == 2
        assert body.count("False") == 2

    def test_set_column_formatter_list_branch_and_wrapping(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        table = TexTable(df)

        def fmt(v, s):
            return v, f"C{v}"

        table.set_column_formatter(["a", "b"], fmt)
        body = table._generate_body()
        assert "C1" in body
        assert "C2" in body

    def test_set_column_formatter_one_arg_wrapper(self):
        df = pd.DataFrame({"a": [1]})
        table = TexTable(df)

        table.set_column_formatter("a", lambda v: f"{v}X")
        body = table._generate_body()
        assert "1X" in body

    def test_formatters_without_introspectable_signature_raise_explicit_error(self):
        # inspect.signature can fail (functools.partial over C callables, some builtins). Arity
        # sniffing by trial call would swallow a two-argument formatter's own TypeError and
        # re-invoke it one-argument, so the adapter demands an introspectable wrapper instead.
        class _Opaque:
            __signature__ = "not-a-signature"  # makes inspect.signature raise TypeError

            def __call__(self, value):
                return f"ONE{value}"

        df = pd.DataFrame({"a": [1]})
        table = TexTable(df)

        with pytest.raises(TypeError, match="one- or two-parameter signature"):
            table.set_column_formatter("a", _Opaque())

    def test_set_row_formatter_list_branch_and_wrapping(self):
        df = pd.DataFrame({"a": [1, 2]})
        table = TexTable(df)

        def fmt(v, s):
            return v, f"R{v}"

        table.set_row_formatter([0, 1], fmt)

        body = table._generate_body()
        assert "R1" in body
        assert "R2" in body

    def test_set_row_formatter_one_arg_wrapper(self):
        df = pd.DataFrame({"a": [1]})
        table = TexTable(df)

        table.set_row_formatter(0, lambda v: f"R{v}")
        body = table._generate_body()
        assert "R1" in body

    def test_set_index_formatter_one_arg_is_used_in_generated_body(self, table_defaults):
        table = table_defaults
        table.include_index(include=True, name="ID")
        table.set_index_formatter(lambda v: f"IDX-{int(v)}")

        body = table._generate_body()

        assert "IDX-0" in body
        assert "IDX-1" in body

    def test_set_index_formatter_one_arg_supports_hashable_tuple_index(self):
        df = pd.DataFrame({"a": [1, 2]})
        df.index = [_DemoIndexValue("A", 1), _DemoIndexValue("B", 2)]
        table = TexTable(df)
        table.include_index(include=True, name="ID")

        def fmt(v: object) -> str:
            assert isinstance(v, _DemoIndexValue)
            return f"{v.label}-{v.number}"

        table.set_index_formatter(fmt)

        body = table._generate_body()

        assert "A-1" in body
        assert "B-2" in body

    def test_set_index_formatter_two_arg_is_stored_and_used_in_generated_body(self, table_defaults):
        table = table_defaults
        table.include_index(include=True, name="ID")

        def fmt(v, s):
            return v, f"[{s}]"

        table.set_index_formatter(fmt)

        assert table._options.index_fmt_fn is not None
        assert "[0]" in table._generate_body()

    def test_all_formatter_paths_receive_escaped_previous_rendering(self):
        expected = r"A\_B \& C"
        seen: dict[str, str] = {}

        def capture(name):
            def formatter(value, previous):
                seen[name] = previous
                return value, previous

            return formatter

        column = TexTable(pd.DataFrame({"a": ["A_B & C"]}), use_defaults=False)
        column.set_column_formatter("a", capture("column"))
        column._generate_body()

        row = TexTable(pd.DataFrame({"a": ["A_B & C"]}), use_defaults=False)
        row.set_row_formatter(0, capture("row"))
        row._generate_body()

        number_value = type(
            "SpecialFloat",
            (float,),
            {"__str__": lambda self: "A_B & C"},
        )(1.0)
        number = TexTable(
            pd.DataFrame({"a": pd.Series([number_value], dtype=object)}),
            use_defaults=False,
        )
        number.set_number_formatter(capture("number"))
        number._generate_body()

        string = TexTable(pd.DataFrame({"a": ["A_B & C"]}), use_defaults=False)
        string.set_string_formatter(capture("string"))
        string._generate_body()

        index = TexTable(
            pd.DataFrame({"a": [1]}, index=pd.Index(["A_B & C"])),
            use_defaults=False,
        )
        index.include_index()
        index.set_index_formatter(capture("index"))
        index._generate_body()

        assert seen == {path: expected for path in ("column", "row", "number", "string", "index")}

    def test_bool_cells_bypass_default_number_formatter(self):
        body = TexTable(pd.DataFrame({"flag": [True, False]}))._generate_body()

        assert "True" in body
        assert "False" in body
        assert "1.0000" not in body
        assert "0.0000" not in body


# ===========================
# == FORMATTER CONTRACTS ==
# ===========================
class TestFormatterContracts:
    # The contract-violating formatters below are cast to CellWrapper: they exist to prove the
    # runtime validation, so the static signature is deliberately wrong.

    def test_two_arg_formatter_returning_plain_string_raises(self):
        # Regression: a 2-character string unpacks like a pair, so "hello" used to
        # silently truncate the cell to its second character instead of failing.
        table = TexTable(pd.DataFrame({"a": ["hello"]}), use_defaults=False)
        table.set_column_formatter("a", cast(CellWrapper, lambda v, s: "hello"))

        with pytest.raises(TypeError, match=r"must return a \(value, rendered_text\) 2-tuple"):
            table._generate_body()

    def test_two_arg_formatter_returning_wrong_length_tuple_raises(self):
        table = TexTable(pd.DataFrame({"a": [1.0]}), use_defaults=False)
        table.set_number_formatter(cast(CellWrapper, lambda v, s: (v, s, "extra")))

        with pytest.raises(TypeError, match="2-tuple"):
            table._generate_body()

    def test_zero_param_formatter_rejected_at_set_time(self):
        table = TexTable(pd.DataFrame({"a": [1]}), use_defaults=False)

        def zero_param_formatter() -> str:
            return "x"

        with pytest.raises(TypeError, match="one positional argument"):
            table.set_column_formatter("a", cast(CellWrapper, zero_param_formatter))

    def test_three_param_formatter_rejected_at_set_time(self):
        table = TexTable(pd.DataFrame({"a": [1]}), use_defaults=False)

        def three_param_formatter(value: object, escaped: str, extra: str) -> tuple[object, str]:
            return value, escaped

        with pytest.raises(TypeError, match="one positional argument"):
            table.set_row_formatter(0, cast(CellWrapper, three_param_formatter))

    def test_one_arg_string_formatter_receives_escaped_text(self):
        # Regression: one-arg string formatters used to receive the raw value and
        # replace the escaped rendering, letting raw % & _ reach the LaTeX output.
        table = TexTable(pd.DataFrame({"a": ["50% & up_ok"]}), use_defaults=False)
        table.set_string_formatter(lambda s: s.upper())

        body = table._generate_body()

        assert r"50\% \& UP\_OK" in body

    def test_column_formatter_takes_precedence_over_row_formatter(self):
        table = TexTable(pd.DataFrame({"a": [1], "b": [2]}), use_defaults=False)
        table.set_column_formatter("a", lambda v, s: (v, f"COL={s}"))
        table.set_row_formatter(0, lambda v, s: (v, f"ROW={s}"))

        body = table._generate_body()

        assert "COL=1" in body  # column formatter shadows the row formatter for column "a"
        assert "ROW=1" not in body
        assert "ROW=2" in body  # the row formatter still covers cells without a column formatter
