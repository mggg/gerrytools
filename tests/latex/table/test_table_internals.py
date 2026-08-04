"""Tests for LaTeX table internal parsing helpers."""

import pytest

from gerrytools.latex._table_preamble import (
    _consume_balanced,
    _infer_group_cell_align_from_data,
    _parse_tabular_preamble,
)


# ==============================
# == TABULAR PREAMBLE PARSING ==
# ==============================
class TestTabularPreambleParsing:
    @pytest.mark.timeout(1)
    def test_basic_letters_no_rules(self):
        cols, vr, ex = _parse_tabular_preamble("lcr")
        assert cols == ["l", "c", "r"]
        assert vr == [0, 0, 0, 0]
        assert ex == ["", "", "", ""]

    @pytest.mark.timeout(1)
    def test_splits_repeated_letters(self):
        cols, vr, ex = _parse_tabular_preamble("lccr")
        assert cols == ["l", "c", "c", "r"]
        assert vr == [0, 0, 0, 0, 0]
        assert ex == ["", "", "", "", ""]

    @pytest.mark.timeout(1)
    def test_counts_vertical_rules_per_boundary(self):
        cols, vr, ex = _parse_tabular_preamble(r"|l||c|r|")
        assert cols == ["l", "c", "r"]
        # boundaries: before l, between l-c, between c-r, after r
        assert vr == [1, 2, 1, 1]
        assert ex == ["", "", "", ""]

    @pytest.mark.timeout(1)
    def test_boundary_extras_gt_applies_to_next_column_boundary(self):
        cols, vr, ex = _parse_tabular_preamble(r"l||>{\scriptsize}cc|cc||c")
        assert cols == ["l", "c", "c", "c", "c", "c"]
        assert vr == [0, 2, 0, 1, 0, 2, 0]
        assert ex == ["", r">{\scriptsize}", "", "", "", "", ""]

    @pytest.mark.timeout(1)
    def test_boundary_extras_lt_applies_to_boundary_before_column(self):
        # l  <{...} c  means the <{...} is attached to the boundary before that c
        cols, vr, ex = _parse_tabular_preamble(r"l<{X}c")
        assert cols == ["l", "c"]
        assert vr == [0, 0, 0]
        assert ex == ["", r"<{X}", ""]

    @pytest.mark.timeout(1)
    def test_boundary_extras_at_and_bang_preserved(self):
        cols, vr, ex = _parse_tabular_preamble(r"@{}l!{\hspace{2pt}}c@{}")
        assert cols == ["l", "c"]
        assert vr == [0, 0, 0]
        assert ex == [r"@{}", r"!{\hspace{2pt}}", r"@{}"]

    @pytest.mark.timeout(1)
    def test_mixed_rules_and_extras(self):
        cols, vr, ex = _parse_tabular_preamble(r"|@{}l||>{\bfseries}c|r@{}|")
        assert cols == ["l", "c", "r"]
        assert vr == [1, 2, 1, 1]
        assert ex == [r"@{}", r">{\bfseries}", "", r"@{}"]

    @pytest.mark.timeout(1)
    def test_p_m_b_columns_are_single_column_each(self):
        cols, vr, ex = _parse_tabular_preamble(r"p{2cm}|m{1in}b{3em}")
        assert cols == [r"p{2cm}", r"m{1in}", r"b{3em}"]
        assert vr == [0, 1, 0, 0]
        assert ex == ["", "", "", ""]

    @pytest.mark.timeout(1)
    def test_S_column_with_options_is_single_column(self):
        cols, vr, ex = _parse_tabular_preamble(r"l|S[table-format=2.3]|r")
        assert cols == ["l", r"S[table-format=2.3]", "r"]
        assert vr == [0, 1, 1, 0]
        assert ex == ["", "", "", ""]

    @pytest.mark.timeout(1)
    def test_D_column_is_single_column(self):
        cols, vr, ex = _parse_tabular_preamble(r"l|D{.}{.}{-1}|r")
        assert cols == ["l", r"D{.}{.}{-1}", "r"]
        assert vr == [0, 1, 1, 0]
        assert ex == ["", "", "", ""]

    @pytest.mark.timeout(1)
    def test_parse_complex_preamble(self):
        fmt_complex = r"|@{}l||>{\scriptsize\bfseries\color{purpleheart}}c<{\,}!{\hspace{2pt}}S[table-format=2.3,table-number-alignment=center]|D{.}{\cdot}{-1}||>{\raggedleft\arraybackslash}p{2.5cm}|m{1.8cm}b{3em}|r@{}|"
        expected_cols = [
            "l",
            "c",
            r"S[table-format=2.3,table-number-alignment=center]",
            r"D{.}{\cdot}{-1}",
            r"p{2.5cm}",
            r"m{1.8cm}",
            r"b{3em}",
            "r",
        ]

        expected_vrules = [1, 2, 0, 1, 2, 1, 0, 1, 1]

        expected_extras = [
            r"@{}",
            r">{\scriptsize\bfseries\color{purpleheart}}",
            r"<{\,}!{\hspace{2pt}}",
            "",
            r">{\raggedleft\arraybackslash}",
            "",
            "",
            "",
            r"@{}",
        ]
        cols, vr, ex = _parse_tabular_preamble(fmt_complex)
        assert cols == expected_cols
        assert vr == expected_vrules
        assert ex == expected_extras
        assert len(vr) == len(cols) + 1
        assert len(ex) == len(cols) + 1


# =================================
# == TABULAR PREAMBLE WHITESPACE ==
# =================================
class TestTabularPreambleWhitespace:
    def test_parse_tabular_preamble_ignores_whitespace_simple(self):
        fmt = "   |   l   c   r   "
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)

        # Columns are correctly parsed
        assert colspecs == ["l", "c", "r"]

        # Vertical rules:
        #   leading '|' applies before first column
        #   then no more bars
        assert vrules == [1, 0, 0, 0]

        # Extras: length ncols + 1, all empty strings
        assert extras == ["", "", "", ""]

    def test_parse_tabular_preamble_whitespace_around_S_and_brackets(self):
        fmt = "  S  [  table-format = 1.2 ]   l  "
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)

        # S with bracketed options is preserved, with whatever is inside [...]
        assert colspecs[0].startswith("S[")
        assert "table-format" in colspecs[0]
        assert colspecs[1] == "l"

        # 2 columns => vrules length 3
        assert len(vrules) == 3
        assert len(extras) == 3

    def test_parse_tabular_preamble_whitespace_with_boundary_extras_and_cols(self):
        fmt = "  @{}   l   !{foo}  c  "
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)

        # Two simple columns
        assert colspecs == ["l", "c"]

        # No explicit '|' -> all zero vrules; length n+1 = 3
        assert vrules == [0, 0, 0]

        # First boundary has @{}, second has !{foo}, third empty
        assert extras == ["@{}", "!{foo}", ""]


# =============================
# == TABULAR PREAMBLE ERRORS ==
# =============================
class TestTabularPreambleErrors:
    def test_parse_tabular_preamble_stray_open_brace_raises(self):
        fmt = "{}"
        with pytest.raises(ValueError) as excinfo:
            _parse_tabular_preamble(fmt)
        msg = str(excinfo.value)
        assert "Stray '{' at pos 0" in msg
        assert fmt in msg

    def test_parse_tabular_preamble_stray_close_brace_raises(self):
        fmt = " }l"
        with pytest.raises(ValueError) as excinfo:
            _parse_tabular_preamble(fmt)
        msg = str(excinfo.value)
        # pos 1 after initial space
        assert "Stray '}' at pos 1" in msg
        assert fmt in msg

    def test_parse_tabular_preamble_stray_open_bracket_raises(self):
        fmt = "[l]"
        with pytest.raises(ValueError) as excinfo:
            _parse_tabular_preamble(fmt)
        msg = str(excinfo.value)
        assert "Stray '[' at pos 0" in msg
        assert fmt in msg

    def test_parse_tabular_preamble_stray_close_bracket_raises(self):
        fmt = " ]c"
        with pytest.raises(ValueError) as excinfo:
            _parse_tabular_preamble(fmt)
        msg = str(excinfo.value)
        assert "Stray ']' at pos 1" in msg
        assert fmt in msg

    def test_parse_tabular_preamble_boundary_extra_missing_brace_raises(self):
        # '@' must be followed immediately by '{'
        fmt = "@c"
        with pytest.raises(ValueError) as excinfo:
            _parse_tabular_preamble(fmt)
        msg = str(excinfo.value)
        assert "Expected '{' after @" in msg
        assert fmt in msg

        fmt2 = "!  c"
        # Here fmt[0] = '!', fmt[1] = ' ' -> also wrong
        with pytest.raises(ValueError) as excinfo2:
            _parse_tabular_preamble(fmt2)
        msg2 = str(excinfo2.value)
        assert "Expected '{' after !" in msg2
        assert fmt2 in msg2

    def test_parse_tabular_preamble_p_column_missing_brace_raises(self):
        # 'p' must be followed by '{'
        fmt = "p l"
        with pytest.raises(ValueError) as excinfo:
            _parse_tabular_preamble(fmt)
        msg = str(excinfo.value)
        assert "Expected '{' after p" in msg
        assert fmt in msg

    def test_parse_tabular_preamble_m_column_missing_brace_raises(self):
        fmt = "m"
        with pytest.raises(ValueError) as excinfo:
            _parse_tabular_preamble(fmt)
        msg = str(excinfo.value)
        assert "Expected '{' after m" in msg
        assert fmt in msg

    def test_parse_tabular_preamble_b_column_missing_brace_raises(self):
        fmt = "b c"
        with pytest.raises(ValueError) as excinfo:
            _parse_tabular_preamble(fmt)
        msg = str(excinfo.value)
        assert "Expected '{' after b" in msg
        assert fmt in msg

    def test_parse_tabular_preamble_D_missing_brace_raises(self):
        # After 'D' (and optional whitespace) we must see '{'
        fmt = "D  l"
        with pytest.raises(ValueError) as excinfo:
            _parse_tabular_preamble(fmt)
        msg = str(excinfo.value)
        assert "Expected '{' after D" in msg
        assert fmt in msg

    def test_parse_tabular_preamble_unsupported_token_raises(self):
        # 'x' is not recognized as any valid token
        fmt = "x"
        with pytest.raises(ValueError) as excinfo:
            _parse_tabular_preamble(fmt)
        msg = str(excinfo.value)
        assert "Unsupported token 'x' at pos 0" in msg
        assert fmt in msg


# =====================
# == GROUP ALIGNMENT ==
# =====================
class TestGroupAlignment:
    def test_infer_group_align_all_same(self):
        colspecs = ["l", "l", "l", "l"]
        assert _infer_group_cell_align_from_data(colspecs, 0, 4) == "l"

        colspecs = ["r", "r"]
        assert _infer_group_cell_align_from_data(colspecs, 0, 2) == "r"

        colspecs = ["c", "c", "c"]
        assert _infer_group_cell_align_from_data(colspecs, 0, 3) == "c"

    def test_infer_group_align_mixed_defaults_to_c(self):
        colspecs = ["l", "r", "l"]
        assert _infer_group_cell_align_from_data(colspecs, 0, 3) == "c"

        colspecs = ["l", "p{2cm}", "l"]  # complex treated as 'c'
        assert _infer_group_cell_align_from_data(colspecs, 0, 3) == "c"


# ===============================
# == CONSUME BALANCED FUNCTION ==
# ===============================
class TestConsumeBalanced:
    def test_consume_balanced_simple_braces_whole_string(self):
        s = "{abc}"
        content, idx = _consume_balanced(s, 0, "{", "}")
        assert content == "abc"
        # index should be just past the closing brace
        assert idx == len(s)

    def test_consume_balanced_empty_group(self):
        s = "{}"
        content, idx = _consume_balanced(s, 0, "{", "}")
        assert content == ""
        assert idx == len(s)

    def test_consume_balanced_nested_braces(self):
        s = "{a{b}c}"
        content, idx = _consume_balanced(s, 0, "{", "}")
        # inner braces are preserved in the returned content
        assert content == "a{b}c"
        assert idx == len(s)

    def test_consume_balanced_multiple_nested_levels(self):
        s = "{a{b{c}d}e}"
        content, idx = _consume_balanced(s, 0, "{", "}")
        assert content == "a{b{c}d}e"
        assert idx == len(s)

    def test_consume_balanced_from_middle_of_string(self):
        s = "xxx{hello}yyy"
        #     0123456789 10
        #           ^ i = 3
        content, idx = _consume_balanced(s, 3, "{", "}")
        assert content == "hello"
        # index should point just past the '}'
        assert idx == 3 + len("{hello}")
        assert s[idx:] == "yyy"

    def test_consume_balanced_square_brackets_simple(self):
        s = "[table-format=1.3]"
        content, idx = _consume_balanced(s, 0, "[", "]")
        assert content == "table-format=1.3"
        assert idx == len(s)

    def test_consume_balanced_square_brackets_with_braces_inside(self):
        s = "[D{.}{,}{3}]"
        #     0123456789 10
        content, idx = _consume_balanced(s, 0, "[", "]")
        # braces are treated as ordinary characters here
        assert content == "D{.}{,}{3}"
        assert idx == len(s)

    def test_consume_balanced_braces_with_brackets_inside(self):
        s = "{S[table-format=1.3]}"
        content, idx = _consume_balanced(s, 0, "{", "}")
        # square brackets are just characters; only { } affect depth
        assert content == "S[table-format=1.3]"
        assert idx == len(s)

    def test_consume_balanced_start_not_on_open_char_raises(self):
        s = "abc}"
        with pytest.raises(ValueError, match=r"Expected '\{' at position 0"):
            _consume_balanced(s, 0, "{", "}")

    def test_consume_balanced_start_past_end_raises(self):
        s = "{abc}"
        with pytest.raises(ValueError, match=r"Expected '\{' at position 5"):
            _consume_balanced(s, len(s), "{", "}")

    def test_consume_balanced_unbalanced_missing_closing(self):
        s = "{abc"
        with pytest.raises(ValueError, match=r"Unbalanced \{\} in format string"):
            _consume_balanced(s, 0, "{", "}")

    def test_consume_balanced_unbalanced_nested_missing_closing(self):
        s = "{a{b}c"
        with pytest.raises(ValueError, match=r"Unbalanced \{\} in format string"):
            _consume_balanced(s, 0, "{", "}")

    def test_consume_balanced_unbalanced_extra_closing(self):
        s = "{{a}}}"
        # Consume the first balanced group starting at 0
        content, idx = _consume_balanced(s, 0, "{", "}")
        assert content == "{a}"
        # idx should point just past the *matching* closing brace at index 4
        assert idx == 5
        assert s[idx] == "}"  # the extra unmatched closing brace

        # Now starting at that extra '}' should fail, since it's not an opening '{'
        with pytest.raises(ValueError, match=r"Expected '\{' at position 5"):
            _consume_balanced(s, idx, "{", "}")

    def test_consume_balanced_zero_length_string_raises(self):
        s = ""
        with pytest.raises(ValueError, match=r"Expected '\{' at position 0"):
            _consume_balanced(s, 0, "{", "}")

    def test_consume_balanced_deep_nesting_near_end(self):
        s = "X{a{b{c{d}e}f}g}Y"
        #        ^ start at index of first '{'
        start_idx = s.index("{")
        content, idx = _consume_balanced(s, start_idx, "{", "}")
        assert content == "a{b{c{d}e}f}g"
        # idx points to just after the matching '}'
        assert s[idx] == "Y"
        assert idx == len(s) - 1
