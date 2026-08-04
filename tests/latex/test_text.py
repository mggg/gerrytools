"""Tests for LaTeX text escaping helpers."""

from gerrytools.latex._text import latex_escape


# ====================
# == LATEX ESCAPING ==
# ====================
class TestLatexEscape:
    def test_latex_escape_escapes_all_special_characters(self):
        text = r"\&%$#_{}~^<>|"
        expected = (
            r"\textbackslash{}"
            r"\&"
            r"\%"
            r"\$"
            r"\#"
            r"\_"
            r"\{"
            r"\}"
            r"\textasciitilde{}"
            r"\textasciicircum{}"
            r"\textless{}"
            r"\textgreater{}"
            r"\textbar{}"
        )
        assert latex_escape(text) == expected

    def test_latex_escape_leaves_plain_text_unchanged(self):
        assert latex_escape("Vote Share 2024") == "Vote Share 2024"

    def test_latex_escape_handles_empty_string(self):
        assert latex_escape("") == ""

    def test_latex_escape_unicode_passthrough(self):
        text = "Δistrict – café 🗳️"
        assert latex_escape(text) == text

    def test_latex_escape_is_exported_publicly(self):
        from gerrytools.latex import latex_escape as public_latex_escape

        assert public_latex_escape is latex_escape
