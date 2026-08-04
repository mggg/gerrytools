"""Image snapshot tests for TeX table rendering.

These tests require a TeX engine to be installed (e.g., tectonic or pdflatex).
Run with: pytest -m latex
"""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from gerrytools.latex._render import _which_any
from gerrytools.latex.commands import (
    tex_diverging_gradient_command,
    tex_gradient_command,
    tex_twocolor_gradient_command,
)
from gerrytools.latex.document import TexDocument
from gerrytools.latex.formatters import (
    compose_formatters,
    highlight_between,
    highlight_ge,
    highlight_gt,
    highlight_le,
    highlight_lt,
    round_decimals,
    wrap_with_tex_command,
)
from gerrytools.latex.tikz_table import TikzTable
from tests._image_snapshots import assert_image_snapshot


def require_tex_engine():
    """Skip tests cleanly if no TeX engine is available."""
    engines = TexDocument().engine_preference_order
    if _which_any(engines) is None:
        reason_str = (
            f"No TeX engine found for engines: {', '.join(engines)}; "
            "install one (e.g., tectonic or pdflatex) to run image snapshot tests."
        )
        pytest.skip(reason_str)


def render_doc_to_image(doc: TexDocument, *, dpi: int = 250) -> Image.Image:
    doc._render_to_temp_png(dpi=dpi)
    return Image.open(doc._png_path)


# ===============================
# == TEX TABLE IMAGE SNAPSHOTS ==
# ===============================
class TestTexTableImageSnapshots:
    @pytest.mark.latex
    def test_full_table_defaults_image_snapshot(self, table_defaults, tmp_path):
        require_tex_engine()

        # Use the document that TexTable already owns/configures
        doc = table_defaults.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_defaults_full",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_header_only_image_snapshot(self, table_defaults, tmp_path):
        require_tex_engine()

        header = table_defaults._generate_header()
        # header already contains \begin{tabular}{...} and header row; just close it
        header_only = header + "\n\\end{tabular}"

        doc = table_defaults.document
        doc.body_string = header_only
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_defaults_header_only",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_body_only_image_snapshot(self, table_defaults, tmp_path):
        require_tex_engine()

        # Build a minimal tabular around just the body rows
        fmt = table_defaults._column_format()  # e.g. "|c|c|c|..."

        body = table_defaults._generate_body()
        tex_body = rf"\begin{{tabular}}{{{fmt}}}" + "\n" + body + "\n\\end{tabular}"

        doc = table_defaults.document
        doc.body_string = tex_body
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_defaults_body_only",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_include_index_image_snapshot(self, table_plain, tmp_path):
        require_tex_engine()

        table_plain.include_index(name="ID", alignment="l", include=True)

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_plain_include_index",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_header_groups_image_snapshot(self, table_plain, tmp_path):
        require_tex_engine()

        table_plain.set_header_groups(
            {
                "Numbers A": ["Column 1", "Column 2", "Column 3"],
                "Numbers B": ["Column 4", "Column 5"],
                "Labels": ["Names"],
            }
        )

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_plain_header_groups",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_row_highlight_image_snapshot(self, table_plain, tmp_path):
        require_tex_engine()

        table_plain.highlight_rows([0], color="yellow")
        table_plain.highlight_rows([3], color="#FF00AA")
        table_plain.highlight_rows([5], color=(10, 20, 30))  # RGB ints
        table_plain.highlight_rows([7], color=(0.1, 0.2, 0.3))  # rgb floats

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_plain_row_highlight",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_simple_string_formatter_snapshot(self, table_plain, tmp_path):
        require_tex_engine()

        def make_strings_uppercase(x):
            if isinstance(x, str):
                return x.upper()
            return x

        table_plain.set_string_formatter(make_strings_uppercase)
        table_plain.set_tabular_format(r"ccccc>{\cellcolor{amber}}c")

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_plain_string_uppercase",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_multiple_highlights_one_row(self, table_plain, tmp_path):
        require_tex_engine()

        table_plain.set_row_formatter(
            1,
            compose_formatters(highlight_gt(0.5, color="cherryblossompink"), round_decimals(2)),
        )
        table_plain.highlight_rows(1, color="lightblue")

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_plain_multiple_highlights_one_row",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_grouped_headers(self, table_defaults, tmp_path):
        require_tex_engine()

        table_defaults.set_header_groups(
            {
                "Group 1": ["Column 3", "Column 5"],
                "Group 3": ["Column 1", "Column 2", "Column 4"],
            }
        )
        table_defaults.set_tabular_format(r"cc||ccc||>{\bfseries}c")
        table_defaults.include_index(name="My Index", alignment=r"c|")
        table_defaults.set_decimal_count(4)
        table_defaults.set_column_headers_text_format(bold=False, italic=True)

        doc = table_defaults.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_defaults_with_grouped_headers",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_several_highlight_rows_and_grouped_headers(self, table_defaults, tmp_path):
        require_tex_engine()

        table_defaults.set_header_groups(
            {
                "Group 1": ["Column 3", "Column 5"],
                "Group 3": ["Column 1", "Column 2", "Column 4"],
            }
        )
        table_defaults.set_group_tabular_format("lc||c")
        table_defaults.set_tabular_format(r"cc||ccc||>{\bfseries}c")
        table_defaults.include_index(name="My Index", alignment=r">{\bfseries}c|")
        table_defaults.highlight_rows([2, 3], color="amber")
        table_defaults.highlight_rows([4], color="amber!80!gray")
        table_defaults.highlight_rows([5], color="#e6b319")
        table_defaults.highlight_rows([6], color="applegreen!60!gray")
        table_defaults.highlight_rows([7], color="#87a033")
        table_defaults.set_decimal_count(2)
        table_defaults.set_column_headers_text_format(bold=False, italic=True)

        doc = table_defaults.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_defaults_several_highlight_rows_and_grouped_headers",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_gradient_command_image_snapshot(self, table_plain, tmp_path):
        require_tex_engine()

        table_plain.set_header_groups(
            {
                "Group 1": ["Column 1", "Column 5", "Column 2"],
                "Group 2": ["Column 4", "Column 3"],
            }
        )

        table_plain.set_nan_string("---")
        table_plain.set_number_formatter(
            compose_formatters(
                wrap_with_tex_command("myheatmap"),
                round_decimals(3),
            )
        )

        table_plain.document.add_command(tex_gradient_command("myheatmap", precision=3))

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_plain_gradient_command",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_twocolor_gradient_command_image_snapshot(self, table_plain, tmp_path):
        require_tex_engine()

        table_plain.set_header_groups(
            {
                "Group 1": ["Column 1", "Column 5", "Column 2"],
                "Group 2": ["Column 4", "Column 3"],
            }
        )

        table_plain.set_nan_string("---")
        table_plain.set_number_formatter(
            compose_formatters(
                wrap_with_tex_command("myheatmap"),
                round_decimals(3),
            )
        )

        table_plain.document.add_command(tex_twocolor_gradient_command("myheatmap", precision=3))

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_plain_twocolor_gradient_command",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_diverging_gradient_command_image_snapshot(self, table_plain, tmp_path):
        require_tex_engine()

        table_plain.set_header_groups(
            {
                "Group 1": ["Column 1", "Column 5", "Column 2"],
                "Group 2": ["Column 4", "Column 3"],
            }
        )

        table_plain.set_nan_string("---")
        table_plain.set_number_formatter(
            compose_formatters(
                wrap_with_tex_command("myheatmap"),
                round_decimals(3),
            )
        )

        table_plain.document.add_command(tex_diverging_gradient_command("myheatmap", precision=3))

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_plain_diverging_gradient_command",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    @pytest.mark.parametrize(
        "highlight_func, name_suffix",
        [
            (highlight_gt, "gt"),
            (highlight_ge, "ge"),
            (highlight_lt, "lt"),
            (highlight_le, "le"),
        ],
    )
    @pytest.mark.parametrize(
        "colortype, color",
        [
            ("latex", "teal"),
            ("hex", "#FF0000"),
            ("rgb", (0.0, 1.0, 0.0)),
            ("RGB255", (0, 0, 255)),
        ],
    )
    def test_simple_highlight_snapshot(
        self, table_plain, tmp_path, highlight_func, name_suffix, colortype, color
    ):
        require_tex_engine()

        table_plain.set_number_formatter(
            compose_formatters(highlight_func(0.7, color=color), round_decimals(2))
        )

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name=f"table_plain_{name_suffix}_highlight_{colortype}_color",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    @pytest.mark.parametrize(
        "colortype, color",
        [
            ("latex", "denim!50"),
            ("hex", "#FFFF00"),
            ("rgb", (0, 1.0, 1.0)),
            ("RGB255", (255, 0, 255)),
        ],
    )
    def test_simple_highlight_between_snapshot(self, table_plain, tmp_path, colortype, color):
        require_tex_engine()

        table_plain.set_number_formatter(
            compose_formatters(highlight_between(0.3, 0.5, color=color), round_decimals(2))
        )

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name=f"table_plain_between_highlight_{colortype}_color",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_highlight_between_exclude_bounds_snapshot(self, df, table_plain, tmp_path):
        require_tex_engine()
        min_val = float(df[[f"Column {i}" for i in range(1, 6)]].min().min())
        max_val = float(df[[f"Column {i}" for i in range(1, 6)]].max().max())

        table_plain.set_number_formatter(
            compose_formatters(
                highlight_between(
                    min_val, max_val, "applegreen", include_lower=False, include_upper=False
                ),
                round_decimals(5),
            )
        )

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_highlight_between_exclude_min_max",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_simple_highlight_ge_with_round_snapshot(self, table_plain, tmp_path):
        require_tex_engine()

        # This should highlight the cell with value 0.8027
        table_plain.set_number_formatter(
            compose_formatters(highlight_ge(0.803, color="denim!50", round_to=3), round_decimals(4))
        )

        doc = table_plain.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_plain_ge_round2_highlight",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.latex
    def test_many_rules(self, table_defaults, tmp_path):
        require_tex_engine()

        table_defaults.set_tabular_format("cccccc")
        table_defaults.add_hrule_above([1, 5, 7])
        table_defaults.add_vrule_left_of([0, 2, 4])
        table_defaults.add_vrule_right_of(0)
        table_defaults.set_toprule_command()
        table_defaults.set_bottomrule_command()

        doc = table_defaults.document
        img = render_doc_to_image(doc)

        assert_image_snapshot(
            img=img,
            name="table_defaults_with_extra_h_and_v_rules",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )


class TestTikzTableImageSnapshots:
    @pytest.mark.latex
    def test_group_header_rule_parity_image_snapshot(self, tmp_path):
        require_tex_engine()
        table = TikzTable(pd.DataFrame({"A": [1], "B": [2], "C": [3]}), use_defaults=False)
        table.set_tabular_format("c|c|c")
        table.set_header_groups({"G1": ["A", "B"], "G2": ["C"]})
        table.set_group_tabular_format("cc")

        img = render_doc_to_image(table.document)

        assert_image_snapshot(
            img=img,
            name="tikz_table_group_header_rule_parity",
            snapshots_dir=Path("tests/latex/table_image_snapshots"),
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )
