"""Regenerate the committed LaTeX documentation images.

Run locally in a TeX-capable environment (pdflatex + nicematrix + tikz):

    uv run python user_guide/_generate_latex_images.py

Ordinary documentation builds never run TeX; they display the committed PNGs in
``user_guide/_static/images/latex/``. Each snippet below is the same code shown in the
corresponding notebook cell, so the images cannot drift from the documented calls without
this file changing too.
"""

from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parent / "_static" / "images" / "latex"

SETUP = 'import numpy as np\nimport pandas as pd\n\ndistricts = pd.DataFrame(\n    {\n        "District": [f"CD {i}" for i in range(1, 9)],\n        "BVAP share": [0.12, 0.18, 0.22, 0.31, 0.38, 0.44, 0.52, 0.58],\n        "Dem share": [0.35, 0.41, 0.44, 0.47, 0.50, 0.55, 0.61, 0.66],\n        "Polsby-Popper": [0.18, 0.22, 0.27, 0.31, 0.33, 0.35, 0.41, 0.44],\n        "Pop. deviation": [0.004, 0.002, np.nan, 0.006, 0.001, 0.008, 0.003, 0.005],\n    }\n)\ndistricts'

SNIPPETS: dict[str, str] = {
    "textable-default": "from gerrytools.latex import TexTable\n\ntable = TexTable(districts)\ntable.set_decimal_count(3)",
    "textable-rules": 'table = TexTable(districts)\ntable.set_decimal_count(2)\ntable.include_index(name="Row", alignment="c")\ntable.add_vrule_right_of(0)\ntable.add_hrule_above([4])',
    "textable-groups": 'table = TexTable(districts)\ntable.set_decimal_count(2)\ntable.set_header_groups(\n    {\n        "Identity": ["District"],\n        "Demographics and votes": ["BVAP share", "Dem share"],\n        "Diagnostics": ["Polsby-Popper", "Pop. deviation"],\n    }\n)\ntable.set_column_headers_text_format(bold=False, italic=True)',
    "textable-highlights": 'table = TexTable(districts)\ntable.set_decimal_count(2)\ntable.highlight_rows([0], color="cherryblossompink")\ntable.highlight_rows([3], color="#c7e5f4")\ntable.highlight_rows([6], color="amber!40!white")',
    "textable-formatters": 'def shout(value):\n    return value.upper() if isinstance(value, str) else value\n\n\ntable = TexTable(districts)\ntable.set_nan_string("---")\ntable.set_string_formatter(shout)\ntable.set_column_formatter("BVAP share", round_decimals(2))\ntable.set_column_formatter("Pop. deviation", round_decimals(3))',
    "textable-conditional": 'table = TexTable(districts)\ntable.set_decimal_count(2)\ntable.set_column_formatter(\n    "BVAP share",\n    compose_formatters(\n        highlight_ge(0.50, color="applegreen!40!white"),\n        round_decimals(2),\n    ),\n)\ntable.set_column_formatter(\n    "Dem share",\n    compose_formatters(\n        highlight_between(0.45, 0.55, color="amber!35!white", include_lower=False),\n        round_decimals(2),\n    ),\n)',
    "textable-diverging": 'table = TexTable(districts)\ntable.set_decimal_count(2)\ntable.set_column_formatter(\n    "Dem share",\n    compose_formatters(\n        diverging_gradient_formatter(\n            lo=0.35,\n            mid=0.50,\n            hi=0.65,\n            color_lo="alizarin",\n            color_mid="white",\n            color_hi="denim",\n            command_name=None,\n        ),\n        round_decimals(2),\n    ),\n)',
    "textable-heatmap": 'table = TexTable(districts)\ntable.set_nan_string("---")\ntable.set_number_formatter(\n    compose_formatters(wrap_with_tex_command("heatmap"), round_decimals(2))\n)\ntable.document.add_command(tex_twocolor_gradient_command("heatmap"))',
    "tikztable-default": "from gerrytools.latex import TikzTable\n\ntable = TikzTable(districts)\ntable.set_decimal_count(3)",
    "tikztable-groups": 'table = TikzTable(districts)\ntable.set_decimal_count(2)\ntable.include_index(name="Row")\ntable.set_header_groups(\n    {\n        "Identity": ["District"],\n        "Demographics and votes": ["BVAP share", "Dem share"],\n        "Diagnostics": ["Polsby-Popper", "Pop. deviation"],\n    }\n)',
    "tikztable-highlights": 'table = TikzTable(districts)\ntable.set_decimal_count(2)\ntable.highlight_rows([1], color="cherryblossompink")\ntable.highlight_rows([4], color="lightblue!35!white")\ntable.highlight_rows([7], color="amber!40!white")',
    "tikztable-rules": "table = TikzTable(districts)\ntable.set_decimal_count(2)\ntable.add_vrule_right_of(0)\ntable.add_hrule_above([4])\ntable.add_toprule()\ntable.add_bottomrule()",
    "tikztable-borders": 'table = TikzTable(districts)\ntable.set_decimal_count(2)\ntable.set_cell_space_limits("2pt")\ntable.highlight_rows([3, 4], color="lightblue!20!white")\ntable.set_cell_border(5, [2, 3, 4, 5], "top")\ntable.set_cell_border(6, [2, 3, 4, 5], "bottom")\ntable.set_cell_border([5, 6], 2, "left")\ntable.set_cell_border([5, 6], 5, "right")',
    "tikztable-diverging": 'table = TikzTable(districts)\ntable.set_decimal_count(2)\ntable.set_column_formatter(\n    "Dem share",\n    compose_formatters(\n        diverging_gradient_formatter(\n            lo=0.35,\n            mid=0.50,\n            hi=0.65,\n            color_lo="alizarin",\n            color_mid="white",\n            color_hi="denim",\n            command_name=None,\n        ),\n        round_decimals(2),\n    ),\n)',
    "tikztable-heatmap": 'table = TikzTable(districts)\ntable.set_nan_string("---")\ntable.set_number_formatter(\n    compose_formatters(wrap_with_tex_command("heatmap"), round_decimals(2))\n)\ntable.document.add_command(tex_twocolor_gradient_command("heatmap"))',
    "tikztable-draws": 'table = TikzTable(districts)\ntable.set_decimal_count(2)\ntable.set_table_name("diagnostics")\ntable.document.add_color("framecolor", "denim")\ntable.add_draw(\n    r"\\draw[framecolor, rounded corners=2pt, line width=0.8pt] "\n    r"(diagnostics-5-2.north west) rectangle "\n    r"(diagnostics-6-5.south east);"\n)',
    "tikztable-showcase": 'table = TikzTable(districts)\ntable.set_header_groups(\n    {\n        "": ["District"],\n        "Representation": ["BVAP share", "Dem share"],\n        "Diagnostics": ["Polsby-Popper", "Pop. deviation"],\n    }\n)\ntable.set_cell_space_limits("1.5pt")\ntable.set_column_formatter(\n    "Dem share",\n    compose_formatters(\n        diverging_gradient_formatter(\n            lo=0.35,\n            mid=0.50,\n            hi=0.65,\n            color_lo="alizarin",\n            color_mid="white",\n            color_hi="denim",\n            command_name=None,\n        ),\n        round_decimals(2),\n    ),\n)\ntable.set_column_formatter(\n    "Pop. deviation",\n    compose_formatters(\n        highlight_ge(0.006, color="amber!35!white"),\n        round_decimals(3),\n    ),\n)\ntable.set_cell_border([6, 8], 5, "all")\ntable.add_toprule()\ntable.add_bottomrule()',
    "latex-paintball": 'from gerrytools.latex import TikzPaintballPlot\n\nseats = [13, 15, 17, 21, 21, 14, 14, 15, 13, 16, 24, 24, 24, 24]\nseat_shares = [s / 29 for s in seats]\nvote_shares = [0.5231, 0.4782, 0.5057, 0.5674, 0.5865, 0.5007, 0.5020, 0.5174, 0.4996, 0.5169, 0.5977, 0.5948, 0.5930, 0.5650]\n\nplot = TikzPaintballPlot(vote_shares, seat_shares)\nplot.add_efficiency_gap_line()\nplot.add_proportionality_line()\nplot.set_marker_options(color="cadmiumgreen")',
    "latex-paintball-hull": 'plot.set_hull_options(\n    color="denim",\n    alpha=0.25,\n    edgecolor="denim",\n    edgewidth=1.5,\n)',
    "latex-seatsvotes": 'from gerrytools.latex import TikzSeatsVotesPlot\n\ndemocratic_votes = np.array([42_000, 48_000, 53_000, 57_000, 62_000, 68_000])\ntotal_votes = np.array([100_000, 99_000, 101_000, 98_000, 102_000, 100_000])\n\nplot = TikzSeatsVotesPlot()\nplot.add_election(\n    democratic_votes,\n    total_votes,\n    linecolor="denim",\n    markercolor="alizarin",\n    marker_label="Observed result",\n)\nplot.add_proportionality_line()\nplot.add_efficiency_gap_line()',
}

IMPORTS = """
from gerrytools.latex import TexTable, TikzPaintballPlot, TikzSeatsVotesPlot, TikzTable
from gerrytools.latex.commands import tex_twocolor_gradient_command
from gerrytools.latex.formatters import (
    compose_formatters,
    diverging_gradient_formatter,
    highlight_between,
    highlight_ge,
    round_decimals,
    wrap_with_tex_command,
)
"""


def main() -> None:
    namespace: dict = {}
    exec(IMPORTS, namespace)
    exec(SETUP, namespace)
    for name, snippet in SNIPPETS.items():
        exec(snippet, namespace)
        target = (
            namespace.get("table")
            if name.startswith(("textable", "tikztable"))
            else namespace.get("plot")
        )
        document = target.hull_document if name == "latex-paintball-hull" else target.document
        out = IMAGE_DIR / f"{name}.png"
        document.save_png(out)
        print("wrote", out.name)


if __name__ == "__main__":
    main()
