# LaTeX output

GerryTools builds report-ready TeX source in Python: tables from DataFrames and TikZ-native
plots. Inspecting source needs no compiler; previews and exports use a local LaTeX
installation, and every rendered example in these guides is a committed image, so the
documentation itself never compiles TeX.

## Which table class?

There are two table classes with **one shared builder API**: every option, formatter, and
rule call works identically on both. They exist because they emit different TeX dialects:

- **`TexTable`** emits a plain `tabular`. Minimal dependencies, one compile pass, and
  source that pastes into any report, journal, or filing template you don't control.
  Reach for it by default.
- **`TikzTable`** emits a `nicematrix` `NiceTabular`, which requires the `nicematrix` and
  `tikz` packages and two compile passes. In exchange you get a cell-position lattice:
  per-cell borders, free-form `\draw` commands, and cell-geometry control that a plain
  `tabular` cannot express.

If you don't need to draw on the table, use `TexTable`; switching later is a one-word
change because the APIs match.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`table` TexTable
:link: textable
:link-type: doc

The default choice: a plain `tabular` that compiles anywhere. Rules, header groups,
highlights, formatters, and gradients.
:::

:::{grid-item-card} {octicon}`apps` TikzTable basics
:link: tikztable
:link-type: doc

Same API, different dialect: a `nicematrix` table you can draw on, at the cost of two
compile passes.
:::

:::{grid-item-card} {octicon}`paintbrush` TikzTable styling
:link: tikztable_styling
:link-type: doc

Cell borders, gradient fills, and free-form TikZ drawing.
:::

:::{grid-item-card} {octicon}`dot-fill` Paintball plots
:link: paintball
:link-type: doc

The vote-seat cloud and its hull as native TikZ.
:::

:::{grid-item-card} {octicon}`graph` Seats-votes plots
:link: seats_votes
:link-type: doc

Uniform-swing step curves with benchmark lines as native TikZ.
:::

:::{grid-item-card} {octicon}`code` LaTeX API
:link: ../../api/latex
:link-type: doc

Signatures for every public table, plot, formatter, and command helper.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

TexTable <textable>
TikzTable basics <tikztable>
TikzTable styling <tikztable_styling>
Paintball plots <paintball>
Seats-votes plots <seats_votes>
```
