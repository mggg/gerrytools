# Colors

The colors module provides one color vocabulary for Matplotlib plots, geographic layers, and
LaTeX output. GerryTools accepts ordinary Matplotlib colors and adds named palettes, deterministic
district colors, xcolor-style mixtures, and palette preview helpers.

Most plotting methods accept any of these forms directly:

- a Matplotlib name such as `"steelblue"`;
- a LaTeX Colors name such as `"denim"`;
- a GerryTools name such as `"ensemble:forest"`;
- a hex string such as `"#0064bd"`; or
- an RGB or RGBA tuple.

That means a named color can move from TeX to Python without maintaining a separate set of hex 
values. The LaTeX module of GerryTools also maintains the reverse compatibility through this
color module so it remains simple to copy-paste output from GerryTools into active TeX documents.

## Resolve a named color

```python
from gerrytools.colors import get_named_color

get_named_color("cc:denim")
```

The `cc:` names form a compact color-corrected set. The `ensemble:` names provide stable colors
for the supported ensemble methods, while the unprefixed registry also includes the Districtr,
LaTeX, and Matplotlib named colors.

Some names occur in more than one source. `which_color_source()` reports which definition wins,
and `get_all_supported_colors_dict()` returns the complete resolved mapping.

```python
from gerrytools.colors import which_color_source

which_color_source("salmon")
```

`convert_color_to_hexa_or_none()` normalizes any supported input to an eight-digit hex value.
It also understands xcolor-style mixtures. For example, `"cc:denim!70!white"` mixes 70 percent
denim with 30 percent white.

## Create a district palette

```python
from gerrytools.colors import districtr

district_colors = districtr(8)
```

`districtr(n)` returns exactly `n` colors in a stable order, making district assignments visually
consistent across related maps. For a plan whose labels are not consecutive integers, build an
explicit label-to-color mapping rather than using the labels as list positions.

```python
district_labels = ["A", "B", "C", "D"]
district_color = dict(zip(district_labels, districtr(len(district_labels))))
```

## Sequential and diverging palettes

Use a sequential palette when magnitude increases in one direction. `greens()`, `purples()`, and
`flare()` return a requested number of ordered colors. Use `redbluecmap()` or
`greenpurplecmap()` when values diverge around a meaningful midpoint.

```python
from gerrytools.colors import greenpurplecmap, purples

population_colors = purples(5)
deviation_colors = greenpurplecmap(7)
```

## Preview palettes

```python
from gerrytools.colors import compare_palettes, greenpurplecmap, redbluecmap

compare_palettes(
    {
        "red-blue": redbluecmap(7),
        "green-purple": greenpurplecmap(7),
    }
)
```

`preview_palette()` displays one sequence. `compare_palettes()` aligns several sequences in rows,
which is useful when choosing a palette or checking that related figures use compatible colors.
Both functions return their Matplotlib figure and axes, so the preview can be embedded or saved
like any other figure.

The {doc}`colors API <../api/colors>` lists the named-color dictionaries, sequential palettes,
resolution helpers, and preview options.
