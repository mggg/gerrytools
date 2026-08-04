import random
import re

from gerrytools.typing import HexColor

_HEX_DIGITS = "0123456789abcdef"

_HEX_COLOR_PATTERN = re.compile(r"#?([0-9a-fA-F]{6})")

DISTRICTR_COLOR_DICT = {
    "tombblue": "#0099cd",
    "nacho": "#ffca5d",
    "carribeangreen": "#00cd99",
    "viricgreen": "#99cd00",
    "indianlake": "#cd0099",
    "violetink": "#9900cd",
    "greendaze": "#8dd3c7",
    "playfulpurple": "#bebada",
    "smokedsalmon": "#fb8072",
    "meadowblossomblue": "#80b1d3",
    "mangocreamsicles": "#fdb462",
    "lastoflettuce": "#b3de69",
    "classicrose": "#fccde5",
    "lavendersweater": "#bc80bd",
    "mintfrappe": "#ccebc5",
    "yellowsand": "#ffed6f",
    "vic20creme": "#ffffb3",
    "bluecalico": "#a6cee3",
    "bumanguesblue": "#1f78b4",
    "sagesensation": "#b2df8a",
    "dryhighlightergreen": "#33a02c",
    "rubberradish": "#fb9a99",
    "akirared": "#e31a1c",
    "cinnamonbuff": "#fdbf6f",
    "orangejuice": "#ff7f00",
    "elfinherb": "#cab2d6",
    "poppypompadour": "#6a3d9a",
    "stirlandbattlemire": "#b15928",
    "spindrift": "#64ffda",
    "maldives": "#00b8d4",
    "velvetychestnut": "#a1887f",
    "radium": "#76ff03",
    "mindaro": "#dce775",
    "lilacgeode": "#b388ff",
    "informativepink": "#ff80ab",
    "exoticliras": "#d81b60",
    "tropicalhideaway": "#26a69a",
    "middleyellow": "#ffea00",
    "gonzoviolet": "#6200ea",
}


def hexshift(color: str, *, seed: int = 42) -> str:
    """Deterministically modify the provided hexadecimal color.

    Picks one hex digit occurring in the color and replaces *every* occurrence of it with a
    different hex digit, so the result is always a different color value. Input case is
    ignored; the result is lowercase with a leading ``#``.

    Args:
        color (str): A six-digit hexadecimal color string, with or without the leading ``#``;
            e.g. `"#FFFF00"`.
        seed (int): Seed for the deterministic digit substitution. Defaults to 42.

    Returns:
        str: A lowercase hexadecimal color string different from ``color``.

    Raises:
        ValueError: If ``color`` is not a six-digit hexadecimal color string.
    """
    match = _HEX_COLOR_PATTERN.fullmatch(color) if isinstance(color, str) else None
    if match is None:
        raise ValueError(
            f"color must be a six-digit hex color string like '#ffca5d', "
            f"with or without the leading '#'; got {color!r}"
        )
    normalized_color = f"#{match.group(1).lower()}"
    rng = random.Random(seed)

    sub = rng.choice(_HEX_DIGITS)
    char = rng.choice(normalized_color[1:])

    # Redraw until the substitute differs from the chosen digit, so the replacement always
    # changes the color value.
    while sub == char:
        sub = rng.choice(_HEX_DIGITS)

    return normalized_color.replace(char, sub)


def districtr(N: int) -> list[HexColor]:
    """Returns a list of N hex colors from the districtr palette.

    When ``N`` exceeds the number of base palette colors, the palette is extended by appending hex-
    shifted variants of the base colors.

    Args:
        N (int): The number of colors to return.

    Returns:
        list[HexColor]: A list of ``N`` hex color strings from the districtr
        palette.

    Raises:
        ValueError: If ``N`` is negative.
        RuntimeError: If the requested number of distinct colors cannot be generated.
    """

    if N < 0:
        raise ValueError("N must be nonnegative.")

    colors = list(DISTRICTR_COLOR_DICT.values())
    if N <= len(colors):
        return colors[:N]

    # Vary the seed per shift: hexshift is deterministic for a fixed seed, so reusing one seed
    # would make every extension round identical. Skip any shift that collides with a color already
    # in the palette, and fail loudly rather than pad with silent duplicates.
    seen = set(colors)
    extended = list(colors)
    seed = 42
    remaining_attempts = 100 * N
    while len(extended) < N:
        for color in colors:
            if remaining_attempts <= 0:
                raise RuntimeError(
                    f"Could not generate {N} distinct districtr colors after {100 * N} "
                    f"hexshift attempts; got {len(extended)}."
                )
            shifted = hexshift(color, seed=seed)
            seed += 1
            remaining_attempts -= 1
            if shifted not in seen:
                seen.add(shifted)
                extended.append(shifted)
                if len(extended) == N:
                    break
    return extended
