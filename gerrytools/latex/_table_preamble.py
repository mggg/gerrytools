"""Helpers for parsing and formatting tabular preambles."""

from __future__ import annotations


def _infer_group_cell_align_from_data(colspecs: list[str], start: int, end: int) -> str:
    """Infer group-header alignment from spanned data column specs.

    Heuristic:
      - if all are 'l' -> 'l'
      - if all are 'r' -> 'r'
      - if all are 'c' -> 'c'
      - else -> 'c'

    Args:
        colspecs (list[str]): Parsed tabular column specification tokens.
        start (int): Inclusive start index of the spanned range.
        end (int): Exclusive end index of the spanned range.

    Returns:
        str: Alignment token ``"l"``, ``"c"``, or ``"r"``.
    """

    def base_align(spec: str) -> str:
        """Extract a base alignment token from a column spec.

        Args:
            spec (str): One parsed column spec token.

        Returns:
            str: ``"l"``, ``"c"``, or ``"r"`` if explicitly present; otherwise ``"c"``.
        """
        spec = spec.strip()
        if spec in ("l", "c", "r"):
            return spec
        return "c"

    aligns = {base_align(s) for s in colspecs[start:end]}
    return aligns.pop() if len(aligns) == 1 else "c"


def _consume_balanced(s: str, i: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    """Consume a balanced delimited group from a string.

    Args:
        s (str): Source string.
        i (int): Index where ``open_ch`` is expected.
        open_ch (str): Opening delimiter character.
        close_ch (str): Closing delimiter character.

    Returns:
        tuple[str, int]: The consumed content (without delimiters) and the next index.

    Raises:
        ValueError: If delimiters are missing or unbalanced.
    """
    if i >= len(s) or s[i] != open_ch:
        raise ValueError(f"Expected '{open_ch}' at position {i}")
    depth = 1
    i += 1
    start = i

    while i < len(s) and depth:
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
        i += 1

    if depth != 0:
        raise ValueError(f"Unbalanced {open_ch}{close_ch} in format string")

    return s[start : i - 1], i


def _parse_tabular_preamble(fmt: str) -> tuple[list[str], list[int], list[str]]:
    """Parse a LaTeX tabular preamble.

    Args:
        fmt (str): Raw tabular preamble string (for example ``"|l|c|r|"``).

    Returns:
        tuple[list[str], list[int], list[str]]: Parsed column specs, vertical-rule counts, and
        boundary-extra strings.

    Raises:
        ValueError: If ``fmt`` contains unsupported tokens or unbalanced delimiters.
    """
    i, n = 0, len(fmt)
    colspecs: list[str] = []
    vrules: list[int] = [0]
    extras: list[str] = [""]

    def skip_ws(i: int) -> int:
        """Advance an index past whitespace in the preamble string.

        Args:
            i (int): Starting index in ``fmt``.

        Returns:
            int: First index at or after ``i`` that is not whitespace.
        """
        while i < n and fmt[i].isspace():
            i += 1
        return i

    simple_cols = set("lcr")

    while True:
        i = skip_ws(i)
        if i >= n:
            break

        ch = fmt[i]

        if ch in "{}[]":
            raise ValueError(f"Stray {ch!r} at pos {i} in preamble: {fmt!r}")

        if ch == "|":
            vrules[-1] += 1
            i += 1
            continue

        if ch in ("@", "!", ">", "<"):
            if i + 1 >= n or fmt[i + 1] != "{":
                raise ValueError(f"Expected '{{' after {ch} at pos {i} in preamble: {fmt!r}")
            grp, i = _consume_balanced(fmt, i + 1, "{", "}")
            extras[-1] += f"{ch}{{{grp}}}"
            continue

        if ch in ("p", "m", "b"):
            if i + 1 >= n or fmt[i + 1] != "{":
                raise ValueError(f"Expected '{{' after {ch} at pos {i} in preamble: {fmt!r}")
            tok, i = _consume_balanced(fmt, i + 1, "{", "}")
            colspecs.append(f"{ch}{{{tok}}}")
            vrules.append(0)
            extras.append("")
            continue

        if ch == "S":
            i += 1
            tok = "S"
            i = skip_ws(i)
            if i < n and fmt[i] == "[":
                grp, i = _consume_balanced(fmt, i, "[", "]")
                tok += f"[{grp}]"
            colspecs.append(tok)
            vrules.append(0)
            extras.append("")
            continue

        if ch == "D":
            i += 1
            i = skip_ws(i)
            if i >= n or fmt[i] != "{":
                raise ValueError(f"Expected '{{' after D at pos {i} in preamble: {fmt!r}")
            g1, i = _consume_balanced(fmt, i, "{", "}")
            i = skip_ws(i)
            g2, i = _consume_balanced(fmt, i, "{", "}")
            i = skip_ws(i)
            g3, i = _consume_balanced(fmt, i, "{", "}")
            colspecs.append(f"D{{{g1}}}{{{g2}}}{{{g3}}}")
            vrules.append(0)
            extras.append("")
            continue

        if ch in simple_cols:
            colspecs.append(ch)
            vrules.append(0)
            extras.append("")
            i += 1
            continue

        # Falling through without consuming input would make this loop spin forever.
        raise ValueError(f"Unsupported token {fmt[i]!r} at pos {i} in preamble: {fmt!r}")

    return colspecs, vrules, extras
