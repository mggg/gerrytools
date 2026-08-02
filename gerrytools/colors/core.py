import logging
import math

import matplotlib.colors as mcolors

from gerrytools.colors._sources import (
    CITIZEN_BLUE,
    COLOR_CORRECTED_BASESET,
    DEFAULT_GREY,
    ENSEMBLE_COLORS,
    GERRYTOOLS_EXTRA_COLORS_DICT,
    OVERLAYS,
    get_all_supported_colors_dict,
    get_named_color,
    which_color_source,
)
from gerrytools.colors._value import _Color, validate_alpha
from gerrytools.logging import get_logger
from gerrytools.typing import HexColor, MplCompatibleColor, ResolvedColor

__all__ = [
    "CITIZEN_BLUE",
    "COLOR_CORRECTED_BASESET",
    "DEFAULT_GREY",
    "ENSEMBLE_COLORS",
    "GERRYTOOLS_EXTRA_COLORS_DICT",
    "OVERLAYS",
    "convert_color_to_hexa_or_none",
    "get_all_supported_colors_dict",
    "get_named_color",
    "resolve_color_and_alpha",
    "resolve_rgba",
    "which_color_source",
]

gt_logger = get_logger(__name__)


def convert_color_to_hexa_or_none(color: MplCompatibleColor | None) -> HexColor:
    """Convert a color input to a hex8 string or "none".

    Args:
        color (MplCompatibleColor | None): The color input to convert. This can be a named color,
            a LaTeX color string, an RGB(A) tuple, or other formats supported
            by Matplotlib.

    Returns:
        HexColor: Lowercase ``#rrggbbaa`` color, or ``"none"`` for a transparent color.

    Raises:
        ValueError: If ``color`` is not a recognized color value.
    """
    return _Color.from_any(color, logger=gt_logger).to_hex8()


def resolve_color_and_alpha(
    color: MplCompatibleColor | None,
    alpha: float | None = None,
    *,
    allow_none: bool = True,
    field: str = "color",
    owner: str | None = None,
    logger: logging.Logger | None = None,
) -> ResolvedColor:
    """Normalize a (color, alpha) pair into (hex6_or_none, resolved_alpha).

    The color is first converted to ``"none"`` or ``"#RRGGBBAA"``. If it resolves to ``"none"``, the
    function returns ``("none", 0.0)`` when ``allow_none`` is ``True`` and raises ``ValueError``
    otherwise. For an RGBA color, it returns the RGB value with either the embedded alpha or the
    validated explicit alpha. An explicit alpha is validated even when the color resolves to
    ``"none"``. An optional debug message records when an explicit value overrides the embedded
    alpha.

    Args:
        color (MplCompatibleColor | None): The color input to convert.
        alpha (float | None, optional): Explicit alpha value between 0.0 and 1.0. Defaults to None.
        allow_none (bool): Whether "none" is an acceptable color. Defaults to True.
        field (str, optional): Name of the field being processed. Defaults to ``"color"``.
        owner (str | None, optional): Owner name for logging context. Defaults to None.
        logger (logging.Logger | None, optional): Logger for parse-failure diagnostics and alpha
            override notices. Defaults to None, which uses this module's logger for parse failures.

    Returns:
        ResolvedColor: A tuple of (hex6_or_none, resolved_alpha).

    Raises:
        TypeError: If an explicit alpha is not a real number.
        ValueError: If the color is unknown, alpha is invalid, or ``allow_none`` rejects it.
    """
    resolved_color = _Color.from_any(color, logger=logger if logger is not None else gt_logger)
    # Validate before the "none" early return so a bad explicit alpha never passes silently.
    validated_alpha = None if alpha is None else validate_alpha(alpha, field=f"{field} alpha")

    if resolved_color.is_none:
        if not allow_none:
            raise ValueError(f"{field} cannot be 'none'.")
        return "none", 0.0

    if validated_alpha is None:
        return resolved_color.hex6, resolved_color.alpha

    explicit_alpha_overrides_embedded = not math.isclose(
        validated_alpha, resolved_color.alpha, abs_tol=1e-4
    )
    if logger is not None and explicit_alpha_overrides_embedded:
        owner_prefix = f"For {owner}: " if owner else ""
        logger.log(
            level=logging.DEBUG,
            msg=(
                f"{owner_prefix}Ignoring alpha embedded in {field} "
                f"{resolved_color.to_hex8()!r} because explicit alpha "
                f"{validated_alpha} was provided."
            ),
        )

    return resolved_color.hex6, validated_alpha


def resolve_rgba(
    color: MplCompatibleColor | None,
    alpha: float | None = None,
    *,
    field: str = "color",
    owner: str = "gerrytools",
) -> tuple[float, float, float, float]:
    """Resolve a ``Color`` plus optional alpha override to an RGBA tuple.

    Args:
        color (MplCompatibleColor | None): GerryTools color input. ``None`` resolves to
            the fully transparent ``"none"`` color.
        alpha (float | None, optional): Optional alpha override. Defaults to None.
        field (str, optional): Field name used in validation and warning messages.
            Defaults to ``"color"``.
        owner (str, optional): Owner name used in log messages. Defaults to ``"gerrytools"``.

    Returns:
        tuple[float, float, float, float]: Resolved RGBA values in ``[0, 1]``.
    """
    resolved_color, resolved_alpha = resolve_color_and_alpha(
        color,
        alpha=alpha,
        allow_none=True,
        field=field,
        owner=owner,
        logger=gt_logger,
    )
    rgba = mcolors.to_rgba(resolved_color, alpha=resolved_alpha)
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
