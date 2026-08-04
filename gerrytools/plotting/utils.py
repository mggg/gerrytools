import dataclasses
import logging
import math
from collections.abc import Iterable
from numbers import Real
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger

# Re-exported so plotting callers keep importing the sentinel from here; the single definition
# lives in the dependency-light typing hub and is shared with gerrytools.latex.
from gerrytools.typing import UNSET as UNSET
from gerrytools.typing import Color, Numeric, NumericArrayLike, NumericIterable
from gerrytools.typing import Unset as Unset

logger = get_logger(__name__)

DataclassT = TypeVar("DataclassT")


def _validated_finite(value: object, *, field: str) -> float:
    """Validate a finite numeric option field, returning it as a float.

    Args:
        value (object): The value to validate.
        field (str): Dotted field name used in error messages, e.g. ``"TitleStyle.pad"``.

    Raises:
        TypeError: If ``value`` is not an int or float (bools are rejected).
        ValueError: If ``value`` is not finite.
    """
    # bool is an int subclass, but True/False are never meant as numeric options.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a float or int.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


def _validated_nonneg_finite(value: object, *, field: str) -> float:
    """Validate a nonnegative finite numeric option field, returning it as a float.

    Args:
        value (object): The value to validate.
        field (str): Dotted field name used in error messages, e.g. ``"TickStyle.size"``.

    Raises:
        TypeError: If ``value`` is not an int or float (bools are rejected).
        ValueError: If ``value`` is not finite or is negative.
    """
    number = _validated_finite(value, field=field)
    if number < 0:
        raise ValueError(f"{field} must be nonnegative.")
    return number


def _resolve_color_clamped_width(
    color: Color | None,
    alpha: float | None,
    width: float,
    *,
    color_field: str,
    width_field: str,
    owner: str,
    log: logging.Logger | None = None,
) -> tuple[Color, float, float]:
    """Resolve a color/alpha pair and clamp its paired stroke width.

    The one shared rule for every "color plus width" styling pair (line, edge, outline):
    a color that resolves to ``"none"`` means the stroke is invisible, so a positive width
    is clamped to 0 rather than drawing nothing at nonzero width.

    Returns:
        tuple[Color, float, float]: Resolved color, resolved alpha, and the (possibly
        clamped) width.
    """
    active_logger = log if log is not None else logger
    resolved_color, resolved_alpha = resolve_color_and_alpha(
        color,
        alpha,
        allow_none=True,
        field=color_field,
        owner=owner,
        logger=active_logger,
    )
    if resolved_color.lower() == "none" and width > 0:
        active_logger.log(
            level=logging.DEBUG,
            msg=(
                f"{owner}: {color_field} is 'none' but {width_field} is {width}>0; "
                f"setting {width_field} to 0."
            ),
        )
        width = 0.0
    return resolved_color, resolved_alpha, width


def _coerce_real_iter(values: Numeric | NumericIterable, *, field: str) -> list[float]:
    """Normalize scalar/iterable numeric input to a list of floats.

    Args:
        values (Numeric | NumericIterable): Scalar real value or iterable of real values.
        field (str): Field name used in validation error messages.

    Returns:
        list[float]: Parsed numeric values as Python floats.

    Raises:
        TypeError: If ``values`` is not numeric or contains non-numeric entries.
    """
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a number or an iterable of numbers, not a string.")
    # bool is a Real, but treating True/False as 1.0/0.0 is never what callers mean.
    if isinstance(values, bool):
        raise TypeError(f"{field} must be a number or an iterable of numbers, not a bool.")
    if isinstance(values, Real):
        return [float(values)]
    if isinstance(values, Iterable):
        out: list[float] = []
        for v in values:
            if isinstance(v, bool) or not isinstance(v, Real):
                raise TypeError(
                    f"All items in {field} must be real numbers; got {type(v).__name__}."
                )
            out.append(float(v))
        return out
    raise TypeError(f"{field} must be a number or an iterable of numbers.")


def _resolve_alpha_override(
    color_given: bool,
    alpha: float | None,
    base_color: object,
    base_alpha: float | None,
) -> float | None:
    """Resolve the alpha of a color/alpha kwarg pair against resolved base options.

    A caller's explicit alpha always wins, and without a color override the base alpha is inherited
    as usual. When only the color is overridden, the base alpha is still kept (it may have been set
    deliberately in an options dataclass) unless the base color is the fully transparent "none": its
    resolved alpha of 0.0 would render the override invisibly, so the alpha instead derives from the
    override color itself (``None``).

    Args:
        color_given (bool): Whether the caller explicitly passed a color override.
        alpha (float | None): The alpha kwarg passed by the caller, or None if not given.
        base_color (object): The (already resolved) color from the base options.
        base_alpha (float | None): The (already resolved) alpha from the base options.

    Returns:
        float | None: The alpha to style with.
    """
    if alpha is not None:
        return alpha
    if not color_given:
        return base_alpha
    if base_color is None or (isinstance(base_color, str) and base_color.lower() == "none"):
        return None
    return base_alpha


def _replace_non_none(options: DataclassT, **overrides: object) -> DataclassT:
    """Copy a dataclass, applying only the overrides that are not None.

    Shared by ``add_*`` methods that accept both a pre-built options dataclass and individual
    override kwargs: an explicit kwarg wins, while ``None`` means "keep the value from the options
    dataclass." The merged result goes through the dataclass's ``__init__``/``__post_init__``, so
    field validation is re-applied.

    This is the merge for options where ``None`` is never a meaningful field value; the
    face/edge options classes instead use ``_FaceEdgeStyle.merged``, whose ``UNSET`` sentinel
    lets an explicit ``None`` color mean "none".

    Args:
        options (DataclassT): The base options dataclass instance.
        **overrides (object): Field overrides; entries that are None are ignored.

    Returns:
        DataclassT: A new instance of the same dataclass type with overrides applied.
    """
    field_updates = {name: value for name, value in overrides.items() if value is not None}
    return cast(DataclassT, dataclasses.replace(cast(Any, options), **field_updates))


def _replace_with_color_overrides(
    options: DataclassT,
    *color_alpha_fields: tuple[str, str],
    **overrides: object,
) -> DataclassT:
    """Copy an options dataclass while distinguishing omitted colors from ``None``.

    Non-color fields retain the usual ``None``-means-inherit behavior. Each named color uses
    ``UNSET`` for inheritance, while an explicit ``None`` becomes Matplotlib's transparent
    ``"none"`` color. Paired alpha fields follow :func:`_resolve_alpha_override`.

    Args:
        options (DataclassT): Base options dataclass.
        *color_alpha_fields (tuple[str, str]): ``(color_field, alpha_field)`` pairs.
        **overrides (object): Explicit field overrides.

    Returns:
        DataclassT: A validated copy with the explicit overrides applied.
    """
    paired_names = {name for pair in color_alpha_fields for name in pair}
    updates = {
        name: value
        for name, value in overrides.items()
        if name not in paired_names and value is not None and not isinstance(value, Unset)
    }
    for color_field, alpha_field in color_alpha_fields:
        color = overrides.get(color_field, UNSET)
        alpha = overrides.get(alpha_field)
        color_given = not isinstance(color, Unset)
        if not color_given and alpha is None:
            continue
        base_color = getattr(options, color_field)
        base_alpha = getattr(options, alpha_field)
        updates[color_field] = (
            base_color if not color_given else ("none" if color is None else color)
        )
        updates[alpha_field] = _resolve_alpha_override(
            color_given,
            cast("float | None", alpha),
            base_color,
            cast("float | None", base_alpha),
        )
    if not updates:
        return cast(DataclassT, dataclasses.replace(cast(Any, options)))
    return cast(DataclassT, dataclasses.replace(cast(Any, options), **updates))


def _coerce_to_1d_float_array(
    values: NumericArrayLike, *, column: str | None = None, field: str
) -> NDArray[np.float64]:
    """Coerce various inputs into a 1D float ndarray (no finite-filtering).

    Args:
        values (NumericArrayLike): Input values. Supported forms include scalar numerics, iterables,
            numpy arrays, pandas Series, and pandas DataFrames.
        column (str | None, optional): DataFrame column name to extract when ``values``
            is a DataFrame. Defaults to None.
        field (str): Field name used in validation error messages.

    Returns:
        NDArray[np.float64]: One-dimensional float array.
    """
    if values is None:
        raise ValueError(f"{field}: cannot be None.")

    if isinstance(values, pd.DataFrame):
        if column is None:
            if values.shape[1] != 1:
                raise ValueError(
                    f"{field}: DataFrame input must have exactly one column or pass column=..."
                )
            ser = values.iloc[:, 0]
        else:
            if column not in values.columns:
                raise ValueError(f"{field}: column {column!r} not found in DataFrame.")
            ser = values[column]
        arr = ser.to_numpy(dtype=float)
    elif isinstance(values, pd.Series):
        arr = values.to_numpy(dtype=float)
    elif np.isscalar(values):
        # bool passes np.isscalar, but treating True/False as 1.0/0.0 is never what
        # callers mean (matches _coerce_real_iter).
        if isinstance(values, bool):
            raise TypeError(f"{field}: expected numeric values, not a bool.")
        arr = np.array([values], dtype=float)
    elif isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=float)
    elif isinstance(values, (list, tuple)):
        arr = np.asarray(values, dtype=float)
    else:
        if not isinstance(values, Iterable):
            raise TypeError(
                f"{field}: expected an iterable of numeric values, got {type(values).__name__!r}."
            )
        # generators/iterators need materializing for numpy coercion
        arr = np.asarray(list(values), dtype=float)

    arr = np.asarray(arr, dtype=float).ravel()
    return arr


def _coerce_values_and_weights(
    values: NumericArrayLike,
    *,
    weights: NumericArrayLike | None,
    column: str | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce values and weights while preserving alignment through shared masking.

    Args:
        values (NumericArrayLike): Histogram values input.
        weights (NumericArrayLike | None): Optional weights input aligned to ``values``.
        column (str | None): DataFrame column name for ``values`` when applicable.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Finite values and matching
            weights arrays.

    Raises:
        ValueError: If values are empty/non-finite or weights are length-incompatible.
    """
    vals_raw = _coerce_to_1d_float_array(values, column=column, field="values")
    if vals_raw.size == 0:
        raise ValueError("values: must have at least one entry.")

    mask = np.isfinite(vals_raw)
    vals = vals_raw[mask]
    if vals.size == 0:
        raise ValueError("values: must have at least one finite entry.")

    if weights is None:
        wts = np.ones(vals.shape[0], dtype=float)
    else:
        w_raw = _coerce_to_1d_float_array(weights, column=None, field="weights")
        if w_raw.size != vals_raw.size:
            raise ValueError("weights must have the same length as values (before filtering).")

        wts = w_raw[mask]

        if not np.all(np.isfinite(wts)):
            raise ValueError("weights must be finite wherever values are finite.")

    return vals, wts


def _coerce_to_1d_finite_float_array(
    values: NumericArrayLike, *, column: str | None = None, field: str
) -> NDArray[np.float64]:
    """Coerce various inputs into a finite 1D float ndarray.

    Args:
        values (NumericArrayLike): Input values. Supported forms include scalar numerics, iterables,
            numpy arrays, pandas Series, and pandas DataFrames.
        column (str | None, optional): DataFrame column name to extract when ``values``
            is a DataFrame. Defaults to None.
        field (str): Field name used in validation error messages.

    Returns:
        NDArray[np.float64]: 1D ndarray of finite float values.

    Raises:
        ValueError: If input cannot be coerced to 1D float array.
    """
    arr = _coerce_to_1d_float_array(values, column=column, field=field)
    return arr[np.isfinite(arr)]
