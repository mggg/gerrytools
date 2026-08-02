"""Versioned configuration documents passed to sampler containers."""

import copy
import math
from dataclasses import fields
from typing import Any, Literal, NotRequired, TypedDict

from .constraints import ConstraintSpec


def check_finite_number(name: str, value: object) -> None:
    """Reject non-numeric, infinite, and NaN values at construction, before Docker starts.

    NaN in particular would serialize to a bare ``NaN`` token that strict container-side
    JSON parsers reject only after Docker has started.

    Args:
        name (str): Configuration field name used in the error message.
        value (object): Value to validate.

    Raises:
        ValueError: If ``value`` is a Boolean, non-numeric, infinite, or NaN.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, but found {value!r}.")


def check_finite_nonnegative(name: str, value: float) -> None:
    """Reject non-numeric, negative, infinite, and NaN values before Docker starts.

    Args:
        name (str): Configuration field name used in the error message.
        value (float): Value to validate.

    Raises:
        ValueError: If ``value`` is not finite and nonnegative.
    """
    check_finite_number(name, value)
    if value < 0:
        raise ValueError(f"{name} must be finite and nonnegative, but found {value!r}.")


def check_unit_interval(name: str, value: float) -> None:
    """Reject non-numeric values and values outside ``[0, 1]`` before Docker starts.

    Args:
        name (str): Configuration field name used in the error message.
        value (float): Value to validate.

    Raises:
        ValueError: If ``value`` is not a finite number in ``[0, 1]``.
    """
    check_finite_number(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one, but found {value!r}.")


def check_integer(name: str, value: object) -> None:
    """Reject booleans and non-integers before Docker starts.

    Args:
        name (str): Configuration field name used in the error message.
        value (object): Value to validate.

    Raises:
        ValueError: If ``value`` is a Boolean or non-integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, but found {value!r}.")


def check_rng_seed(name: str, value: int, *, minimum: int, maximum: int) -> None:
    """Reject seeds outside the configured engine runtime's accepted range.

    Args:
        name (str): Configuration field name used in the error message.
        value (int): Seed to validate.
        minimum (int): Inclusive lower bound accepted by the engine.
        maximum (int): Inclusive upper bound accepted by the engine.

    Raises:
        ValueError: If ``value`` is not an integer within the bounds.
    """
    check_integer(name, value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, but found {value!r}.")


def check_positive_int(name: str, value: int) -> None:
    """Reject non-integers and values below one before Docker starts.

    Args:
        name (str): Configuration field name used in the error message.
        value (int): Value to validate.

    Raises:
        ValueError: If ``value`` is not a positive integer.
    """
    check_integer(name, value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, but found {value!r}.")


def check_boolean(name: str, value: object) -> None:
    """Reject non-boolean flag values before Docker starts.

    Args:
        name (str): Configuration field name used in the error message.
        value (object): Value to validate.

    Raises:
        ValueError: If ``value`` is not a Boolean.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, but found {value!r}.")


def check_nonempty_string(name: str, value: object) -> None:
    """Reject non-string and empty configuration values.

    Args:
        name (str): Configuration field name used in the error message.
        value (object): Value to validate.

    Raises:
        ValueError: If ``value`` is not a nonempty string.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, but found {value!r}.")


def check_string_list(name: str, value: object) -> None:
    """Reject non-lists and lists containing non-string or empty column names.

    Args:
        name (str): Configuration field name used in the error message.
        value (object): Value to validate.

    Raises:
        ValueError: If ``value`` is not a list of nonempty strings.
    """
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings, but found {value!r}.")


class EngineRunConfig(TypedDict):
    """Version-1 configuration envelope shared by the Forest and SMC containers."""

    version: Literal[1]
    engine: str
    io: dict[str, object]
    run: dict[str, object]
    constraints: list[ConstraintSpec]
    map: NotRequired[dict[str, object]]


NOT_ENGINE_CONFIG = {"engine_config": False}
"""Field metadata excluding a run-info field from the config's ``run`` section, either because
it is Python-side only (updaters, output naming) or because it is marshaled into another
section (``io``, ``map``, ``constraints``)."""


def dataclass_config(value: Any) -> dict[str, object]:
    """Project the dataclass fields destined for the config's ``run`` section.

    Fields carrying ``NOT_ENGINE_CONFIG`` metadata are omitted, which also avoids copying
    excluded callables such as updaters.

    Args:
        value (Any): Dataclass instance to project.

    Returns:
        dict[str, object]: Run-section fields and their current values.
    """
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.metadata.get("engine_config", True)
    }


def build_run_config(
    engine: str,
    *,
    io: dict[str, object],
    run: dict[str, object],
    constraints: list[ConstraintSpec],
    map_info: dict[str, object] | None = None,
) -> EngineRunConfig:
    """Build an independent version-1 effective run configuration.

    Args:
        engine (str): Engine identifier.
        io (dict[str, object]): Input, output, and writer configuration.
        run (dict[str, object]): Engine-specific run settings.
        constraints (list[ConstraintSpec]): Validated engine constraint specifications.
        map_info (dict[str, object] | None, optional): SMC map settings. Defaults to None.

    Returns:
        EngineRunConfig: Deep copy of the assembled configuration.
    """
    config = EngineRunConfig(
        version=1,
        engine=engine,
        io=io,
        run=run,
        constraints=constraints,
    )
    if map_info is not None:
        config["map"] = map_info
    return copy.deepcopy(config)
