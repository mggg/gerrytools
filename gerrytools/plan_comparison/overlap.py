"""Population and areal overlap tables for district plans."""

from collections.abc import Hashable, Iterable
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from gerrytools._geodataframe import (
    _object_index,
    _require_projected_crs,
    _require_valid_polygons,
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Require named, unambiguous columns; an index is not an implicit substitute."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing!r}")
    # A duplicated label makes frame[column] a DataFrame, which fails later with an opaque
    # pandas error, so reject it here by name.
    frame_columns = list(frame.columns)
    duplicated = [column for column in columns if frame_columns.count(column) > 1]
    if duplicated:
        raise ValueError(f"duplicate column labels: {duplicated!r}")


def _labels(values: pd.Series, name: str) -> pd.Index:
    """Return unique labels in first-seen order after rejecting missing values."""
    if values.isna().any():
        raise ValueError(f"{name} cannot contain missing labels")
    return _object_index(cast(Iterable[Hashable], pd.unique(values)), name)


def _weights(values: pd.Series, name: str) -> pd.Series:
    """Convert a population column to finite, nonnegative floating-point weights."""
    try:
        array = np.asarray(pd.to_numeric(values.to_numpy(), errors="raise"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if (array < 0).any():
        raise ValueError(f"{name} cannot contain negative values")
    return pd.Series(array, index=values.index, name=name)


def population_overlap(
    units: pd.DataFrame,
    *,
    source: str,
    target: str,
    population: str,
) -> pd.DataFrame:
    r"""Sum population for every realized source-target district pair.

    If :math:`s(u)` and :math:`t(u)` are the source and target assignments of unit :math:`u`,
    the returned matrix contains

    .. math::

        M_{ij} = \sum_{u : s(u)=i,\ t(u)=j} p(u).

    Rows follow source-label first-seen order and columns follow target-label first-seen order.
    Unobserved pairs are represented by zeros. The input is never modified.

    Args:
        units (pd.DataFrame): Unit-level table containing both assignments and a population weight.
        source (str): Column containing the plan labels that will be relabeled.
        target (str): Column containing the reference plan labels.
        population (str): Column containing finite, nonnegative population weights.

    Returns:
        pd.DataFrame: A floating-point DataFrame with source labels as its index and target labels
            as columns.

    Raises:
        ValueError: If a required column is absent, the assignment columns contain missing labels,
            or the population column contains nonnumeric, nonfinite, or negative values.
    """
    if len({source, target, population}) != 3:
        raise ValueError("source, target, and population must name different columns")
    _require_columns(units, (source, target, population))
    if units.empty:
        raise ValueError("units cannot be empty")
    source_values = cast(pd.Series, units[source])
    target_values = cast(pd.Series, units[target])
    population_values = cast(pd.Series, units[population])
    source_labels = _labels(source_values, source)
    target_labels = _labels(target_values, target)
    weights = _weights(population_values, population)

    records = pd.DataFrame(
        {
            "__source": source_values.to_numpy(),
            "__target": target_values.to_numpy(),
            "__population": weights.to_numpy(),
        }
    )
    grouped = cast(
        pd.Series,
        records.groupby(["__source", "__target"], observed=True, sort=False)["__population"].sum(),
    )
    matrix = grouped.unstack(fill_value=0.0)
    matrix = matrix.reindex(index=source_labels, columns=target_labels, fill_value=0.0).astype(
        np.float64
    )
    matrix.index.name = source
    matrix.columns.name = target
    return matrix


def areal_overlap(
    source: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    *,
    source_label: str,
    target_label: str,
    crs: Any | None = None,
) -> pd.DataFrame:
    r"""Measure the area shared by every source-target district pair.

    Each input must contain one valid, nonempty geometry per district label. Both inputs are
    transformed to ``crs`` when supplied. Otherwise, the target is transformed to the source CRS.
    The resulting CRS must be projected so the values are planar areas in its squared units.
    Geometries are validated both before and after the transformation.

    Args:
        source (gpd.GeoDataFrame): District geometries for the plan that will be relabeled.
        target (gpd.GeoDataFrame): District geometries for the reference plan.
        source_label (str): Unique district-label column in ``source``.
        target_label (str): Unique district-label column in ``target``.
        crs (Any | None, optional): Projected CRS in any form accepted by GeoPandas. Defaults to
            None, which transforms ``target`` to ``source.crs``.

    Returns:
        pd.DataFrame: A floating-point DataFrame with source labels as its index and target labels
            as columns. Nonintersecting pairs have value zero.

    Raises:
        TypeError: If either input is not a GeoDataFrame.
        ValueError: If labels or geometries are missing or duplicated, geometries are empty or
            invalid, CRS metadata is absent, or the calculation would use a geographic CRS.
    """
    if not isinstance(source, gpd.GeoDataFrame) or not isinstance(target, gpd.GeoDataFrame):
        raise TypeError("source and target must be GeoDataFrames")
    _require_columns(source, (source_label,))
    _require_columns(target, (target_label,))
    if source.empty or target.empty:
        raise ValueError("source and target cannot be empty")

    source_values = cast(pd.Series, source[source_label])
    target_values = cast(pd.Series, target[target_label])
    source_labels = _labels(source_values, source_label)
    target_labels = _labels(target_values, target_label)
    if len(source_labels) != len(source):
        raise ValueError("source labels must be unique")
    if len(target_labels) != len(target):
        raise ValueError("target labels must be unique")
    if source.crs is None or target.crs is None:
        raise ValueError("source and target must have a CRS")

    left = source.loc[:, [source_label, source.geometry.name]].copy()
    right = target.loc[:, [target_label, target.geometry.name]].copy()
    left = left.rename(columns={source_label: "__source"})
    right = right.rename(columns={target_label: "__target"})
    for name, frame in (("source", left), ("target", right)):
        _require_valid_polygons(frame, name)
    if crs is not None:
        left = left.to_crs(crs)
        right = right.to_crs(crs)
    else:
        right = right.to_crs(left.crs)
    _require_projected_crs(left, "areal overlap")

    # Reprojection can corrupt geometries (e.g. collapse or self-intersect near projection
    # singularities), so validate again with an error that names the reprojection step.
    for name, frame in (("source", left), ("target", right)):
        try:
            _require_valid_polygons(frame, name)
        except ValueError as error:
            raise ValueError(f"after reprojection, {error}") from error

    intersections = gpd.overlay(left, right, how="intersection", keep_geom_type=False)
    matrix = pd.DataFrame(0.0, index=source_labels, columns=target_labels)
    if intersections.empty:
        return matrix

    intersections["__area"] = intersections.geometry.area
    areas = cast(
        pd.Series,
        intersections.groupby(["__source", "__target"], observed=True, sort=False)["__area"].sum(),
    )
    for key, area in areas.items():
        source_value, target_value = cast(tuple[Hashable, Hashable], key)
        matrix.at[source_value, target_value] = area
    return matrix
