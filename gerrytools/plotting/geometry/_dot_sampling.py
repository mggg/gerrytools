"""Pure random-point sampling used by dot-density plots.

These functions know nothing about matplotlib or plot state: they take polygons and
densities and return dot coordinates.
"""

import os
from numbers import Integral
from typing import cast

import numpy as np
import shapely
from geopandas import GeoDataFrame
from joblib import Parallel, delayed
from numpy.random import Generator
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from gerrytools.plotting._rng import spawn_child_seeds

MAX_CORES = max(int(os.cpu_count() or 1) - 2, 1)
_MAX_REJECTED_SAMPLES = 1_000_000


def _random_xy_in_poly(
    poly: BaseGeometry, n_points: int, *, rng: Generator
) -> tuple[NDArray, NDArray]:
    """Generate random x, y coordinates within a polygon.

    Args:
        poly (BaseGeometry): The polygon within which to generate points.
        n_points (int): The number of random points to generate.
        rng (Generator): NumPy random generator used for coordinate sampling.
    """
    minx, miny, maxx, maxy = poly.bounds
    xs_out = []
    ys_out = []
    n_points_so_far = 0

    # Each point needs to be checked for inclusion and since we generate the points
    # randomly withing the bounding box, the probabilty of inclusion is area(poly) / area(bbox)
    # we are expected to need roughly n_points / probability_of_inclusion points to get n_points
    # inside of the provided polygon

    bounding_box_area = (maxx - minx) * (maxy - miny)
    # An axis-aligned degenerate geometry has a zero-area bounding box; guard before dividing.
    if poly.area <= 0 or bounding_box_area <= 0:
        raise ValueError("Polygon has zero area, cannot generate points within it.")
    probability_of_inclusion = poly.area / bounding_box_area
    max_samples = n_points + _MAX_REJECTED_SAMPLES
    if (
        not np.isfinite(probability_of_inclusion)
        or probability_of_inclusion <= 0
        or n_points / probability_of_inclusion > max_samples
    ):
        raise ValueError(
            "Polygon occupies too little of its bounding box for practical rejection sampling."
        )

    samples_drawn = 0
    while n_points_so_far < n_points:
        if samples_drawn >= max_samples:
            raise ValueError(
                "Could not generate enough points within the rejection sampling attempt limit."
            )
        remaining = n_points - n_points_so_far
        k = min(
            10_000,
            max(1, int(np.ceil(remaining / probability_of_inclusion))),
            max_samples - samples_drawn,
        )
        xs = rng.uniform(minx, maxx, size=k)
        ys = rng.uniform(miny, maxy, size=k)
        samples_drawn += k

        cand = shapely.points(xs, ys)
        mask = shapely.contains(poly, cand)

        xs_out.append(xs[mask])
        ys_out.append(ys[mask])
        n_points_so_far += int(mask.sum())

    x = np.concatenate(xs_out)[:n_points]
    y = np.concatenate(ys_out)[:n_points]
    return x, y


def _make_random_points(
    gdf: GeoDataFrame,
    people_per_dot: int | float,
    datacolumn_name: str,
    rng: Generator,
    n_jobs: int = -1,
    n_chunks: int | np.integer = 10,
) -> tuple[NDArray, NDArray, NDArray]:
    """Generates random points within polygons in a GeoDataFrame.

    Each polygon's dot count uses floor-plus-stochastic-remainder rounding: a value ``v``
    yields ``floor(v / people_per_dot)`` dots plus one more with probability equal to the
    fractional part. Expected dot counts are therefore conserved (many polygons each below
    ``people_per_dot`` still contribute dots in aggregate), and exact multiples of
    ``people_per_dot`` produce exact counts.

    One child seed is spawned per polygon (row) in a single draw from ``rng``, so the
    output depends only on the generator state and the row order, never on ``n_chunks``
    or ``n_jobs``.

    Args:
        gdf (GeoDataFrame): A GeoDataFrame containing polygons.
        people_per_dot (int | float): Number of people represented by each dot.
        datacolumn_name (str): The name of the data column to use for dot density.
        rng (Generator): NumPy random generator used to derive per-polygon generators.
        n_jobs (int): Number of CPU cores to use for parallel processing. Defaults to -1 (all
            available cores minus two).
        n_chunks (int): Number of chunks to split the GeoDataFrame into for parallel processing.
    """
    if gdf.empty:
        raise ValueError("The GeoDataFrame is empty; there are no polygons to place dots in.")

    # Real census extracts routinely carry degenerate rows (missing/empty/zero-area geometry)
    # whose population also rounds to zero dots; those pass. A degenerate row that could
    # produce dots errors up front and deterministically, rather than leaving fractional-count
    # rows to fail only when the stochastic rounding happens to demand a dot.
    for geom, value in zip(gdf.geometry, gdf[datacolumn_name]):
        geom_is_degenerate = geom is None or geom.is_empty or geom.area <= 0
        if geom_is_degenerate and float(value) / people_per_dot > 0:
            raise ValueError("Polygon has zero area, cannot generate points within it.")

    if not isinstance(n_chunks, Integral) or isinstance(n_chunks, (bool, np.bool_)) or n_chunks < 1:
        raise ValueError(f"n_chunks must be a positive integer, but found {n_chunks!r}.")
    n_chunks = int(n_chunks)
    if not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError(
            f"n_jobs must be a positive integer or -1 (all available cores minus two), "
            f"but found {n_jobs!r}."
        )

    use_cores: int = min(MAX_CORES, n_jobs) if n_jobs > 0 else MAX_CORES

    row_seeds = spawn_child_seeds(rng, len(gdf))

    chunk_size = max(1, (len(gdf) + n_chunks - 1) // n_chunks)  # ceil
    chunk_bounds = [(i, min(len(gdf), i + chunk_size)) for i in range(0, len(gdf), chunk_size)]

    def process_chunk(
        chunk: GeoDataFrame, chunk_row_seeds: list[int], start: int
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Generate random dot coordinates for one GeoDataFrame chunk.

        Args:
            chunk (GeoDataFrame): Subset of polygons with density values.
            chunk_row_seeds (list[int]): One RNG seed per row of ``chunk``.
            start (int): Positional row offset of ``chunk`` in the full frame.

        Returns:
            tuple[NDArray, NDArray, NDArray]: X coordinates, Y coordinates, and polygon ids
                for generated dots.
        """
        x_parts = []
        y_parts = []
        pid_parts = []

        for offset, (geom, val, row_seed) in enumerate(
            zip(chunk.geometry.values, chunk[datacolumn_name].values, chunk_row_seeds, strict=True)
        ):
            row_rng = np.random.default_rng(row_seed)
            raw_count = float(val) / people_per_dot
            n_dots = int(np.floor(raw_count))
            fractional_part = raw_count - n_dots
            if fractional_part > 0.0 and row_rng.random() < fractional_part:
                n_dots += 1
            if n_dots <= 0:
                continue

            x, y = _random_xy_in_poly(geom, n_dots, rng=row_rng)
            x_parts.append(x)
            y_parts.append(y)
            pid_parts.append(np.full(n_dots, start + offset, dtype=np.int64))

        if not x_parts:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )

        return (
            np.concatenate(x_parts),
            np.concatenate(y_parts),
            np.concatenate(pid_parts),
        )

    # joblib is untyped; checkers infer a None-bearing element type for Parallel results,
    # but each element is process_chunk's return value.
    results = cast(
        "list[tuple[NDArray, NDArray, NDArray]]",
        Parallel(n_jobs=use_cores)(
            delayed(process_chunk)(gdf.iloc[start:stop], row_seeds[start:stop], start)
            for start, stop in chunk_bounds
        ),
    )

    xs = np.concatenate([r[0] for r in results])
    ys = np.concatenate([r[1] for r in results])
    pids = np.concatenate([r[2] for r in results])
    return xs, ys, pids
