from typing import cast

import httpx
import pandas as pd
import us

from gerrytools.logging import TRACE, get_logger

from ._api import (
    PL_BASE_URL,
    REQUEST_TIMEOUT,
    _cast_count_columns_to_numeric,
    _census_get,
    _validate_year,
)
from .census_tables import (
    PL_YEARS,
    PLTableInfo,
    pl_table,
)

logger = get_logger(__name__)


def census(
    state: us.states.State,
    geometry: str = "state",
    year: int = 2020,
    table: PLTableInfo | str = "P1",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Retrieve decennial PL94-171 Census data for one state, geometry, and table.

    Issues a single ``get=group({table})`` request to the decennial PL Census API, drops empty and
    ``*err`` columns, renames the raw Census variables to semantic names carrying the requested
    vintage (e.g. ``total_vap_20``), and casts those count columns to numeric. The returned
    DataFrame is indexed by ``GEOID`` with the Census-stub ``"US"`` prefix stripped.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``,
            ``"block group"``, or ``"block"``. Block queries cover every county and tract in the
            state and can return large responses. Defaults to ``"state"``.
        year (int): Decennial PL vintage to query. ``2010`` and ``2020`` are supported. Defaults to
            ``2020``.
        table (PLTableInfo | str): Either a ``PLTableInfo`` instance or a shortcut string (``"P1"``,
            ``"P2"``, ``"P3"``, ``"P4"``, ``"P5"``, or ``"H1"`` when available for ``year``)
            that is resolved via :func:`pl_table`. Defaults to ``"P1"``.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        pd.DataFrame: One row per geography at ``geometry``, indexed by ``GEOID``, with one numeric
        column per renamed variable from ``table``.

    Raises:
        ValueError: If ``year`` is unsupported, ``table`` is an invalid shortcut string, carries a
            different vintage than ``year``, or has no single API group code, ``geometry`` is
            unrecognized, or the response is missing any of the table's variables (a raw variable
            spelling drifted between the API and the table definition).
        CensusRateLimitError: Propagated from the underlying fetch if the API returns HTTP 429.
        httpx.HTTPStatusError: Propagated from the underlying fetch for other HTTP errors.
    """

    _validate_year(year)
    if isinstance(table, str):
        table = pl_table(table, year)  # validates both the year and the table shortcut
    elif year not in PL_YEARS:
        raise ValueError(f"Decennial PL data is only available for years {PL_YEARS}; got {year}.")
    elif table.year != year:
        raise ValueError(
            f"Table {table.table_name!r} carries PL vintage {table.year} variable names, but "
            f"census() was asked for year {year}; raw variable names differ across vintages, "
            "so a cross-vintage request would return no usable columns."
        )

    api_table_code = table.api_table_code
    if api_table_code is None:
        raise ValueError(
            f"Table {table.table_name!r} has no single Census API group code, so it cannot be "
            "fetched with census(); it combines multiple API groups."
        )

    logger.log(
        TRACE,
        "Decennial PL %s %s for %s (%s).",
        year,
        api_table_code,
        state.abbr,
        geometry,
    )

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        df = _census_get(
            client,
            PL_BASE_URL.format(year=year),
            f"group({api_table_code})",
            state,
            geometry,
            api_key=api_key,
        )

    if df.empty:
        # Header-only response: keep the table's column schema so callers see a stable shape.
        short_names = dict.fromkeys(table.construct_short_names(year=year))
        return pd.DataFrame(columns=pd.Index(short_names), index=pd.Index([], name="GEOID"))

    err_cols = [col for col in df.columns if col.lower().endswith("err")]
    df.drop(columns=err_cols, inplace=True)

    table.rename_columns(df, year=year)

    short_names = list(table.construct_short_names(year=year))
    missing_columns = sorted(set(short_names) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Census PL response for table {table.table_name!r} is missing expected columns "
            f"{missing_columns} after renaming; the API's variable spellings and the table "
            "definition are out of sync."
        )

    # Intentional: variables the API returned but left entirely NA are dropped rather than kept
    # as all-NA columns.
    all_na_columns = df.columns[df.isna().all()].tolist()
    df.drop(columns=all_na_columns, inplace=True)

    count_columns = [column for column in short_names if column in df.columns]
    _cast_count_columns_to_numeric(df, count_columns)

    # Return only the count columns indexed by GEOID; the Census name and geography-breakdown
    # columns (state/county/tract, derivable from GEOID) are dropped for parity with acs().
    df.set_index("GEOID", inplace=True)
    return cast(pd.DataFrame, df[count_columns])
