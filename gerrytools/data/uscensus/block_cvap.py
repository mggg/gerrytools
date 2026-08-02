import math
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
from .acs import _acs_estimates
from .census_tables import (
    RACE_CATEGORIES,
    ACSCVAPTableInfo,
    ACSTableInfo,
    ACSVAPTableInfo,
    PLBlockVAPTableInfo,
    PLTableInfo,
    census_column_name,
)

logger = get_logger(__name__)

DEFAULT_BLOCK_CVAP_ACS_YEAR = 2024
DEFAULT_BLOCK_CVAP_PL_YEAR = 2020
DEFAULT_BLOCK_CVAP_DENOMINATOR_THRESHOLD = 20

# PL vintages whose block VAP variables PLBlockVAPTableInfo carries; only 2020 P3/P4 today.
SUPPORTED_BLOCK_CVAP_PL_YEARS: tuple[int, ...] = (2020,)


def _fetch_decennial_pl_county_fips(
    client: httpx.Client,
    state: us.states.State,
    pl_year: int,
    api_key: str | None = None,
) -> list[str]:
    """Fetch the county FIPS codes for a state from the decennial PL API.

    Statewide block responses can approach the request timeout for large states, so callers
    enumerate counties and issue smaller block requests. This helper performs that enumeration by
    asking the PL API for ``NAME`` at county geography.

    Args:
        client (httpx.Client): HTTP client used to issue the request. ``block_cvap_estimates``
            shares one client across its whole pipeline (ACS rates plus every PL call) so httpx
            reuses the underlying TCP/TLS connection, which matters because per-block-query
            latency dominates when there are many counties.
        state (us.states.State): State the query is scoped to.
        pl_year (int): Decennial PL vintage to query (e.g. ``2020``).
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        list[str]: County FIPS codes (3-digit strings, state-local), sorted lexicographically so the
        subsequent per-county fetch proceeds in a deterministic order.

    Raises:
        CensusRateLimitError: Propagated from ``_census_get`` if the API returns HTTP 429.
        httpx.HTTPStatusError: Propagated from ``_census_get`` for other HTTP errors.
    """

    counties = _census_get(
        client,
        PL_BASE_URL.format(year=pl_year),
        "NAME",
        state,
        "county",
        api_key=api_key,
    )
    return sorted(counties["county"].tolist())


def _get_decennial_pl_data(
    client: httpx.Client,
    state: us.states.State,
    year: int,
    geometry: str,
    table: PLTableInfo,
    county_fips: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Retrieve decennial PL data for one table and geometry level.

    Unlike ACS variables, PL variables do not carry estimate/MOE suffixes; this function requests
    the raw variable names declared by ``table`` and renames them to the table's short local names.
    The Census-returned ``GEO_ID`` column (which is prefixed with a summary-level stub like
    ``"1000000US"``) is split on ``"US"`` and the trailing GEOID substring is stored on the returned
    DataFrame as ``GEOID``.

    Args:
        client (httpx.Client): HTTP client used to issue the request. Reused across per-county block
            queries to amortize TCP/TLS setup.
        state (us.states.State): State the query is scoped to.
        year (int): Decennial PL vintage to query.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``,
            ``"block group"``, or ``"block"``.
        table (PLTableInfo): Table definition supplying the Census variable names to request and the
            short names to rename them to.
        county_fips (str | None): Optional 3-digit county scope for block queries; ignored for
            coarser geometries.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        pd.DataFrame: DataFrame containing one row per geography at ``geometry`` within the
        requested scope, with the table's short column names and a ``GEOID`` column replacing the
        Census-native ``GEO_ID``.

    Raises:
        CensusRateLimitError: Propagated from ``_census_get`` if the API returns HTTP 429.
        httpx.HTTPStatusError: Propagated from ``_census_get`` for other HTTP errors.
    """

    query_columns = ["GEO_ID"] + list(table.construct_variable_names())
    data = _census_get(
        client,
        PL_BASE_URL.format(year=year),
        ",".join(query_columns),
        state,
        geometry,
        county_fips=county_fips,
        api_key=api_key,
    )
    table.rename_columns(data, year=year)

    return data


def _fetch_block_pl_vap_for_county(
    client: httpx.Client,
    state: us.states.State,
    county_fips: str,
    pl_year: int,
    table: PLBlockVAPTableInfo,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch block-level VAP by race from decennial PL data for one county.

    The raw PL columns are coerced to numeric (the Census API returns all values as JSON strings)
    and the 15-character block GEOID is split into ``STATEFP`` (2), ``COUNTYFP`` (3), ``TRACTCE``
    (6), ``BLOCKCE`` (4), and ``TRACT_GEOID`` (first 11 chars) so that downstream code can join the
    block VAP frame against ACS tract-level CVAP/VAP rates by ``TRACT_GEOID``.

    Args:
        client (httpx.Client): HTTP client used to issue the request. The same client is reused for
            every per-county call when invoked from ``block_cvap_estimates``.
        state (us.states.State): State the query is scoped to.
        county_fips (str): 3-digit county FIPS code identifying the county to fetch blocks within.
            The decennial PL API requires block queries to be scoped to a single county.
        pl_year (int): Decennial PL vintage to query.
        table (PLBlockVAPTableInfo): Table definition describing the block-level VAP variables to
            request (P3 race-by-VAP and P4 Hispanic-by-VAP variables, named per
            ``table.variable_to_short_name``).
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        pd.DataFrame: Block-level DataFrame with one row per block in the county. Columns include
        ``GEOID``, the parsed GEOID components (``STATEFP``, ``COUNTYFP``, ``TRACTCE``, ``BLOCKCE``,
        ``TRACT_GEOID``), and the race-specific VAP columns named per
        ``table.construct_short_names(year=pl_year)`` (e.g. ``total_vap_20`` and
        ``hispanic_vap_20``).

    Raises:
        CensusRateLimitError: Propagated from ``_census_get`` if the API returns HTTP 429.
        httpx.HTTPStatusError: Propagated from ``_census_get`` for other HTTP errors.
    """

    blocks = _get_decennial_pl_data(
        client,
        state,
        pl_year,
        "block",
        table,
        county_fips=county_fips,
        api_key=api_key,
    )

    _cast_count_columns_to_numeric(blocks, list(table.construct_short_names(year=pl_year)))

    blocks["STATEFP"] = blocks["GEOID"].str[:2]
    blocks["COUNTYFP"] = blocks["GEOID"].str[2:5]
    blocks["TRACTCE"] = blocks["GEOID"].str[5:11]
    blocks["BLOCKCE"] = blocks["GEOID"].str[11:15]
    blocks["TRACT_GEOID"] = blocks["GEOID"].str[:11]

    return blocks


def _state_cvap_rates(
    state_est: pd.DataFrame,
    acs_year: int,
    race_categories: tuple[str, ...] = RACE_CATEGORIES,
) -> dict[str, float]:
    """Compute statewide race-specific CVAP/VAP fallback rates.

    Args:
        state_est (pd.DataFrame): Single-row state-level ACS estimate DataFrame.
        acs_year (int): ACS 5-year vintage used in the DataFrame's column names.
        race_categories (tuple[str, ...]): Race categories to compute rates for.

    Returns:
        dict[str, float]: Mapping from race category to statewide ``CVAP / VAP`` rate. Rates for
        categories with zero statewide VAP are reported as ``0.0``.

    Raises:
        ValueError: If the DataFrame does not contain exactly one row, an estimate is non-finite
            or negative, or a CVAP estimate exceeds its containing VAP estimate.
    """

    if len(state_est) != 1:
        raise ValueError(
            f"State-level ACS estimates must contain exactly one row; found {len(state_est)}."
        )

    state_row = state_est.iloc[0]
    rates = {}
    for race in race_categories:
        vap = state_row[census_column_name(f"{race}_vap", source="acs5", year=acs_year)]
        cvap = state_row[census_column_name(f"{race}_cvap", source="acs5", year=acs_year)]
        if not (math.isfinite(vap) and math.isfinite(cvap) and vap >= 0 and cvap >= 0):
            raise ValueError(
                f"ACS {acs_year} {race} VAP and CVAP estimates must be finite and nonnegative; "
                f"found VAP {vap} and CVAP {cvap}."
            )
        if cvap > vap:
            raise ValueError(
                f"ACS {acs_year} {race} CVAP estimate {cvap} exceeds VAP estimate {vap}."
            )
        rates[race] = float(cvap / vap) if vap > 0 else 0.0
    return rates


def _tract_cvap_rates(
    tract_est: pd.DataFrame,
    denominator_threshold: int,
    acs_year: int,
    race_categories: tuple[str, ...] = RACE_CATEGORIES,
) -> dict[str, pd.Series]:
    """Compute tract-level CVAP/VAP rates, masking low ACS VAP denominators.

    When an ACS tract's VAP for a race is below ``denominator_threshold``, the resulting rate is
    ``NaN``, signalling to the caller that a fallback rate should be used for that tract.

    Args:
        tract_est (pd.DataFrame): Tract-level ACS estimate DataFrame indexed by tract GEOID.
        denominator_threshold (int): Minimum tract VAP required for a tract rate to be computed.
        acs_year (int): ACS 5-year vintage used in the DataFrame's column names.
        race_categories (tuple[str, ...]): Race categories to compute rates for.

    Returns:
        dict[str, pd.Series]: Mapping from race category to a Series of tract rates, with ``NaN``
        for tracts below the denominator threshold.

    Raises:
        ValueError: If a tract VAP or CVAP estimate is non-finite or negative, or a tract CVAP
            estimate exceeds its containing VAP estimate.
    """

    rates = {}
    for race in race_categories:
        vap = tract_est[census_column_name(f"{race}_vap", source="acs5", year=acs_year)]
        cvap = tract_est[census_column_name(f"{race}_cvap", source="acs5", year=acs_year)]
        invalid_value = ~vap.map(math.isfinite) | ~cvap.map(math.isfinite) | (vap < 0) | (cvap < 0)
        if invalid_value.any():
            geoid = invalid_value[invalid_value].index[0]
            raise ValueError(
                f"ACS {acs_year} {race} tract {geoid} VAP and CVAP estimates must be finite "
                f"and nonnegative; found VAP {vap.loc[geoid]} and CVAP {cvap.loc[geoid]}."
            )
        invalid = cvap > vap
        if invalid.any():
            geoid = invalid[invalid].index[0]
            raise ValueError(
                f"ACS {acs_year} {race} tract {geoid} has CVAP estimate {cvap.loc[geoid]} "
                f"exceeding VAP estimate {vap.loc[geoid]}."
            )
        denominator = vap.where(vap >= denominator_threshold)
        rates[race] = cvap.divide(denominator)
    return rates


def _estimate_block_cvap_from_inputs(
    blocks: pd.DataFrame,
    tract_est: pd.DataFrame,
    state_est: pd.DataFrame,
    denominator_threshold: int,
    table: PLBlockVAPTableInfo,
    acs_year: int,
    pl_year: int,
) -> pd.DataFrame:
    """Estimate block CVAP by applying ACS citizenship rates to PL block VAP.

    For each race category ``r`` in ``table.race_categories``:

    - ``tract_rate[r]`` is ACS 5-year tract CVAP divided by VAP
      when the tract's ACS VAP is at least ``denominator_threshold``;
      otherwise NaN.
    - ``state_rate[r]`` is ACS 5-year state CVAP divided by VAP.
    - Each block's estimated CVAP is block VAP times the selected tract or state rate.

    Args:
        blocks (pd.DataFrame): Block-level DataFrame (from ``_fetch_block_pl_vap_for_county``) with
            ``TRACT_GEOID``, parsed GEOID components, and vintage-suffixed VAP columns.
        tract_est (pd.DataFrame): Tract-level ACS estimate DataFrame used for ``tract_rate``.
        state_est (pd.DataFrame): Single-row state-level ACS estimate DataFrame used for
            ``state_rate``.
        denominator_threshold (int): Minimum ACS tract VAP required to use a tract-level rate.
        table (PLBlockVAPTableInfo): Table definition providing the race categories and PL block
            VAP column names used as the CVAP allocation base.
        acs_year (int): ACS 5-year vintage used to calculate citizenship rates.
        pl_year (int): Decennial PL vintage supplying block VAP.

    Returns:
        pd.DataFrame: Block-level DataFrame with GEOID components, block VAP, and one estimated
        CVAP column per race carrying both input vintages.
    """

    race_categories = table.race_categories
    block_vap_columns = table.construct_short_names(year=pl_year)
    block_vap_column_by_race = {
        race: census_column_name(f"{race}_vap", year=pl_year) for race in race_categories
    }

    tract_rates = _tract_cvap_rates(
        tract_est,
        denominator_threshold,
        acs_year,
        race_categories,
    )
    state_rates = _state_cvap_rates(
        state_est,
        acs_year,
        race_categories,
    )
    estimated = blocks.copy()

    total_tract_blocks = 0
    total_fallback_blocks = 0
    # Estimated columns carry both input vintages: the ACS rate suffix, then the PL VAP suffix.
    cvap_column_by_race = {
        race: census_column_name(
            census_column_name(f"{race}_cvap", source="acs5", year=acs_year),
            source="pl",
            year=pl_year,
        )
        for race in race_categories
    }
    for race in race_categories:
        tract_rate = cast(pd.Series, estimated["TRACT_GEOID"]).map(tract_rates[race])
        fallback_mask = tract_rate.isna()
        selected_rate = tract_rate.fillna(state_rates[race])
        block_vap_column = block_vap_column_by_race[race]
        estimated[cvap_column_by_race[race]] = estimated[block_vap_column] * selected_rate

        fallback_blocks = int(fallback_mask.sum())
        tract_rate_blocks = int((~fallback_mask).sum())
        total_tract_blocks += tract_rate_blocks
        total_fallback_blocks += fallback_blocks

        logger.debug(
            "%s_cvap: tract rate for %s blocks, state fallback for %s (rate=%.6f).",
            race,
            tract_rate_blocks,
            fallback_blocks,
            state_rates[race],
        )

    total = total_tract_blocks + total_fallback_blocks
    if total:
        fallback_pct = 100.0 * total_fallback_blocks / total
        logger.info(
            "Estimated %s block × race CVAP cells (%.1f%% used state fallback rate).",
            total,
            fallback_pct,
        )

    output_columns = (
        ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLOCKCE", "TRACT_GEOID"]
        + list(block_vap_columns)
        + list(cvap_column_by_race.values())
    )
    return cast(pd.DataFrame, estimated[output_columns])


def block_cvap_estimates(
    state: us.states.State,
    acs_year: int = DEFAULT_BLOCK_CVAP_ACS_YEAR,
    pl_year: int = DEFAULT_BLOCK_CVAP_PL_YEAR,
    denominator_threshold: int = DEFAULT_BLOCK_CVAP_DENOMINATOR_THRESHOLD,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Estimate block-level Citizen Voting-Age Population (CVAP) by race.

    Decennial PL data publishes voting-age population (VAP) at the block level but does not publish
    CVAP at any sub-tract geography. This function estimates block CVAP by applying ACS 5-year
    citizenship rates to the decennial PL block VAP counts.

    For each race category ``r`` and each decennial PL block ``b`` in tract ``t`` of state ``s``:

    1. Compute the **tract rate**::

           tract_rate[t, r] = ACS5 tract[t] CVAP / ACS5 tract[t] VAP

       but only when the ACS tract VAP is at least ``denominator_threshold``.
       When the tract's ACS VAP is below the threshold (or the tract is
       missing from the ACS table entirely, which can happen for very small
       tracts), the tract rate is undefined.

    2. Compute the **state fallback rate**::

           state_rate[s, r] = ACS5 state[s] CVAP / ACS5 state[s] VAP

       This is always defined; if the ACS state VAP is zero, the rate is taken to be ``0``.

    3. Select the per-block rate::

           if tract_rate[t, r] is defined:
               selected_rate[b, r] = tract_rate[t, r]
           else:
               selected_rate[b, r] = state_rate[s, r]

    4. Estimate the block CVAP::

           block[b] CVAP = PL block[b] VAP * selected_rate[b, r]

    Block VAP counts come from the decennial PL94-171 Census API (vintage ``pl_year``), queried one
    county at a time to keep large-state responses comfortably below the request timeout. Tract and
    state VAP/CVAP come from ACS 5-year (vintage ``acs_year``); picking an ACS vintage centered on
    the PL year (e.g. ACS 2024 5-year centered on 2020 PL) keeps the citizenship rates temporally
    aligned with the block VAP counts.

    Args:
        state (us.states.State): State to estimate block CVAP for.
        acs_year (int): ACS 5-year vintage end year supplying tract and state VAP/CVAP. Defaults to
            ``DEFAULT_BLOCK_CVAP_ACS_YEAR``.
        pl_year (int): Decennial PL vintage supplying block VAP. Must be one of
            ``SUPPORTED_BLOCK_CVAP_PL_YEARS`` (currently only ``2020``, whose P3/P4 variable
            names the block VAP table carries). Defaults to ``DEFAULT_BLOCK_CVAP_PL_YEAR``.
        denominator_threshold (int): Minimum ACS tract VAP required to use the tract rate rather
            than the state fallback rate. Defaults to ``DEFAULT_BLOCK_CVAP_DENOMINATOR_THRESHOLD``.
            Raising this trades bias (more blocks fall back to the state rate) against variance
            (tract rates from tiny ACS denominators are noisy).
        api_key (str | None): Census API key used for both the ACS and decennial PL requests. If
            omitted, falls back to the ``CENSUS_API_KEY`` environment variable. As of 12 May 2026,
            an API key is required for all requests made to the Census API.

    Returns:
        pd.DataFrame: One row per decennial PL block in the state, with GEOID components (``GEOID``,
        ``STATEFP``, ``COUNTYFP``, ``TRACTCE``, ``BLOCKCE``, ``TRACT_GEOID``), decennial PL VAP
        counts such as ``white_vap_20``, and estimated values such as
        ``white_cvap_acs5_24_pl_20``. Returns an empty DataFrame if the state has no counties in the
        PL table.

    Raises:
        ValueError: If ``acs_year`` or ``pl_year`` is not a 4-digit integer in [2000, 2050],
            ``denominator_threshold`` is not a positive integer, ``pl_year`` is not a supported
            block-VAP vintage, or ``acs_year`` predates 2020 while ``pl_year`` is 2020 (the ACS
            tract vintages would not match the 2020 block GEOIDs).
        CensusRateLimitError: If the Census API returns HTTP 429. This typically means the caller is
            unauthenticated and has exceeded the 500-request daily IP cap; supplying an ``api_key``
            (or setting ``CENSUS_API_KEY``) lifts it.
    """

    _validate_year(acs_year)
    _validate_year(pl_year)
    if (
        isinstance(denominator_threshold, bool)
        or not isinstance(denominator_threshold, int)
        or denominator_threshold < 1
    ):
        raise ValueError(
            "denominator_threshold must be a positive integer, "
            f"but found {denominator_threshold!r}."
        )
    if pl_year not in SUPPORTED_BLOCK_CVAP_PL_YEARS:
        raise ValueError(
            f"block_cvap_estimates supports decennial PL vintages "
            f"{SUPPORTED_BLOCK_CVAP_PL_YEARS}; got {pl_year}. The block VAP table carries "
            "2020 P3/P4 variable names only."
        )
    if acs_year < 2020 and pl_year == 2020:
        raise ValueError(
            f"acs_year {acs_year} predates 2020, so its ACS 5-year tracts use 2010-vintage "
            "boundaries whose GEOIDs do not match the tract GEOIDs derived from 2020 PL "
            "blocks; nearly every block would silently fall back to the statewide rate. "
            "Use an ACS vintage of 2020 or later with pl_year=2020."
        )

    table = PLBlockVAPTableInfo()

    logger.info(
        "Computing block CVAP for %s (ACS5 %s, decennial PL %s).",
        state.name,
        acs_year,
        pl_year,
    )

    county_frames: list[pd.DataFrame] = []
    # One client for the whole pipeline (ACS rates plus every per-county PL query), so httpx
    # reuses the underlying TCP/TLS connection across all ~county-count requests.
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        logger.info("Fetching ACS5 %s tract and state VAP/CVAP for %s.", acs_year, state.name)
        rate_tables: list[ACSTableInfo] = [ACSVAPTableInfo(), ACSCVAPTableInfo()]
        # Estimates-only fetches: the rate computation never consumes margins of error.
        tract_est = _acs_estimates(
            state, "tract", acs_year, rate_tables, api_key=api_key, client=client
        )
        state_est = _acs_estimates(
            state, "state", acs_year, rate_tables, api_key=api_key, client=client
        )
        logger.log(
            TRACE,
            "ACS5 tract VAP/CVAP: %s tracts for %s.",
            len(tract_est),
            state.abbr,
        )

        county_fips_values = _fetch_decennial_pl_county_fips(
            client, state, pl_year, api_key=api_key
        )
        total_counties = len(county_fips_values)
        logger.info(
            "Fetching %s decennial PL block VAP across %s counties in %s.",
            pl_year,
            total_counties,
            state.name,
        )
        # Counties fetch sequentially on purpose: a deliberate rate-limit posture for the
        # per-key Census API caps.
        for index, county_fips in enumerate(county_fips_values, start=1):
            blocks = _fetch_block_pl_vap_for_county(
                client,
                state,
                county_fips,
                pl_year,
                table,
                api_key=api_key,
            )
            logger.log(
                TRACE,
                "[%s %s/%s] County %s: %s blocks.",
                state.abbr,
                index,
                total_counties,
                county_fips,
                len(blocks),
            )
            county_frames.append(blocks)

    if not county_frames:
        return pd.DataFrame()

    blocks = pd.concat(county_frames, ignore_index=True)
    logger.info(
        "Fetched %s PL block rows across %s counties for %s.",
        len(blocks),
        len(county_frames),
        state.name,
    )

    return _estimate_block_cvap_from_inputs(
        blocks,
        tract_est,
        state_est,
        denominator_threshold,
        table,
        acs_year,
        pl_year,
    )
