import warnings
from typing import Literal, cast

import httpx
import pandas as pd
import us

from gerrytools.logging import TRACE, get_logger

from ._api import (
    ACS_BASE_URL,
    REQUEST_TIMEOUT,
    CensusRateLimitError,
    _census_get,
    _validate_year,
)
from .census_tables import (
    ACSCVAPTableInfo,
    ACSTableInfo,
    ACSTotPopTableInfo,
    ACSVAPTableInfo,
    format_acs_column_names,
)

logger = get_logger(__name__)

ACSSurvey = Literal["acs1", "acs5"]
_MISSING_SENTINELS = (-222222222, -333333333, -666666666, -888888888, -999999999)
_CONTROLLED_ESTIMATE_SENTINEL = -555555555


def _normalize_acs_survey(survey: str | int) -> ACSSurvey:
    """Normalize common ACS survey-period inputs to the Census API dataset name.

    Args:
        survey (str | int): ACS survey period to query. Accepts ``"acs5"``, ``"acs1"``, ``5``, or
            ``1`` (case- and separator-insensitive).

    Returns:
        ACSSurvey: The normalized ACS survey period, either ``"acs1"`` or ``"acs5"``.

    Raises:
        ValueError: If ``survey`` is neither a recognized string nor ``1``/``5``.
    """

    if isinstance(survey, int):
        survey = str(survey)
    elif not isinstance(survey, str):
        raise ValueError("Invalid ACS survey. Must be one of 'acs1', 'acs5', 1, or 5.")

    normalized_survey = survey.lower().replace("_", "").replace("-", "").replace(" ", "")

    if normalized_survey in {"1", "1year", "acs1", "acs1year"}:
        return "acs1"

    if normalized_survey in {"5", "5year", "acs5", "acs5year"}:
        return "acs5"

    raise ValueError("Invalid ACS survey. Must be one of 'acs1', 'acs5', 1, or 5.")


def _warn_if_partial_acs1_data(
    data: pd.DataFrame,
    state: us.states.State,
    year: int,
    geometry: str,
    api_key: str | None = None,
) -> None:
    """Warn when ACS 1-year returns only the population-threshold geographies.

    ACS 1-year estimates are only published for geographies of at least 65,000 people, so a
    county-level query may return fewer counties than the state actually has. This helper compares
    the returned set against the complete set from ACS 5-year and emits a ``UserWarning`` when any
    are missing.

    Args:
        data (pd.DataFrame): ACS 1-year DataFrame returned to the caller, indexed by GEOID.
        state (us.states.State): State the query is scoped to.
        year (int): ACS data year (5-year vintage or 1-year year).
        geometry (str): Geometry level. No-op unless this is ``"county"``.
        api_key (str | None): Census API key forwarded to the completeness check. If omitted, falls
            back to the ``CENSUS_API_KEY`` environment variable.
    """

    if geometry != "county":
        return

    # The complete geography set comes from ACS 5-year, which is not population-thresholded.
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            complete = _census_get(
                client,
                ACS_BASE_URL.format(year=year, survey="acs5"),
                "GEO_ID",
                state,
                geometry,
                api_key=api_key,
            )
        expected_geo_ids = set(complete["GEOID"])
    except (httpx.HTTPError, CensusRateLimitError, ValueError, KeyError):
        # Advisory probe: any failure (transport, rate limit, junk payload, missing GEOID
        # column) must not clobber the already-fetched ACS1 result.
        return

    returned_geo_ids = set(data.index)
    missing_geo_ids = expected_geo_ids - returned_geo_ids

    if not missing_geo_ids:
        return

    warnings.warn(
        (
            f"ACS 1-year returned {len(returned_geo_ids)} of "
            f"{len(expected_geo_ids)} {state.name} counties for {year}. "
            "ACS 1-year estimates are only available for geographies with "
            "at least 65,000 people; use survey='acs5' for complete county coverage."
        ),
        UserWarning,
        stacklevel=3,
    )


def _get_acs_data(
    client: httpx.Client,
    state: us.states.State,
    geometry: str,
    year: int,
    table: ACSTableInfo,
    survey: ACSSurvey = "acs5",
    suffix: str = "E",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Retrieve raw ACS data from the Census API for one state/geometry/table.

    Query parameters follow the Census Bureau API user guide
    (https://www.census.gov/content/dam/Census/data/developers/api-user-guide/api-user-guide.pdf).
    Variable and example pages are published per dataset, e.g.
    https://api.census.gov/data/2022/acs/acs5/variables.html.

    Args:
        client (httpx.Client): HTTP client used to issue the request.
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``, or
            ``"block group"``. ACS 1-year does not publish ``"tract"`` or ``"block group"``.
        year (int): ACS data year (5-year vintage end year or 1-year year).
        table (ACSTableInfo): Table definition supplying the Census variable names to request.
        survey (ACSSurvey): Normalized ACS survey period, either ``"acs5"`` or ``"acs1"``. Defaults
            to ``"acs5"``.
        suffix (str): Census variable suffix: ``"E"`` for estimates, ``"M"`` for margins of error.
            Defaults to ``"E"``.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        pd.DataFrame: DataFrame indexed by GEOID (with the ``"US"`` prefix stripped) and containing
        the requested variables cast to ``float``.

    Raises:
        ValueError: If ``survey`` is ``"acs1"`` and ``geometry`` is ``"tract"`` or ``"block
            group"``.
    """

    if survey == "acs1" and geometry in {"tract", "block group"}:
        raise ValueError(
            "ACS 1-year data are not available for 'tract' or 'block group' geometry. "
            "Use ACS 5-year data for those geometry levels."
        )

    cols = list(table.construct_long_names(suffix=suffix, year=year).keys())
    query_cols = ["GEO_ID"] + cols

    # The Census API rejects requests for more than 50 variables.
    if len(query_cols) > 50:
        raise ValueError(
            f"The Census API accepts at most 50 variables per request; table "
            f"{table.table_name!r} needs {len(query_cols)} (including GEO_ID). "
            "Request fewer variables or split the table."
        )

    logger.log(
        TRACE,
        "ACS %s %s %s %s for %s (%d vars).",
        survey,
        year,
        table.table_name or "<unnamed>",
        "EST" if suffix == "E" else "MOE",
        state.abbr,
        len(cols),
    )
    new_df = _census_get(
        client,
        ACS_BASE_URL.format(year=year, survey=survey),
        ",".join(query_cols),
        state,
        geometry,
        api_key=api_key,
    )
    new_df.set_index("GEOID", inplace=True)
    data = cast(pd.DataFrame, new_df[cols].astype(float))
    replacements = {sentinel: float("nan") for sentinel in _MISSING_SENTINELS}
    replacements[_CONTROLLED_ESTIMATE_SENTINEL] = 0.0 if suffix == "M" else float("nan")
    return data.replace(replacements)


def _fetch_est_moe(
    client: httpx.Client,
    state: us.states.State,
    geometry: str,
    year: int,
    table: ACSTableInfo,
    survey: ACSSurvey,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the raw estimate and margin-of-error frames for one table.

    Args:
        client (httpx.Client): Shared HTTP client used for both requests.
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level.
        year (int): ACS data year.
        table (ACSTableInfo): Table definition describing the variables to request.
        survey (ACSSurvey): Normalized ACS survey period.
        api_key (str | None): Census API key, or ``CENSUS_API_KEY`` fallback.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Raw estimate (EST) and margin-of-error (MOE)
        DataFrames with Census variable names, in that order.
    """

    est_data = _get_acs_data(
        client,
        state,
        geometry,
        year,
        table,
        survey=survey,
        suffix="E",
        api_key=api_key,
    )
    moe_data = _get_acs_data(
        client,
        state,
        geometry,
        year,
        table,
        survey=survey,
        suffix="M",
        api_key=api_key,
    )
    return est_data, moe_data


def acs_full(
    state: us.states.State,
    geometry: str,
    year: int,
    table: ACSTableInfo,
    rename_columns: bool = True,
    survey: str | int = "acs5",
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve full (ungrouped) ACS data for one state, geometry, and table.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``, or
            ``"block group"``.
        year (int): ACS data year (5-year vintage end year or 1-year year).
        table (ACSTableInfo): Table definition describing the variables to request.
        rename_columns (bool): Whether to rename raw Census variable names to long English-language
            descriptions. Defaults to ``True``.
        survey (str | int): ACS survey period. Accepts ``"acs5"``, ``"acs1"``, ``5``, or ``1``.
            Defaults to ``"acs5"``.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Estimate (EST) and margin-of-error (MOE) DataFrames, in
        that order. Descriptive names include the queried product and vintage, such as
        ``total_pop_est_acs5_23``. An ACS 1-year county query that returns only the geographies
        meeting the 65,000-population threshold emits a ``UserWarning``.

    Raises:
        ValueError: If ``year`` is not a 4-digit integer in [2000, 2050], or ``survey`` is not
            recognized.
    """

    _validate_year(year)
    survey = _normalize_acs_survey(survey)

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        est_data, moe_data = _fetch_est_moe(
            client, state, geometry, year, table, survey, api_key=api_key
        )

    if survey == "acs1":
        _warn_if_partial_acs1_data(est_data, state, year, geometry, api_key=api_key)

    if rename_columns:
        est_data = est_data.rename(
            columns=table.construct_long_names(
                suffix="E",
                year=year,
                source_suffix=survey,
            )
        )
        moe_data = moe_data.rename(
            columns=table.construct_long_names(
                suffix="M",
                year=year,
                source_suffix=survey,
            )
        )

    return est_data, moe_data


def _condense(
    data: pd.DataFrame,
    table: ACSTableInfo,
    suffix: str,
    label: str,
) -> pd.DataFrame:
    """Collapse raw ACS columns into the grouped sums in ``condense_group_dict``.

    Args:
        data (pd.DataFrame): DataFrame of raw ACS variables (Census variable names with the suffix
            already applied).
        table (ACSTableInfo): Table definition whose ``condense_group_dict`` drives the grouping.
        suffix (str): Census variable suffix (``"E"`` or ``"M"``) used to reconstruct source column
            names from the table's group specifications.
        label (str): Suffix appended to each output group column name (``""`` for estimates,
            ``"_moe"`` for margins of error).

    Returns:
        pd.DataFrame: DataFrame with one column per group. Estimate cells are summed; margins of
        error use root-sum-of-squares.

    Raises:
        ValueError: If any group's source columns are missing from ``data``. The fetchers always
            request every variable a table declares, so a gap means the frame and the table
            definition are out of sync.
    """

    columns = set(data.columns)
    result = pd.DataFrame(index=data.index)
    for group, variables in table.condense_group_dict.items():
        source_cols = [variable + suffix for variable in variables]
        missing = sorted(set(source_cols) - columns)
        if missing:
            raise ValueError(
                f"Cannot condense group {group!r} for table {table.table_name!r}: "
                f"source columns {missing} are missing from the fetched frame."
            )
        values = data[source_cols]
        result[group + label] = (
            values.pow(2).sum(axis=1, skipna=False).pow(0.5)
            if suffix == "M"
            else values.sum(axis=1, skipna=False)
        )
    return result


def _acs(
    client: httpx.Client,
    state: us.states.State,
    geometry: str,
    year: int,
    table: ACSTableInfo,
    survey: ACSSurvey = "acs5",
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve ACS data for one table and consolidate per ``condense_group_dict``.

    ``survey`` must already be normalized to ``"acs1"`` or ``"acs5"`` — this is a private helper
    called only from ``acs()``, which performs the normalization once.

    Args:
        client (httpx.Client): Shared HTTP client used for both table requests.
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level.
        year (int): ACS data year.
        table (ACSTableInfo): Table to query.
        survey (ACSSurvey): Normalized ACS survey period. Defaults to ``"acs5"``.
        api_key (str | None): Census API key, or ``CENSUS_API_KEY`` fallback.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Condensed estimate (EST) and margin-of-error (MOE)
        DataFrames, in that order. Estimate columns omit the redundant ``est`` marker; MOE columns
        retain ``moe``.
    """

    est_data, moe_data = _fetch_est_moe(
        client, state, geometry, year, table, survey, api_key=api_key
    )

    short_est_data = _condense(est_data, table, suffix="E", label="")
    short_moe_data = _condense(moe_data, table, suffix="M", label="_moe")

    format_acs_column_names(short_est_data, source=survey, year=year)
    format_acs_column_names(short_moe_data, source=survey, year=year)

    return short_est_data, short_moe_data


def _acs_estimates(
    state: us.states.State,
    geometry: str,
    year: int,
    tables: list[ACSTableInfo],
    survey: ACSSurvey = "acs5",
    api_key: str | None = None,
    *,
    client: httpx.Client,
) -> pd.DataFrame:
    """Retrieve and condense ACS estimates only, issuing no margin-of-error requests.

    Estimates-only sibling of :func:`acs` for callers that never consume MOE frames (block CVAP
    rate inputs), halving the request count. ``survey`` must already be normalized.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level.
        year (int): ACS data year.
        tables (list[ACSTableInfo]): Tables to query.
        survey (ACSSurvey): Normalized ACS survey period. Defaults to ``"acs5"``.
        api_key (str | None): Census API key, or ``CENSUS_API_KEY`` fallback.
        client (httpx.Client): HTTP client reused across the per-table requests (and, in
            ``block_cvap_estimates``, across the whole pipeline).

    Returns:
        pd.DataFrame: Condensed estimate DataFrame with columns from every requested table joined
        side by side, named as in the estimate frame returned by :func:`acs`.
    """

    est_frames: list[pd.DataFrame] = []
    for table in tables:
        est_data = _get_acs_data(
            client,
            state,
            geometry,
            year,
            table,
            survey=survey,
            suffix="E",
            api_key=api_key,
        )
        condensed = _condense(est_data, table, suffix="E", label="")
        format_acs_column_names(condensed, source=survey, year=year)
        est_frames.append(condensed)
    return pd.concat(est_frames, axis=1)


def acs(
    state: us.states.State,
    geometry: str,
    year: int,
    tables: list[ACSTableInfo] | None = None,
    survey: str | int = "acs5",
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve ACS data for one or more tables and concatenate column-wise.

    Each table is consolidated per its own ``condense_group_dict``; the resulting group columns from
    all tables are concatenated into the output DataFrames.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``, or
            ``"block group"``.
        year (int): ACS data year (5-year vintage end year or 1-year year).
        tables (list[ACSTableInfo] | None): Tables to query. Defaults to ``[ACSTotPopTableInfo(),
            ACSVAPTableInfo(), ACSCVAPTableInfo()]``. ``ACSRacePopTableInfo`` and
            ``ACSAgeTableInfo`` provide opt-in race and sex-by-age population columns.
        survey (str | int): ACS survey period. Accepts ``"acs5"``, ``"acs1"``, ``5``, or ``1``.
            Defaults to ``"acs5"``.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Condensed estimate (EST) and margin-of-error (MOE)
            DataFrames, in that order, with columns from every requested table joined side by side.
            Names include the normalized survey and vintage, such as ``total_pop_acs5_23`` and
            ``total_pop_moe_acs5_23``.

    Raises:
        ValueError: If ``year`` is not a 4-digit integer in [2000, 2050], if ``tables`` is empty, or
            if ``survey`` is not recognized.
        TypeError: If any element of ``tables`` is not an ``ACSTableInfo``.
    """

    _validate_year(year)

    if tables is None:
        tables = [ACSTotPopTableInfo(), ACSVAPTableInfo(), ACSCVAPTableInfo()]

    if not tables:
        raise ValueError("Must provide at least one table.")

    for table in tables:
        if not isinstance(table, ACSTableInfo):
            raise TypeError(f"Each table must be an ACSTableInfo; got {type(table).__name__}.")

    condensed_columns: dict[str, str] = {}
    for table in tables:
        for column in table.condense_group_dict:
            if column in condensed_columns:
                previous = condensed_columns[column]
                raise ValueError(
                    f"ACS tables {previous!r} and {table.table_name!r} both produce condensed "
                    f"column {column!r}."
                )
            condensed_columns[column] = table.table_name

    survey = _normalize_acs_survey(survey)

    est_frames: list[pd.DataFrame] = []
    moe_frames: list[pd.DataFrame] = []
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for table in tables:
            est_data, moe_data = _acs(
                client,
                state,
                geometry,
                year,
                table,
                survey=survey,
                api_key=api_key,
            )
            est_frames.append(est_data)
            moe_frames.append(moe_data)

    est_result = pd.concat(est_frames, axis=1)
    if survey == "acs1":
        _warn_if_partial_acs1_data(est_result, state, year, geometry, api_key=api_key)
    return est_result, pd.concat(moe_frames, axis=1)


def cvap(
    state: us.states.State,
    geometry: str,
    year: int,
    survey: str | int = "acs5",
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve Citizen Voting-Age Population (CVAP) data for one state and geometry.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``, or
            ``"block group"``.
        year (int): ACS data year (5-year vintage end year or 1-year year).
        survey (str | int): ACS survey period. Accepts ``"acs5"``, ``"acs1"``, ``5``, or ``1``.
            Defaults to ``"acs5"``.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Condensed CVAP estimate (EST) and margin-of-error (MOE)
        DataFrames, in that order, with names such as ``total_cvap_acs5_23``.
    """

    return acs(
        state,
        geometry,
        year,
        tables=[ACSCVAPTableInfo()],
        survey=survey,
        api_key=api_key,
    )
