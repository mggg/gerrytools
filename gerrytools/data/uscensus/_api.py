"""Shared HTTP/query plumbing for the Census API (ACS and decennial PL).

This module holds the pieces common to both the ACS fetchers (``acs.py``) and the decennial PL
fetchers (``census.py``, ``block_cvap.py``): the request timeout, base-URL templates, the rate-limit
error, and the single composed fetch, :func:`_census_get`.

Key policy: as of 12 May 2026 the Census API requires an API key for every request. Keys are
resolved from the ``api_key`` argument or the ``CENSUS_API_KEY`` environment variable, and a
missing key raises before any request is issued. Register at
https://api.census.gov/data/key_signup.html.
"""

import logging
import os
import re
import time
from email.utils import parsedate_to_datetime
from typing import cast

import httpx
import pandas as pd
import us

from gerrytools.logging import get_logger

logger = get_logger(__name__)

_KEY_IN_URL = re.compile(r"([?&]key=)[^&\s'\"]+", re.IGNORECASE)

# Shared request timeout for every Census API call.
REQUEST_TIMEOUT = httpx.Timeout(120)

# Transient-failure retry policy for _census_get: up to MAX_REQUEST_ATTEMPTS total attempts with
# exponential backoff. A 429's Retry-After header is honored up to the safety limit below.
# Module-level sleep hook so tests can patch the waits away.
MAX_REQUEST_ATTEMPTS = 4
RETRY_BASE_DELAY_SECONDS = 1.0
MAX_RETRY_AFTER_SECONDS = 60.0
_sleep = time.sleep

# Base-URL templates. The decennial PL endpoint takes only a year; the ACS endpoint additionally
# takes the survey ("acs1" or "acs5").
PL_BASE_URL = "https://api.census.gov/data/{year}/dec/pl"
ACS_BASE_URL = "https://api.census.gov/data/{year}/acs/{survey}"


class CensusRateLimitError(RuntimeError):
    """Raised when the Census API keeps returning HTTP 429 Too Many Requests.

    Every request carries an API key (see the module docstring for the key policy), so a 429
    means the key itself is being rate-limited. ``_census_get`` retries bounded 429 delays and
    raises immediately rather than retrying before an excessive ``Retry-After`` delay.
    """


def _redacted_url(url: httpx.URL | str) -> str:
    """Return ``url`` as a string with any ``key`` query-parameter value redacted.

    The Census API key travels as a query parameter, so raw request URLs must never reach log
    lines or exception messages.
    """

    url = httpx.URL(url)
    if "key" in url.params:
        url = url.copy_set_param("key", "REDACTED")
    return str(url)


def _redacted_request(request: httpx.Request) -> httpx.Request:
    """Copy request metadata without retaining a Census credential."""
    return httpx.Request(
        request.method,
        _redacted_url(request.url),
        headers=request.headers,
        extensions=request.extensions,
    )


def _redacted_response(response: httpx.Response, key: str) -> httpx.Response:
    """Copy an error response without retaining an echoed Census credential."""
    redacted_headers = [
        (name, _KEY_IN_URL.sub(r"\1REDACTED", value.replace(key, "REDACTED")))
        for name, value in response.headers.multi_items()
        if name.casefold() not in {"content-encoding", "content-length"}
    ]
    return httpx.Response(
        response.status_code,
        headers=redacted_headers,
        content=response.content.replace(key.encode(), b"REDACTED"),
        request=_redacted_request(response.request),
        history=[_redacted_response(item, key) for item in response.history],
    )


def _redacted_log_arg(value: object) -> object:
    if isinstance(value, httpx.URL):
        return _redacted_url(value)
    if isinstance(value, str) and "key=" in value.casefold():
        return _KEY_IN_URL.sub(r"\1REDACTED", value)
    return value


def _is_census_log_arg(value: object) -> bool:
    if not isinstance(value, (httpx.URL, str)):
        return False
    try:
        return httpx.URL(value).host == "api.census.gov"
    except httpx.InvalidURL:
        return False


class _CensusKeyFilter(logging.Filter):
    """Redact Census keys before httpx request records reach any handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and any(
            _is_census_log_arg(value) for value in record.args
        ):
            record.args = tuple(_redacted_log_arg(value) for value in record.args)
        elif isinstance(record.args, dict) and any(
            _is_census_log_arg(value) for value in record.args.values()
        ):
            record.args = {key: _redacted_log_arg(value) for key, value in record.args.items()}
        return True


_httpx_logger = logging.getLogger("httpx")
if not any(isinstance(log_filter, _CensusKeyFilter) for log_filter in _httpx_logger.filters):
    _httpx_logger.addFilter(_CensusKeyFilter())


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    """Return a bounded Retry-After delay for a 429, or exponential backoff."""

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                delay = int(retry_after)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        raise ValueError
                    delay = max(retry_at.timestamp() - time.time(), 0.0)
                except (TypeError, ValueError, OverflowError, OSError):
                    delay = -1
            if delay >= 0:
                if delay > MAX_RETRY_AFTER_SECONDS:
                    raise CensusRateLimitError(
                        f"Retry-After {retry_after!r} exceeds the "
                        f"{MAX_RETRY_AFTER_SECONDS:g}-second safety limit; "
                        "refusing to retry early."
                    )
                return float(delay)
    return RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1)


def _get_with_retries(client: httpx.Client, base_url: str, params: dict) -> httpx.Response:
    """Issue one GET, retrying transport failures, bounded 429 delays, and 5xx responses.

    Other statuses (including non-429 4xx) are returned to the caller on the first attempt; the
    final transient response is returned once the attempt budget is exhausted. A final transport
    failure is re-raised.
    """

    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = client.get(base_url, params=params)
        except httpx.TransportError as error:
            if attempt == MAX_REQUEST_ATTEMPTS:
                if error.request is not None:
                    error.request = _redacted_request(error.request)
                raise
            delay = RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1)
            logger.warning(
                "Census API request to %s raised %s; retrying in %.1fs (attempt %d of %d).",
                base_url,
                type(error).__name__,
                delay,
                attempt + 1,
                MAX_REQUEST_ATTEMPTS,
            )
            _sleep(delay)
            continue

        if response.status_code != 429 and response.status_code < 500:
            return response
        if attempt == MAX_REQUEST_ATTEMPTS:
            return response
        delay = _retry_delay_seconds(response, attempt)
        logger.warning(
            "Census API returned %s for %s; retrying in %.1fs (attempt %d of %d).",
            response.status_code,
            _redacted_url(response.request.url),
            delay,
            attempt + 1,
            MAX_REQUEST_ATTEMPTS,
        )
        _sleep(delay)
    # Every final-attempt branch returns or raises; this guards future loop refactors.
    raise AssertionError("retry loop exhausted without returning or raising")  # pragma: no cover


def _validate_year(year: int) -> None:
    """Validate that ``year`` is a plausible 4-digit Census data year.

    Args:
        year (int): Candidate Census data year.

    Raises:
        ValueError: If ``year`` is not an integer between 2000 and 2050 inclusive.
    """

    if not isinstance(year, int) or not (2000 <= year <= 2050):
        raise ValueError(f"Year must be a 4-digit integer in [2000, 2050]. Received {year}.")


def _resolved_api_key(api_key: str | None) -> str:
    """Resolve the Census API key from the argument or the environment.

    Args:
        api_key (str | None): Explicit Census API key. When ``None``, the ``CENSUS_API_KEY``
            environment variable is consulted as a fallback.

    Returns:
        str: The resolved API key.

    Raises:
        ValueError: If neither source yields a key.
    """

    key = api_key if api_key is not None else os.getenv("CENSUS_API_KEY")
    if isinstance(key, str) and key.strip():
        return key.strip()
    raise ValueError(
        "No Census API key provided. Supply one via the api_key argument "
        "or the CENSUS_API_KEY environment variable. Register at "
        "https://api.census.gov/data/key_signup.html."
    )


def _census_get(
    client: httpx.Client,
    base_url: str,
    get: str,
    state: us.states.State,
    geometry: str,
    *,
    county_fips: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Issue one Census API query and return the payload as a DataFrame.

    Owns the whole request round trip: the ``for``/``in`` geography clauses for ``geometry``,
    API-key resolution, bounded retries for transient failures (429 and 5xx), the 429
    translation, the list-of-lists JSON to DataFrame conversion, and GEOID handling. When the
    response carries a ``GEO_ID`` column, it is replaced by a bare ``GEOID`` column with the
    Census summary-level stub stripped (e.g. ``0500000US55001`` becomes ``55001``).

    Args:
        client (httpx.Client): HTTP client used to issue the request.
        base_url (str): Fully formatted Census dataset URL (see ``PL_BASE_URL``/``ACS_BASE_URL``).
        get (str): Value for the Census ``get=`` parameter (a comma-joined variable list or a
            ``group(...)`` expression).
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``,
            ``"block group"``, or ``"block"``.
        county_fips (str | None): Optional 3-digit county scope for block queries. When omitted,
            blocks are requested across every county and tract in the state.
        api_key (str | None): Census API key, or ``CENSUS_API_KEY`` fallback.

    Returns:
        pd.DataFrame: One row per geography, with the first JSON row as column names and
        ``GEO_ID`` (when requested) replaced by the stripped ``GEOID`` column.

    Raises:
        ValueError: If ``geometry`` is unrecognized, no API key is available, or the payload is
            not a list of rows.
        CensusRateLimitError: When the Census API exceeds the bounded 429 retry policy.
        httpx.HTTPStatusError: For other HTTP error statuses (5xx also retried first).
    """

    key = _resolved_api_key(api_key)
    params: dict = {"get": get, "for": f"{geometry}:*", "key": key}
    if geometry == "state":
        geography_columns = ["state"]
        params["for"] = f"state:{state.fips}"
    elif geometry == "county":
        geography_columns = ["state", "county"]
        params["in"] = f"state:{state.fips}"
    elif geometry == "tract":
        geography_columns = ["state", "county", "tract"]
        params["in"] = [f"state:{state.fips}", "county:*"]
    elif geometry == "block group":
        geography_columns = ["state", "county", "tract", "block group"]
        params["in"] = [f"state:{state.fips}", "county:*", "tract:*"]
    elif geometry == "block":
        geography_columns = ["state", "county", "tract", "block"]
        county = f"county:{county_fips}" if county_fips is not None else "county:*"
        params["in"] = [f"state:{state.fips}", county, "tract:*"]
    else:
        raise ValueError(
            "Invalid geometry level. Must be one of 'state', 'county', 'tract', "
            "'block group', or 'block'."
        )

    response = _get_with_retries(client, base_url, params)
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise CensusRateLimitError(
            "Census API returned 429 Too Many Requests for "
            f"{_redacted_url(response.request.url)}. "
            + (f"Retry-After: {retry_after}s. " if retry_after else "")
            + "The Census API rate-limits per key; wait and retry, or spread requests out."
        )
    # Explicit stand-in for raise_for_status(): httpx embeds the full request URL, API key
    # included, in its message. Same exception type, redacted URL.
    if not response.is_success:
        location = response.headers.get("Location", "")
        if response.is_redirect and "invalid_key" in location:
            detail = " Census API rejected the API key."
        else:
            body = response.text.strip().replace(key, "REDACTED")
            detail = f" Response: {body[:200]}" if body else ""
        response = _redacted_response(response, key)
        raise httpx.HTTPStatusError(
            f"HTTP status {response.status_code} {response.reason_phrase} for url "
            f"'{_redacted_url(response.request.url)}'.{detail}",
            request=response.request,
            response=response,
        )
    if response.status_code == 204:
        requested_columns = [] if get.startswith("group(") else get.split(",")
        data = [list(dict.fromkeys([*requested_columns, *geography_columns]))]
    else:
        data = response.json()
    if not isinstance(data, list) or not data:
        raise ValueError(
            "Unexpected Census API response (expected a non-empty list of rows): "
            f"{repr(data)[:200]}"
        )
    header = data[0]
    valid_header = (
        isinstance(header, list)
        and bool(header)
        and all(isinstance(column, str) and column for column in header)
        and len(set(header)) == len(header)
    )
    valid_rows = valid_header and all(
        isinstance(row, list) and len(row) == len(header) for row in data[1:]
    )
    if not valid_rows:
        raise ValueError(
            "Unexpected Census API response (expected a unique string header and "
            f"equal-width list rows): {repr(data)[:200]}"
        )
    header = cast(list[str], header)
    frame = pd.DataFrame(data[1:], columns=pd.Index(header))

    if "GEO_ID" in frame.columns:
        frame["GEOID"] = frame["GEO_ID"].astype("string").str.split("US").str[-1]
        frame.drop(columns=["GEO_ID"], inplace=True)
    return frame


def _cast_count_columns_to_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    """Cast ``columns`` of ``frame`` to numeric in place, raising on non-numeric cells.

    The Census API returns every value as a JSON string; getters use this after renaming to
    coerce their count columns.
    """

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
