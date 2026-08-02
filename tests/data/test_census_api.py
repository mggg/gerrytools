import logging
from datetime import datetime, timezone

import httpx
import pytest
import us

from gerrytools.data.uscensus._api import (
    MAX_REQUEST_ATTEMPTS,
    PL_BASE_URL,
    RETRY_BASE_DELAY_SECONDS,
    CensusRateLimitError,
    _census_get,
    _resolved_api_key,
    _validate_year,
)
from tests.data._helpers import MockHTTP

# ==================================
# == YEAR VALIDATION              ==
# ==================================


class TestValidateYear:
    @pytest.mark.parametrize("year", [2000, 2010, 2020, 2050])
    def test_accepts_valid_years(self, year):
        _validate_year(year)  # should not raise

    @pytest.mark.parametrize("year", [1999, 2051, 2020.0, "2020", None])
    def test_rejects_invalid_years(self, year):
        with pytest.raises(ValueError, match="4-digit integer"):
            _validate_year(year)


# ==================================
# == API KEY RESOLUTION           ==
# ==================================


class TestResolvedApiKey:
    def test_explicit_key_wins(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CENSUS_API_KEY", "from-env")
        assert _resolved_api_key("explicit") == "explicit"

    def test_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CENSUS_API_KEY", "from-env")
        assert _resolved_api_key(None) == "from-env"

    @pytest.mark.parametrize("explicit", [True, False])
    def test_strips_key_whitespace(self, explicit: bool, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CENSUS_API_KEY", "  from-env\n")
        key = "  explicit\n" if explicit else None

        assert _resolved_api_key(key) == ("explicit" if explicit else "from-env")

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CENSUS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No Census API key"):
            _resolved_api_key(None)

    @pytest.mark.parametrize("key", ["", "   "])
    def test_blank_explicit_key_raises(self, key, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CENSUS_API_KEY", "from-env")
        with pytest.raises(ValueError, match="No Census API key"):
            _resolved_api_key(key)

    def test_blank_environment_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CENSUS_API_KEY", "   ")
        with pytest.raises(ValueError, match="No Census API key"):
            _resolved_api_key(None)


# ==================================
# == COMPOSED FETCH               ==
# ==================================

BASE_URL = PL_BASE_URL.format(year=2020)


def _client(mock: MockHTTP) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(mock.handle))


def _fetch(mock: MockHTTP, geometry: str, *, county_fips: str | None = None):
    with _client(mock) as client:
        return _census_get(
            client,
            BASE_URL,
            "GEO_ID",
            us.states.WI,
            geometry,
            county_fips=county_fips,
            api_key="k",
        )


class TestCensusGetGeographyClauses:
    def _issued_query(self, mock: MockHTTP) -> str:
        [request] = mock.requests
        return str(request.url)

    def test_state_scopes_the_for_clause(self):
        mock = MockHTTP().route(json=[["GEO_ID"], ["1000000US55"]])
        _fetch(mock, "state")
        url = self._issued_query(mock)
        assert f"for=state%3A{us.states.WI.fips}" in url
        assert "in=" not in url

    def test_county_wildcards_for_and_scopes_in(self):
        mock = MockHTTP().route(json=[["GEO_ID"], ["0500000US55001"]])
        _fetch(mock, "county")
        url = self._issued_query(mock)
        assert "for=county%3A%2A" in url
        assert f"in=state%3A{us.states.WI.fips}" in url

    def test_tract_scopes_through_all_counties(self):
        mock = MockHTTP().route(json=[["GEO_ID"], ["1400000US55001950100"]])
        _fetch(mock, "tract")
        url = self._issued_query(mock)
        assert f"in=state%3A{us.states.WI.fips}" in url
        assert "in=county%3A%2A" in url

    def test_block_group_scopes_through_tract(self):
        mock = MockHTTP().route(json=[["GEO_ID"], ["1500000US550019501001"]])
        _fetch(mock, "block group")
        url = self._issued_query(mock)
        assert "in=county%3A%2A" in url
        assert "in=tract%3A%2A" in url

    def test_block_scopes_through_all_counties_and_tracts(self):
        mock = MockHTTP().route(json=[["GEO_ID"], ["1000000US550019501001000"]])
        _fetch(mock, "block")
        url = self._issued_query(mock)
        assert "in=county%3A%2A" in url
        assert "in=tract%3A%2A" in url

    def test_block_with_county_scopes_to_it(self):
        mock = MockHTTP().route(json=[["GEO_ID"], ["1000000US550019501001000"]])
        _fetch(mock, "block", county_fips="001")
        url = self._issued_query(mock)
        assert "in=county%3A001" in url
        assert "in=tract%3A%2A" in url

    def test_unknown_geometry_raises(self):
        with pytest.raises(ValueError, match="Invalid geometry"):
            _fetch(MockHTTP(), "precinct")

    def test_key_travels_in_query(self):
        mock = MockHTTP().route(json=[["GEO_ID"], ["1000000US55"]])
        _fetch(mock, "state")
        assert "key=k" in self._issued_query(mock)


class TestCensusGetResponses:
    def test_first_row_becomes_header_and_geoid_is_stripped(self):
        mock = MockHTTP().route(
            json=[["GEO_ID", "P1_001N"], ["1000000US55", "5000"], ["0500000US55001", "77"]]
        )
        frame = _fetch(mock, "state")
        assert list(frame.columns) == ["P1_001N", "GEOID"]
        assert list(frame["GEOID"]) == ["55", "55001"]
        assert frame.iloc[0]["P1_001N"] == "5000"

    def test_frames_without_geo_id_pass_through(self):
        mock = MockHTTP().route(json=[["NAME", "county"], ["Adams County", "001"]])
        with _client(mock) as client:
            frame = _census_get(client, BASE_URL, "NAME", us.states.WI, "county", api_key="k")
        assert list(frame.columns) == ["NAME", "county"]

    def test_persistent_429_raises_rate_limit_error_after_retries(self):
        mock = MockHTTP().route(
            responder=lambda request: httpx.Response(429, headers={"Retry-After": "7"})
        )
        with pytest.raises(CensusRateLimitError, match="Retry-After: 7s"):
            _fetch(mock, "state")
        assert len(mock.requests) == MAX_REQUEST_ATTEMPTS

    def test_persistent_500_raises_http_status_error_after_retries(self):
        mock = MockHTTP().route(status_code=500, json=[["GEO_ID"]])
        with pytest.raises(httpx.HTTPStatusError):
            _fetch(mock, "state")
        assert len(mock.requests) == MAX_REQUEST_ATTEMPTS

    def test_non_429_client_error_fails_without_retry(self):
        mock = MockHTTP().route(status_code=404, text="no such dataset")
        with pytest.raises(httpx.HTTPStatusError, match="no such dataset"):
            _fetch(mock, "state")
        assert len(mock.requests) == 1

    def test_no_content_returns_an_empty_frame(self):
        mock = MockHTTP().route(status_code=204)
        frame = _fetch(mock, "county")
        assert frame.empty
        assert list(frame.columns) == ["state", "county", "GEOID"]

    def test_invalid_key_redirect_has_an_actionable_error(self):
        mock = MockHTTP().route(
            responder=lambda request: httpx.Response(
                302, headers={"Location": "https://api.census.gov/data/invalid_key.html"}
            )
        )
        with pytest.raises(httpx.HTTPStatusError, match="rejected the API key"):
            _fetch(mock, "state")

    def test_transient_429s_are_retried_to_success(
        self,
        recorded_retry_sleeps: list[float],
        caplog: pytest.LogCaptureFixture,
    ):
        payload = [["GEO_ID"], ["1000000US55"]]
        responses = iter(
            [
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(429),
                httpx.Response(200, json=payload),
            ]
        )
        mock = MockHTTP().route(responder=lambda request: next(responses))

        with caplog.at_level(logging.WARNING, logger="gerrytools.data.uscensus._api"):
            frame = _fetch(mock, "state")

        assert list(frame["GEOID"]) == ["55"]
        assert len(mock.requests) == 3
        # First delay honors Retry-After; the second falls back to exponential backoff.
        assert recorded_retry_sleeps == [7.0, 2 * RETRY_BASE_DELAY_SECONDS]
        assert sum("retrying" in message for message in caplog.messages) == 2

    def test_http_date_retry_after_is_honored(
        self,
        monkeypatch: pytest.MonkeyPatch,
        recorded_retry_sleeps: list[float],
    ):
        monkeypatch.setattr(
            "gerrytools.data.uscensus._api.time.time",
            lambda: datetime(2026, 10, 21, 7, 27, 53, tzinfo=timezone.utc).timestamp(),
        )
        payload = [["GEO_ID"], ["1000000US55"]]
        responses = iter(
            [
                httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
                httpx.Response(200, json=payload),
            ]
        )
        mock = MockHTTP().route(responder=lambda request: next(responses))

        frame = _fetch(mock, "state")

        assert list(frame["GEOID"]) == ["55"]
        assert recorded_retry_sleeps == [7.0]

    def test_malformed_retry_after_falls_back_to_backoff(
        self,
        recorded_retry_sleeps: list[float],
    ):
        payload = [["GEO_ID"], ["1000000US55"]]
        responses = iter(
            [
                httpx.Response(429, headers={"Retry-After": "not-a-delay"}),
                httpx.Response(200, json=payload),
            ]
        )
        mock = MockHTTP().route(responder=lambda request: next(responses))

        frame = _fetch(mock, "state")

        assert list(frame["GEOID"]) == ["55"]
        assert recorded_retry_sleeps == [RETRY_BASE_DELAY_SECONDS]

    def test_retry_after_beyond_cap_raises_without_retry(
        self,
        recorded_retry_sleeps: list[float],
    ):
        payload = [["GEO_ID"], ["1000000US55"]]
        responses = iter(
            [
                httpx.Response(429, headers={"Retry-After": "3600"}),
                httpx.Response(200, json=payload),
            ]
        )
        mock = MockHTTP().route(responder=lambda request: next(responses))

        with pytest.raises(CensusRateLimitError, match="exceeds the 60-second safety limit"):
            _fetch(mock, "state")

        assert len(mock.requests) == 1
        assert recorded_retry_sleeps == []

    def test_transient_500_is_retried_to_success(self):
        payload = [["GEO_ID"], ["1000000US55"]]
        responses = iter([httpx.Response(503), httpx.Response(200, json=payload)])
        mock = MockHTTP().route(responder=lambda request: next(responses))

        frame = _fetch(mock, "state")

        assert list(frame["GEOID"]) == ["55"]
        assert len(mock.requests) == 2

    def test_transient_transport_errors_are_retried_to_success(
        self,
        recorded_retry_sleeps: list[float],
    ):
        payload = [["GEO_ID"], ["1000000US55"]]
        mock = MockHTTP()

        def responder(request: httpx.Request) -> httpx.Response:
            if len(mock.requests) < 3:
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(200, json=payload)

        mock.route(responder=responder)

        frame = _fetch(mock, "state")

        assert list(frame["GEOID"]) == ["55"]
        assert len(mock.requests) == 3
        assert recorded_retry_sleeps == [1.0, 2.0]

    def test_persistent_transport_error_raises_after_retry_budget(
        self,
        recorded_retry_sleeps: list[float],
    ):
        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection failed", request=request)

        mock = MockHTTP().route(responder=fail)

        with pytest.raises(httpx.ConnectError, match="connection failed") as excinfo:
            _fetch(mock, "state")

        assert excinfo.value.request.url.params["key"] == "REDACTED"
        assert len(mock.requests) == MAX_REQUEST_ATTEMPTS
        assert recorded_retry_sleeps == [1.0, 2.0, 4.0]

    @pytest.mark.parametrize("payload", [{"error": "bad request"}, [], "oops"])
    def test_non_list_payload_raises_value_error(self, payload):
        # A 200 with an unexpected (non-list-of-rows) body fails loudly, not with an opaque slice.
        mock = MockHTTP().route(json=payload)
        with pytest.raises(ValueError, match="Unexpected Census API response"):
            _fetch(mock, "state")

    @pytest.mark.parametrize(
        "payload",
        [
            [[]],
            [["GEO_ID", "GEO_ID"], ["1000000US55", "1000000US55"]],
            [["GEO_ID"], "x"],
            [["GEO_ID", "P1_001N"], ["1000000US55"]],
        ],
    )
    def test_malformed_rows_raise_value_error(self, payload):
        mock = MockHTTP().route(json=payload)
        with pytest.raises(ValueError, match="Unexpected Census API response"):
            _fetch(mock, "state")


class TestApiKeyRedaction:
    # The key travels as a query parameter, so any surfaced request URL would leak it.
    SECRET_KEY = "super-secret-census-key"

    def _fetch(self, mock: MockHTTP):
        with _client(mock) as client:
            return _census_get(
                client,
                BASE_URL,
                "GEO_ID",
                us.states.WI,
                "state",
                api_key=self.SECRET_KEY,
            )

    def test_rate_limit_error_and_retry_logs_redact_the_key(self, caplog: pytest.LogCaptureFixture):
        mock = MockHTTP().route(
            responder=lambda request: httpx.Response(429, headers={"Retry-After": "7"})
        )

        with caplog.at_level(logging.WARNING, logger="gerrytools.data.uscensus._api"):
            with pytest.raises(CensusRateLimitError, match="key=REDACTED") as excinfo:
                self._fetch(mock)

        assert self.SECRET_KEY not in str(excinfo.value)
        retry_messages = [record.getMessage() for record in caplog.records]
        # The retry warnings do format the URL, just with the key masked.
        assert any("api.census.gov" in message for message in retry_messages)
        assert all(self.SECRET_KEY not in message for message in retry_messages)

    def test_http_status_error_and_retry_logs_redact_the_key(
        self, caplog: pytest.LogCaptureFixture
    ):
        mock = MockHTTP().route(status_code=500, text="boom")

        with caplog.at_level(logging.WARNING, logger="gerrytools.data.uscensus._api"):
            with pytest.raises(httpx.HTTPStatusError, match="key=REDACTED") as excinfo:
                self._fetch(mock)

        assert self.SECRET_KEY not in str(excinfo.value)
        assert self.SECRET_KEY not in str(excinfo.value.request.url)
        assert self.SECRET_KEY not in str(excinfo.value.response.request.url)
        assert excinfo.value.request.url.params["key"] == "REDACTED"
        retry_messages = [record.getMessage() for record in caplog.records]
        assert any("api.census.gov" in message for message in retry_messages)
        assert all(self.SECRET_KEY not in message for message in retry_messages)

    def test_httpx_info_log_redacts_the_key(self, caplog: pytest.LogCaptureFixture):
        mock = MockHTTP().route(json=[["GEO_ID"], ["1000000US55"]])

        with caplog.at_level(logging.INFO, logger="httpx"):
            self._fetch(mock)

        messages = [record.getMessage() for record in caplog.records if record.name == "httpx"]
        assert messages
        assert all(self.SECRET_KEY not in message for message in messages)
        assert any("key=REDACTED" in message for message in messages)

    def test_httpx_filter_leaves_non_census_logs_unchanged(self, caplog: pytest.LogCaptureFixture):
        url = httpx.URL(f"https://example.com/data?key={self.SECRET_KEY}")

        with caplog.at_level(logging.INFO, logger="httpx"):
            logging.getLogger("httpx").info("HTTP Request: GET %s", url)

        messages = [record.getMessage() for record in caplog.records if record.name == "httpx"]
        assert messages == [f"HTTP Request: GET {url}"]

    def test_http_error_body_redacts_an_echoed_key(self):
        mock = MockHTTP().route(status_code=400, text=f"bad key {self.SECRET_KEY}")

        with pytest.raises(httpx.HTTPStatusError, match="bad key REDACTED") as excinfo:
            self._fetch(mock)

        assert self.SECRET_KEY not in str(excinfo.value)
        assert self.SECRET_KEY not in excinfo.value.response.text
