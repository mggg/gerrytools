import httpx
import pandas as pd
import pytest
import us

from gerrytools.data.uscensus.block_cvap import (
    _estimate_block_cvap_from_inputs,
    _state_cvap_rates,
    _tract_cvap_rates,
    block_cvap_estimates,
)
from gerrytools.data.uscensus.census_tables import (
    ACSCVAPTableInfo,
    ACSVAPTableInfo,
    PLBlockVAPTableInfo,
)
from tests.data._helpers import MockHTTP, acs_api_payload


def valid_vap_cvap_payload(
    geoid: str = "55001",
    *,
    vap_value: int = 2,
    cvap_value: int = 1,
) -> list[list[str]]:
    """Build a B05003 payload whose citizen child cells do not exceed VAP."""
    vap_table = ACSVAPTableInfo()
    payload = acs_api_payload([vap_table, ACSCVAPTableInfo()], {geoid: cvap_value})
    vap_columns = set(vap_table.construct_long_names(suffix="E"))
    for row in payload[1:]:
        for index, column in enumerate(payload[0]):
            if column in vap_columns:
                row[index] = str(vap_value)
    return payload


# ==========================
# == STATE FALLBACK RATES ==
# ==========================


class TestStateCvapRates:
    def test_rate_is_cvap_over_vap(self):
        state_est = pd.DataFrame([{"total_vap_acs5_24": 1000, "total_cvap_acs5_24": 600}])
        rates = _state_cvap_rates(state_est, acs_year=2024, race_categories=("total",))
        assert rates["total"] == pytest.approx(0.6)

    def test_zero_vap_yields_zero_rate(self):
        state_est = pd.DataFrame([{"white_vap_acs5_24": 0, "white_cvap_acs5_24": 0}])
        rates = _state_cvap_rates(state_est, acs_year=2024, race_categories=("white",))
        assert rates["white"] == 0.0

    def test_cvap_larger_than_vap_raises(self):
        state_est = pd.DataFrame([{"white_vap_acs5_24": 100, "white_cvap_acs5_24": 101}])
        with pytest.raises(ValueError, match=r"white.*CVAP.*101.*VAP.*100"):
            _state_cvap_rates(state_est, acs_year=2024, race_categories=("white",))

    @pytest.mark.parametrize(
        ("vap", "cvap"),
        [
            (float("nan"), 60),
            (100, float("nan")),
            (-1, 0),
            (100, -1),
        ],
    )
    def test_invalid_state_estimates_raise(self, vap, cvap):
        state_est = pd.DataFrame([{"white_vap_acs5_24": vap, "white_cvap_acs5_24": cvap}])
        with pytest.raises(ValueError, match=r"white.*finite and nonnegative"):
            _state_cvap_rates(state_est, acs_year=2024, race_categories=("white",))

    @pytest.mark.parametrize("row_count", [0, 2])
    def test_requires_exactly_one_state_row(self, row_count):
        state_est = pd.DataFrame(
            [{"white_vap_acs5_24": 100, "white_cvap_acs5_24": 60} for _ in range(row_count)]
        )
        with pytest.raises(ValueError, match="exactly one row"):
            _state_cvap_rates(state_est, acs_year=2024, race_categories=("white",))


# =============================
# == TRACT RATES + THRESHOLD ==
# =============================


class TestTractCvapRates:
    def test_below_threshold_is_masked_to_nan(self):
        tract_est = pd.DataFrame(
            {"total_vap_acs5_24": [100, 10], "total_cvap_acs5_24": [80, 5]},
            index=pd.Index(["T1", "T2"]),
        )
        rates = _tract_cvap_rates(
            tract_est,
            denominator_threshold=20,
            acs_year=2024,
            race_categories=("total",),
        )
        assert rates["total"]["T1"] == pytest.approx(0.8)
        assert bool(pd.isna(rates["total"]["T2"]))

    def test_cvap_larger_than_vap_identifies_tract(self):
        tract_est = pd.DataFrame(
            {"total_vap_acs5_24": [100], "total_cvap_acs5_24": [101]},
            index=pd.Index(["55001000100"]),
        )
        with pytest.raises(ValueError, match=r"total.*55001000100.*CVAP.*101.*VAP.*100"):
            _tract_cvap_rates(
                tract_est,
                denominator_threshold=20,
                acs_year=2024,
                race_categories=("total",),
            )

    @pytest.mark.parametrize(
        ("vap", "cvap"),
        [(float("nan"), 10), (100, float("nan")), (-1, 0), (100, -1)],
    )
    def test_invalid_tract_estimates_raise(self, vap, cvap):
        tract_est = pd.DataFrame(
            {"total_vap_acs5_24": [vap], "total_cvap_acs5_24": [cvap]},
            index=pd.Index(["55001000100"]),
        )

        with pytest.raises(ValueError, match=r"total.*55001000100.*finite and nonnegative"):
            _tract_cvap_rates(
                tract_est,
                denominator_threshold=20,
                acs_year=2024,
                race_categories=("total",),
            )


# ===========================
# == BLOCK CVAP ESTIMATION ==
# ===========================


class TestEstimateBlockCvap:
    def _inputs(self):
        table = PLBlockVAPTableInfo(race_categories=("total", "white"))

        blocks = pd.DataFrame(
            {
                "GEOID": ["b1", "b2"],
                "STATEFP": ["55", "55"],
                "COUNTYFP": ["001", "001"],
                "TRACTCE": ["000100", "000200"],
                "BLOCKCE": ["1000", "2000"],
                # b1 is in tract T1 (rate available), b2 in T2 (falls back).
                "TRACT_GEOID": ["T1", "T2"],
            }
        )
        # Every PL VAP column must be present for the output projection.
        for name in table.construct_short_names(year=2020):
            blocks[name] = [0, 0]
        blocks["total_vap_20"] = [50, 5]
        blocks["white_vap_20"] = [50, 5]

        tract_est = pd.DataFrame(
            {
                "total_vap_acs5_24": [100, 10],
                "total_cvap_acs5_24": [80, 5],
                "white_vap_acs5_24": [100, 10],
                "white_cvap_acs5_24": [80, 5],
            },
            index=pd.Index(["T1", "T2"]),
        )
        state_est = pd.DataFrame(
            [
                {
                    "total_vap_acs5_24": 1000,
                    "total_cvap_acs5_24": 600,
                    "white_vap_acs5_24": 1000,
                    "white_cvap_acs5_24": 600,
                }
            ]
        )
        return blocks, tract_est, state_est, table

    def test_uses_tract_rate_then_state_fallback(self):
        blocks, tract_est, state_est, table = self._inputs()

        result = _estimate_block_cvap_from_inputs(
            blocks,
            tract_est,
            state_est,
            denominator_threshold=20,
            table=table,
            acs_year=2024,
            pl_year=2020,
        )
        by_geoid = result.set_index("GEOID")

        # b1's tract T1 has rate 0.8 (VAP 100 >= 20): 50 * 0.8 = 40.
        assert by_geoid.loc["b1", "total_cvap_acs5_24_pl_20"] == pytest.approx(40.0)
        # b2's tract T2 is below threshold, so the 0.6 state rate applies:
        # 5 * 0.6 = 3.0.
        assert by_geoid.loc["b2", "total_cvap_acs5_24_pl_20"] == pytest.approx(3.0)
        assert by_geoid.loc["b1", "white_cvap_acs5_24_pl_20"] == pytest.approx(40.0)

    def test_output_carries_geoid_components_and_cvap_columns(self):
        blocks, tract_est, state_est, table = self._inputs()

        result = _estimate_block_cvap_from_inputs(
            blocks,
            tract_est,
            state_est,
            denominator_threshold=20,
            table=table,
            acs_year=2024,
            pl_year=2020,
        )

        for column in ("GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLOCKCE", "TRACT_GEOID"):
            assert column in result.columns
        assert "total_cvap_acs5_24_pl_20" in result.columns
        assert "white_cvap_acs5_24_pl_20" in result.columns

    def test_tract_absent_from_acs_uses_state_fallback(self):
        # b2's tract is entirely missing from the ACS tract table (documented for very small
        # tracts), not merely below the denominator threshold.
        blocks, tract_est, state_est, table = self._inputs()
        blocks["TRACT_GEOID"] = ["T1", "T_MISSING"]

        result = _estimate_block_cvap_from_inputs(
            blocks,
            tract_est,
            state_est,
            denominator_threshold=20,
            table=table,
            acs_year=2024,
            pl_year=2020,
        )
        by_geoid = result.set_index("GEOID")

        # b2 gets the 0.6 state rate: 5 * 0.6 = 3.0. b1 still uses its tract rate.
        assert by_geoid.loc["b2", "total_cvap_acs5_24_pl_20"] == pytest.approx(3.0)
        assert by_geoid.loc["b1", "total_cvap_acs5_24_pl_20"] == pytest.approx(40.0)


# ==========================
# == BLOCK CVAP ESTIMATES ==
# ==========================


class TestBlockCvapEstimates:
    def test_invalid_year_raises(self):
        with pytest.raises(ValueError, match="4-digit integer"):
            block_cvap_estimates(us.states.WI, acs_year=1999, api_key="k")

    @pytest.mark.parametrize("threshold", [True, 0, -1, 1.5])
    def test_invalid_denominator_threshold_raises_before_any_request(
        self,
        threshold,
        mock_http: MockHTTP,
    ):
        with pytest.raises(ValueError, match="denominator_threshold"):
            block_cvap_estimates(us.states.WI, denominator_threshold=threshold, api_key="k")

        assert mock_http.requests == []

    def test_unsupported_pl_year_raises_before_any_request(self, mock_http: MockHTTP):
        # Regression: pl_year=2010 used to fail late, after burning the ACS requests, because
        # the block VAP table hardcodes 2020 P3/P4 variable names.
        with pytest.raises(ValueError, match=r"PL vintages \(2020,\)"):
            block_cvap_estimates(us.states.WI, acs_year=2024, pl_year=2010, api_key="k")

        assert mock_http.requests == []

    def test_pre_2020_acs_year_raises_tract_vintage_mismatch(self, mock_http: MockHTTP):
        # ACS 5-year vintages before 2020 tabulate on 2010-vintage tract boundaries, which do
        # not match the tract GEOIDs derived from 2020 blocks; every block would silently fall
        # back to the statewide rate.
        with pytest.raises(ValueError, match="2010-vintage"):
            block_cvap_estimates(us.states.WI, acs_year=2019, pl_year=2020, api_key="k")

        assert mock_http.requests == []

    @pytest.mark.parametrize(
        "county_response", [{"json": [["NAME", "state", "county"]]}, {"status_code": 204}]
    )
    def test_state_with_no_counties_returns_empty_frame(self, mock_http: MockHTTP, county_response):
        # ACS tract/state VAP+CVAP fetches succeed...
        mock_http.route(
            url_contains="/acs/acs5",
            json=valid_vap_cvap_payload(),
        )
        # ...but the decennial PL county enumeration returns no counties.
        mock_http.route(url_contains="/dec/pl", **county_response)

        result = block_cvap_estimates(us.states.WI, acs_year=2024, pl_year=2020, api_key="k")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_county_joins_block_to_matching_tract_rate(self, mock_http: MockHTTP):
        pl_variables = list(PLBlockVAPTableInfo().construct_variable_names())
        # 15-char block GEOID: state 55 / county 001 / tract 000100 / block 1000.
        block_geoid = "550010001001000"

        def acs_payload(request: httpx.Request) -> httpx.Response:
            geography = request.url.params["for"]
            if geography.startswith("tract:"):
                payload = valid_vap_cvap_payload("55001000100", vap_value=100, cvap_value=10)
            else:
                payload = valid_vap_cvap_payload("55", vap_value=100, cvap_value=5)
            return httpx.Response(200, json=payload)

        mock_http.route(url_contains="/acs/acs5", responder=acs_payload)
        # County enumeration returns a single county (for=county:*)...
        mock_http.route(
            url_contains="for=county",
            json=[["NAME", "state", "county"], ["Test County, WI", "55", "001"]],
        )
        # ...whose block query (for=block:*) returns a single block.
        mock_http.route(
            url_contains="for=block",
            json=[
                ["GEO_ID"] + pl_variables,
                [f"1000000US{block_geoid}"] + ["10"] * len(pl_variables),
            ],
        )

        result = block_cvap_estimates(us.states.WI, acs_year=2024, pl_year=2020, api_key="k")

        assert len(result) == 1
        block = result.iloc[0]
        assert block["GEOID"] == block_geoid
        assert block["STATEFP"] == "55"
        assert block["TRACT_GEOID"] == "55001000100"
        assert block["total_cvap_acs5_24_pl_20"] == pytest.approx(2.0)
        assert pd.api.types.is_numeric_dtype(result["total_cvap_acs5_24_pl_20"])

    def test_multiple_counties_concatenate_with_clean_index(self, mock_http: MockHTTP):
        pl_variables = list(PLBlockVAPTableInfo().construct_variable_names())
        first_block_geoid = "550010001001000"
        second_block_geoid = "550030002002000"

        mock_http.route(
            url_contains="/acs/acs5",
            json=valid_vap_cvap_payload(),
        )
        mock_http.route(
            url_contains="for=county",
            json=[
                ["NAME", "state", "county"],
                ["First County, WI", "55", "001"],
                ["Second County, WI", "55", "003"],
            ],
        )
        # Each county's block query is scoped via in=county:<fips>; distinct VAP values per
        # county so a skipped or duplicated county changes the output.
        mock_http.route(
            url_contains="county%3A001",
            json=[
                ["GEO_ID"] + pl_variables,
                [f"1000000US{first_block_geoid}"] + ["10"] * len(pl_variables),
            ],
        )
        mock_http.route(
            url_contains="county%3A003",
            json=[
                ["GEO_ID"] + pl_variables,
                [f"1000000US{second_block_geoid}"] + ["20"] * len(pl_variables),
            ],
        )

        result = block_cvap_estimates(us.states.WI, acs_year=2024, pl_year=2020, api_key="k")

        assert list(result["GEOID"]) == [first_block_geoid, second_block_geoid]
        assert list(result["COUNTYFP"]) == ["001", "003"]
        by_geoid = result.set_index("GEOID")
        assert by_geoid.loc[first_block_geoid, "total_vap_20"] == 10
        assert by_geoid.loc[second_block_geoid, "total_vap_20"] == 20
        # pd.concat(..., ignore_index=True): the concatenated frame gets a clean 0..n-1 index.
        assert list(result.index) == [0, 1]

    def test_acs_fetches_skip_margin_of_error_requests(self, mock_http: MockHTTP):
        # Regression: the rate computation never consumes MOE, but the ACS path used to fetch
        # (and drop) one MOE frame per table per geography: 4 wasted requests.
        pl_variables = list(PLBlockVAPTableInfo().construct_variable_names())
        mock_http.route(
            url_contains="/acs/acs5",
            json=valid_vap_cvap_payload(),
        )
        mock_http.route(
            url_contains="for=county",
            json=[["NAME", "state", "county"], ["Test County, WI", "55", "001"]],
        )
        mock_http.route(
            url_contains="for=block",
            json=[
                ["GEO_ID"] + pl_variables,
                ["1000000US550010001001000"] + ["10"] * len(pl_variables),
            ],
        )

        block_cvap_estimates(us.states.WI, acs_year=2024, pl_year=2020, api_key="k")

        acs_urls = [url for url in mock_http.urls if "/acs/acs5" in url]
        # Two tables (VAP, CVAP) at two geographies (tract, state): estimates only.
        assert len(acs_urls) == 4
        moe_variables = ACSVAPTableInfo().construct_long_names(suffix="M").keys()
        assert not any(variable in url for url in acs_urls for variable in moe_variables)

    def test_one_client_serves_the_whole_pipeline(
        self,
        mock_http: MockHTTP,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # ACS rate fetches and the per-county PL fetches all share one httpx client.
        pl_variables = list(PLBlockVAPTableInfo().construct_variable_names())
        mock_http.route(
            url_contains="/acs/acs5",
            json=valid_vap_cvap_payload(),
        )
        mock_http.route(
            url_contains="for=county",
            json=[["NAME", "state", "county"], ["Test County, WI", "55", "001"]],
        )
        mock_http.route(
            url_contains="for=block",
            json=[
                ["GEO_ID"] + pl_variables,
                ["1000000US550010001001000"] + ["10"] * len(pl_variables),
            ],
        )
        client_factory = httpx.Client
        client_count = 0

        def counting_client(*args, **kwargs):
            nonlocal client_count
            client_count += 1
            return client_factory(*args, **kwargs)

        monkeypatch.setattr(httpx, "Client", counting_client)

        block_cvap_estimates(us.states.WI, acs_year=2024, pl_year=2020, api_key="k")

        assert client_count == 1
