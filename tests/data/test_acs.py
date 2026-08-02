import warnings
from dataclasses import replace
from typing import cast

import httpx
import pandas as pd
import pytest
import us

from gerrytools.data.uscensus.acs import (
    _condense,
    _get_acs_data,
    _normalize_acs_survey,
    acs,
    acs_full,
    cvap,
)
from gerrytools.data.uscensus.census_tables import (
    ACSAgeTableInfo,
    ACSCVAPTableInfo,
    ACSHispByRaceTableInfo,
    ACSRacePopTableInfo,
    ACSTableInfo,
    ACSTotPopTableInfo,
    ACSVAPTableInfo,
)
from tests.data._helpers import MockHTTP, acs_api_payload, geo_id_payload

# ==========================
# == SURVEY NORMALIZATION ==
# ==========================


class TestNormalizeAcsSurvey:
    @pytest.mark.parametrize("value", ["acs5", "5", 5, "5year", "ACS-5", "acs 5"])
    def test_normalizes_to_acs5(self, value):
        assert _normalize_acs_survey(value) == "acs5"

    @pytest.mark.parametrize("value", ["acs1", "1", 1, "1year", "ACS_1"])
    def test_normalizes_to_acs1(self, value):
        assert _normalize_acs_survey(value) == "acs1"

    @pytest.mark.parametrize("value", ["acs7", "yearly", 3.5, None])
    def test_rejects_unknown_survey(self, value):
        with pytest.raises(ValueError, match="Invalid ACS survey"):
            _normalize_acs_survey(value)


# ==============
# == CONDENSE ==
# ==============


class TestCondense:
    # VAP table restricted to the ungrouped (total) variables, so frames only need those columns.
    _total_only_vap = replace(ACSVAPTableInfo(), groups_tup=("",))

    def test_sums_group_indices(self):
        data = pd.DataFrame(
            {"B05003_008E": [3.0, 1.0], "B05003_019E": [4.0, 1.0]},
            index=pd.Index(["55001", "55003"]),
        )
        result = _condense(data, self._total_only_vap, suffix="E", label="")
        assert result["total_vap"].tolist() == [7.0, 2.0]

    def test_missing_source_columns_raise(self):
        # Only the total group's columns are present; the fetchers always request every declared
        # variable, so a gap means the frame and table definition are out of sync.
        data = pd.DataFrame({"B05003_008E": [1.0], "B05003_019E": [1.0]}, index=pd.Index(["55001"]))
        with pytest.raises(ValueError, match="Cannot condense group 'white_vap'"):
            _condense(data, ACSVAPTableInfo(), suffix="E", label="")

    def test_margins_of_error_use_root_sum_of_squares(self):
        data = pd.DataFrame(
            {"B05003_008M": [3.0], "B05003_019M": [4.0]},
            index=pd.Index(["55001"]),
        )

        result = _condense(data, self._total_only_vap, suffix="M", label="_moe")

        assert result.loc["55001", "total_vap_moe"] == pytest.approx(5.0)


# ===============
# == ACS FETCH ==
# ===============


class TestAcs:
    def test_no_content_returns_empty_declared_frames(self, mock_http: MockHTTP):
        mock_http.route(url_contains="/acs/", status_code=204)

        est, moe = acs(
            us.states.AS,
            "county",
            2023,
            tables=[ACSTotPopTableInfo()],
            api_key="k",
        )

        assert est.empty and moe.empty
        assert list(est.columns) == ["total_pop_acs5_23"]
        assert list(moe.columns) == ["total_pop_moe_acs5_23"]

    def test_est_and_moe_requests_list_the_vap_variables(self, mock_http: MockHTTP):
        table = ACSVAPTableInfo()
        mock_http.route(url_contains="/acs/", json=acs_api_payload([table], {"55001": 1}))

        acs(us.states.WI, "county", 2023, tables=[table], api_key="k")

        assert mock_http.requests[0].url.params["get"].split(",") == [
            "GEO_ID",
            *table.construct_long_names(suffix="E"),
        ]
        assert mock_http.requests[1].url.params["get"].split(",") == [
            "GEO_ID",
            *table.construct_long_names(suffix="M"),
        ]

    def test_reuses_one_client_across_tables(
        self,
        mock_http: MockHTTP,
        monkeypatch: pytest.MonkeyPatch,
    ):
        tables: list[ACSTableInfo] = [ACSTotPopTableInfo(), ACSVAPTableInfo()]
        mock_http.route(url_contains="/acs/", json=acs_api_payload(tables, {"55001": 1}))
        client_factory = httpx.Client
        client_count = 0

        def counting_client(*args, **kwargs):
            nonlocal client_count
            client_count += 1
            return client_factory(*args, **kwargs)

        monkeypatch.setattr(httpx, "Client", counting_client)

        acs(us.states.WI, "county", 2023, tables=tables, api_key="k")

        assert client_count == 1

    def test_leading_zero_geoids_survive_as_strings(self, mock_http: MockHTTP):
        # Alabama county GEOIDs start with "01"; any integer coercion would strip the zero.
        mock_http.route(
            url_contains="/acs/",
            json=acs_api_payload([ACSTotPopTableInfo()], {"01001": 100}),
        )

        est, _ = acs(us.states.AL, "county", 2023, tables=[ACSTotPopTableInfo()], api_key="k")

        assert list(est.index) == ["01001"]
        assert str(est.index.dtype) in {"object", "string"}

    def test_totpop_county_estimates(self, mock_http: MockHTTP):
        mock_http.route(
            url_contains="/acs/",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 100, "55003": 200}),
        )

        est, moe = acs(
            us.states.WI,
            "county",
            2023,
            tables=[ACSTotPopTableInfo()],
            api_key="k",
        )

        assert list(est.index) == ["55001", "55003"]
        assert est["total_pop_acs5_23"].tolist() == [100.0, 200.0]
        assert "total_pop_moe_acs5_23" in moe.columns

    def test_sentinels_and_nulls_propagate_through_condense(self, mock_http: MockHTTP):
        table = ACSTotPopTableInfo()
        estimate = next(iter(table.construct_long_names(suffix="E")))
        margin = next(iter(table.construct_long_names(suffix="M")))
        mock_http.route(
            url_contains="/acs/",
            json=[
                ["GEO_ID", estimate, margin],
                ["0500000US55001", "-555555555", "-555555555"],
                ["0500000US55003", "-666666666", "-666666666"],
                ["0500000US55005", None, None],
            ],
        )

        est, moe = acs(us.states.WI, "county", 2023, tables=[table], api_key="k")

        assert pd.isna(est.loc["55001", "total_pop_acs5_23"])
        assert moe.loc["55001", "total_pop_moe_acs5_23"] == 0.0
        assert est.loc[["55003", "55005"], "total_pop_acs5_23"].isna().all()
        assert moe.loc[["55003", "55005"], "total_pop_moe_acs5_23"].isna().all()

    def test_default_tables_concatenate_pop_vap_cvap(self, mock_http: MockHTTP):
        tables = [ACSTotPopTableInfo(), ACSVAPTableInfo(), ACSCVAPTableInfo()]
        mock_http.route(url_contains="/acs/", json=acs_api_payload(tables, {"55001": 1}))

        est, _ = acs(us.states.WI, "county", 2020, api_key="k")

        # POP has 1 source variable, VAP sums 2 indices, CVAP sums 4.
        assert est.loc["55001", "total_pop_acs5_20"] == 1.0
        assert est.loc["55001", "total_vap_acs5_20"] == 2.0
        assert est.loc["55001", "total_cvap_acs5_20"] == 4.0

    def test_tables_align_rows_by_geoid_when_api_order_changes(self, mock_http: MockHTTP):
        tables: list[ACSTableInfo] = [ACSTotPopTableInfo(), ACSVAPTableInfo()]
        payload_values: list[dict[str, object]] = [
            {"55001": 1, "55003": 3},
            {"55001": 1, "55003": 3},
            {"55003": 30, "55001": 10},
            {"55003": 30, "55001": 10},
        ]
        responses = iter(payload_values)

        def responder(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=acs_api_payload(tables, next(responses)))

        mock_http.route(url_contains="/acs/", responder=responder)

        est, _ = acs(us.states.WI, "county", 2023, tables=tables, api_key="k")

        assert est.loc["55001", "total_pop_acs5_23"] == 1.0
        assert est.loc["55003", "total_pop_acs5_23"] == 3.0
        assert est.loc["55001", "total_vap_acs5_23"] == 20.0
        assert est.loc["55003", "total_vap_acs5_23"] == 60.0

    def test_hispanic_by_race_uses_gerrydb_semantic_names(self, mock_http: MockHTTP):
        table = ACSHispByRaceTableInfo()
        mock_http.route(url_contains="/acs/", json=acs_api_payload([table], {"55001": 1}))

        est, _ = acs(us.states.WI, "county", 2023, tables=[table], api_key="k")

        assert est.loc["55001", "non_hispanic_white_pop_acs5_23"] == 1.0
        assert est.loc["55001", "hispanic_pop_acs5_23"] == 1.0

    def test_race_population_uses_semantic_names(self, mock_http: MockHTTP):
        table = ACSRacePopTableInfo()
        mock_http.route(url_contains="/acs/", json=acs_api_payload([table], {"55001": 1}))

        est, _ = acs(us.states.WI, "county", 2023, tables=[table], api_key="k")

        assert est.loc["55001", "black_pop_acs5_23"] == 1.0
        assert est.loc["55001", "two_or_more_races_pop_acs5_23"] == 1.0

    def test_age_table_returns_aggregates_and_named_sex_bands(self, mock_http: MockHTTP):
        table = ACSAgeTableInfo()
        mock_http.route(url_contains="/acs/", json=acs_api_payload([table], {"55001": 1}))

        est, moe = acs(us.states.WI, "county", 2023, tables=[table], api_key="k")

        assert est.loc["55001", "female_25_to_29_pop_acs5_23"] == 1.0
        assert est.loc["55001", "male_85_plus_pop_acs5_23"] == 1.0
        assert est.loc["55001", "under_18_pop_acs5_23"] == 8.0
        assert est.loc["55001", "18_to_64_pop_acs5_23"] == 26.0
        assert est.loc["55001", "65_plus_pop_acs5_23"] == 12.0
        assert moe.loc["55001", "under_18_pop_moe_acs5_23"] == pytest.approx(8**0.5)

    def test_acs1_tract_is_rejected(self, mock_http: MockHTTP):
        with pytest.raises(ValueError, match="not available for 'tract'"):
            acs(
                us.states.WI,
                "tract",
                2020,
                tables=[ACSTotPopTableInfo()],
                survey="acs1",
                api_key="k",
            )

    def test_acs1_partial_county_coverage_warns(self, mock_http: MockHTTP):
        # ACS1 returns one county; ACS5 (completeness check) knows of two.
        mock_http.route(
            url_contains="/acs/acs1",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 100}),
        )
        mock_http.route(url_contains="/acs/acs5", json=geo_id_payload(["55001", "55003"]))

        with pytest.warns(UserWarning, match="ACS 1-year returned 1 of 2"):
            acs(
                us.states.WI,
                "county",
                2020,
                tables=[ACSTotPopTableInfo()],
                survey="acs1",
                api_key="k",
            )

    def test_acs1_complete_county_coverage_does_not_warn(self, mock_http: MockHTTP):
        # ACS1 and ACS5 agree on the county set, so no warning should fire.
        mock_http.route(
            url_contains="/acs/acs1",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 100}),
        )
        mock_http.route(url_contains="/acs/acs5", json=geo_id_payload(["55001"]))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            acs(
                us.states.WI,
                "county",
                2020,
                tables=[ACSTotPopTableInfo()],
                survey="acs1",
                api_key="k",
            )

        assert not any("ACS 1-year returned" in str(w.message) for w in caught)

    @pytest.mark.parametrize("probe_status", [500, 429])
    def test_acs1_completeness_check_failure_is_silent(self, mock_http: MockHTTP, probe_status):
        # The ACS5 completeness probe fails (server error or rate limit); the ACS1 result is
        # still returned and no warning is raised.
        mock_http.route(
            url_contains="/acs/acs1",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 100}),
        )
        mock_http.route(url_contains="/acs/acs5", status_code=probe_status, text="boom")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est, _ = acs(
                us.states.WI,
                "county",
                2020,
                tables=[ACSTotPopTableInfo()],
                survey="acs1",
                api_key="k",
            )

        assert list(est.index) == ["55001"]
        assert not any("ACS 1-year returned" in str(w.message) for w in caught)

    @pytest.mark.parametrize(
        "probe_payload",
        [
            # A 200 whose body is not a list of rows makes _census_get raise ValueError.
            {"error": "unknown variable"},
            # A row payload without GEO_ID yields no GEOID column, so the probe KeyErrors.
            [["NAME"], ["Adams County, Wisconsin"]],
        ],
        ids=["non-list-payload", "missing-geoid-column"],
    )
    def test_acs1_completeness_check_junk_probe_response_is_silent(
        self, mock_http: MockHTTP, probe_payload
    ):
        # The advisory ACS5 probe returning junk must not kill the already-successful ACS1 fetch.
        mock_http.route(
            url_contains="/acs/acs1",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 100}),
        )
        mock_http.route(url_contains="/acs/acs5", json=probe_payload)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est, _ = acs(
                us.states.WI,
                "county",
                2020,
                tables=[ACSTotPopTableInfo()],
                survey="acs1",
                api_key="k",
            )

        assert list(est.index) == ["55001"]
        assert not any("ACS 1-year returned" in str(w.message) for w in caught)

    def test_acs1_non_county_skips_completeness_check(self, mock_http: MockHTTP):
        # State-level ACS1 should not issue the county-only ACS5 probe.
        mock_http.route(
            url_contains="/acs/acs1",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55": 100}),
        )

        acs(
            us.states.WI,
            "state",
            2020,
            tables=[ACSTotPopTableInfo()],
            survey="acs1",
            api_key="k",
        )

        assert all("/acs/acs5" not in url for url in mock_http.urls)


class TestAcsFull:
    def test_renames_to_long_english_names(self, mock_http: MockHTTP):
        mock_http.route(
            url_contains="/acs/",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 7}),
        )

        est, moe = acs_full(us.states.WI, "county", 2020, ACSTotPopTableInfo(), api_key="k")

        assert "total_pop_est_acs5_20" in est.columns
        assert "total_pop_moe_acs5_20" in moe.columns
        assert est.loc["55001", "total_pop_est_acs5_20"] == 7.0

    def test_acs1_partial_county_coverage_warns(self, mock_http: MockHTTP):
        # acs_full runs the same ACS1 completeness probe as acs().
        mock_http.route(
            url_contains="/acs/acs1",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 100}),
        )
        mock_http.route(url_contains="/acs/acs5", json=geo_id_payload(["55001", "55003"]))

        with pytest.warns(UserWarning, match="ACS 1-year returned 1 of 2"):
            acs_full(
                us.states.WI,
                "county",
                2020,
                ACSTotPopTableInfo(),
                survey="acs1",
                api_key="k",
            )


class TestVariableLimitGuard:
    def test_more_than_fifty_variables_raises(self):
        import httpx

        # 50 indices + GEO_ID = 51 requested variables, one over the API cap.
        big_table = ACSTableInfo(
            table_name="big",
            base_table_strings=("B99999",),
            table_indices=tuple(range(1, 51)),
            groups_tup=("",),
        )
        with httpx.Client() as client:
            with pytest.raises(ValueError, match="at most 50 variables"):
                _get_acs_data(client, us.states.WI, "county", 2023, big_table, api_key="k")

    def test_forty_nine_variables_pass_the_guard(self, mock_http: MockHTTP):
        # 49 indices + GEO_ID = 50, exactly at the cap: the request goes through. The age table
        # sits at this fencepost.
        table = ACSTableInfo(
            table_name="big",
            base_table_strings=("B99999",),
            table_indices=tuple(range(1, 50)),
            groups_tup=("",),
        )
        mock_http.route(url_contains="/acs/", json=acs_api_payload([table], {"55001": 1}))

        with httpx.Client() as client:
            result = _get_acs_data(client, us.states.WI, "county", 2023, table, api_key="k")

        assert len(mock_http.requests) == 1
        assert result.shape == (1, 49)
        assert len(ACSAgeTableInfo().construct_long_names(suffix="E")) == 49


class TestAcsValidation:
    def test_invalid_year_raises(self):
        with pytest.raises(ValueError, match="4-digit integer"):
            acs(us.states.WI, "county", 1999, api_key="k")

    def test_empty_tables_raises(self):
        with pytest.raises(ValueError, match="at least one table"):
            acs(us.states.WI, "county", 2020, tables=[], api_key="k")

    def test_non_table_element_raises_type_error(self):
        with pytest.raises(TypeError, match="must be an ACSTableInfo"):
            acs(
                us.states.WI,
                "county",
                2020,
                tables=cast(list[ACSTableInfo], ["not a table"]),
                api_key="k",
            )

    def test_duplicate_condensed_columns_raise_before_fetch(self, mock_http: MockHTTP):
        with pytest.raises(ValueError) as error:
            acs(
                us.states.WI,
                "county",
                2020,
                tables=[ACSTotPopTableInfo(), ACSRacePopTableInfo()],
                api_key="k",
            )

        assert "total_pop" in str(error.value)
        assert "'pop'" in str(error.value)
        assert "'race_pop'" in str(error.value)
        assert not mock_http.requests


# ==================
# == CVAP WRAPPER ==
# ==================


class TestCvap:
    def test_returns_condensed_cvap(self, mock_http: MockHTTP):
        mock_http.route(
            url_contains="/acs/",
            json=acs_api_payload([ACSCVAPTableInfo()], {"55001": 1}),
        )

        est, moe = cvap(us.states.WI, "county", 2020, api_key="k")

        assert est.loc["55001", "total_cvap_acs5_20"] == 4.0
        assert "total_cvap_moe_acs5_20" in moe.columns
