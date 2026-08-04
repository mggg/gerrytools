import pandas as pd
import pytest
import us

from gerrytools.data.uscensus.census import census
from gerrytools.data.uscensus.census_tables import pl_table
from tests.data._helpers import MockHTTP, pl_api_payload

# ==================================
# == VALIDATION                   ==
# ==================================


class TestCensusValidation:
    def test_non_4_digit_year_raises(self):
        with pytest.raises(ValueError, match="4-digit integer"):
            census(us.states.WI, year=1999, api_key="k")

    def test_year_without_pl_data_raises(self):
        with pytest.raises(ValueError, match="PL year"):
            census(us.states.WI, year=2000, api_key="k")

    def test_year_without_pl_data_raises_for_table_instance(self):
        with pytest.raises(ValueError, match="only available for years"):
            census(us.states.WI, year=2000, table=pl_table("P1", 2020), api_key="k")

    @pytest.mark.parametrize("table_year,query_year", [(2010, 2020), (2020, 2010)])
    def test_cross_vintage_table_raises(self, table_year, query_year):
        # Regression: a 2010 table queried for 2020 used to pass validation and return a
        # silently empty frame (the raw variable names share zero overlap across vintages).
        with pytest.raises(ValueError, match=f"vintage {table_year}"):
            census(us.states.WI, year=query_year, table=pl_table("P1", table_year), api_key="k")

    def test_unknown_table_string_raises(self):
        with pytest.raises(ValueError, match="PL table"):
            census(us.states.WI, table="P9", api_key="k")

    def test_table_unavailable_for_vintage_raises(self):
        with pytest.raises(ValueError, match="PL table for 2010"):
            census(us.states.WI, year=2010, table="P5", api_key="k")

    def test_unknown_geometry_raises(self):
        with pytest.raises(ValueError, match="Invalid geometry"):
            census(us.states.WI, geometry="precinct", api_key="k")

    def test_multi_group_table_rejected_before_any_request(self):
        from gerrytools.data.uscensus.census_tables import PLBlockVAPTableInfo

        with pytest.raises(ValueError, match="group code"):
            census(us.states.WI, table=PLBlockVAPTableInfo(), api_key="k")


# ==================================
# == FETCH + NUMERIC DTYPE        ==
# ==================================


class TestCensusFetch:
    def test_block_geometry_queries_all_counties_and_tracts(self, mock_http: MockHTTP):
        table = pl_table("P1", 2020)
        mock_http.route(
            url_contains="/dec/pl",
            json=pl_api_payload(
                table,
                {"550019501001000": "1"},
                extra_columns={"state": "55", "county": "001", "tract": "950100"},
            ),
        )

        result = census(us.states.WI, geometry="block", table=table, api_key="k")

        request = mock_http.requests[0]
        assert request.url.params.get_list("in") == ["state:55", "county:*", "tract:*"]
        assert list(result.index) == ["550019501001000"]

    def test_uses_semantic_name_with_vintage_not_table_code(self, mock_http: MockHTTP):
        mock_http.route(
            url_contains="/dec/pl",
            json=pl_api_payload(
                pl_table("P1", 2020),
                {"55": "1"},
                overrides={"P1_001N": "5000"},
                extra_columns={"state": "55"},
            ),
        )

        df = census(us.states.WI, geometry="state", year=2020, table="P1", api_key="k")

        assert df.loc["55", "total_pop_20"] == 5000
        assert not any("p1" in column for column in df.columns)
        assert mock_http.requests[0].url.params["get"] == "group(P1)"

    def test_counts_are_numeric_and_indexed_by_geoid(self, mock_http: MockHTTP):
        # Build a full P1 group response, pinning two real variable->short-name pairs.
        table = pl_table("P1", 2020)
        (raw_one, short_one), (raw_two, short_two) = list(
            table.construct_rename_map(year=2020).items()
        )[:2]

        mock_http.route(
            url_contains="/dec/pl",
            json=pl_api_payload(
                table,
                {"55": "1"},
                overrides={raw_one: "5000", raw_two: "1200"},
                extra_columns={"NAME": "Wisconsin", "state": "55"},
            ),
        )

        df = census(us.states.WI, geometry="state", year=2020, table="P1", api_key="k")

        # Index is the stripped GEOID, GEO_ID is dropped.
        assert df.index.name == "GEOID"
        assert list(df.index) == ["55"]
        assert "GEO_ID" not in df.columns

        # Regression: count columns come back numeric, not strings.
        assert pd.api.types.is_numeric_dtype(df[short_one])
        assert df.loc["55", short_one] == 5000
        assert df.loc["55", short_two] == 1200

        # Only the renamed count columns are returned; NAME and geography-breakdown columns dropped.
        assert list(df.columns) == list(table.construct_short_names(year=2020))
        assert "NAME" not in df.columns
        assert "state" not in df.columns

    def test_header_only_response_returns_empty_frame_with_schema(self, mock_http: MockHTTP):
        mock_http.route(
            url_contains="/dec/pl",
            json=[["GEO_ID", "P1_001N", "state"]],
        )

        table = pl_table("P1", 2020)
        result = census(us.states.WI, geometry="state", year=2020, table=table, api_key="k")

        assert result.empty
        assert result.index.name == "GEOID"
        # The empty frame still carries the table's column schema.
        assert list(result.columns) == list(table.construct_short_names(year=2020))

    def test_2010_request_and_rename_use_unpadded_spellings(self, mock_http: MockHTTP):
        # The 2010 PL API serves only unpadded variable names (P001001, never P0010001).
        mock_http.route(
            url_contains="/dec/pl",
            json=pl_api_payload(
                pl_table("P1", 2010),
                {"55": "1"},
                overrides={"P001001": "5000"},
                extra_columns={"state": "55"},
            ),
        )

        df = census(us.states.WI, geometry="state", year=2010, table="P1", api_key="k")

        assert df.loc["55", "total_pop_10"] == 5000

    def test_all_na_and_err_columns_are_dropped(self, mock_http: MockHTTP):
        # ERR columns and all-NA variable columns are dropped before the numeric cast.
        table = pl_table("P1", 2020)
        mock_http.route(
            url_contains="/dec/pl",
            json=pl_api_payload(
                table,
                {"55": "1"},
                overrides={"P1_001N": "5000", "P1_002N": None},
                extra_columns={"P1_001NERR": "x", "state": "55"},
            ),
        )

        df = census(us.states.WI, geometry="state", year=2020, table="P1", api_key="k")

        # The all-NA P1_002N column vanishes; every other count column survives, ERR-free.
        assert df.loc["55", "total_pop_20"] == 5000
        assert table.construct_rename_map(year=2020)["P1_002N"] not in df.columns
        assert not any(column.lower().endswith("err") for column in df.columns)

    def test_leading_zero_geoids_survive_as_strings(self, mock_http: MockHTTP):
        # Alabama county GEOIDs start with "01"; any integer coercion would strip the zero.
        mock_http.route(
            url_contains="/dec/pl",
            json=pl_api_payload(
                pl_table("P1", 2020),
                {"01001": "5000"},
                extra_columns={"state": "01", "county": "001"},
            ),
        )

        df = census(us.states.AL, geometry="county", year=2020, table="P1", api_key="k")

        assert list(df.index) == ["01001"]
        assert str(df.index.dtype) in {"object", "string"}

    def test_accepts_pltableinfo_instance(self, mock_http: MockHTTP):
        table = pl_table("P1", 2020)
        raw, short = next(iter(table.construct_rename_map(year=2020).items()))
        mock_http.route(
            url_contains="/dec/pl",
            json=pl_api_payload(
                table, {"55": "1"}, overrides={raw: "42"}, extra_columns={"state": "55"}
            ),
        )

        df = census(us.states.WI, geometry="state", year=2020, table=table, api_key="k")

        assert df.loc["55", short] == 42

    def test_missing_renamed_column_raises(self, mock_http: MockHTTP):
        # A drifted API variable spelling (P1_001M instead of P1_001N) must fail loudly instead
        # of silently dropping the renamed column from the output.
        payload = pl_api_payload(pl_table("P1", 2020), {"55": "1"})
        header = payload[0]
        header[header.index("P1_001N")] = "P1_001M"
        mock_http.route(url_contains="/dec/pl", json=payload)

        with pytest.raises(ValueError, match="total_pop_20"):
            census(us.states.WI, geometry="state", year=2020, table="P1", api_key="k")

    @pytest.mark.parametrize(
        "table,variable,expected",
        [
            ("H1", "H1_001N", "total_housing_units_20"),
            ("P5", "P5_003N", "adult_correctional_facility_pop_20"),
        ],
    )
    def test_housing_and_group_quarters_shortcuts(
        self, mock_http: MockHTTP, table, variable, expected
    ):
        mock_http.route(
            url_contains="/dec/pl",
            json=pl_api_payload(
                pl_table(table, 2020),
                {"55": "1"},
                overrides={variable: "42"},
                extra_columns={"state": "55"},
            ),
        )

        df = census(us.states.WI, geometry="state", year=2020, table=table, api_key="k")

        assert df.loc["55", expected] == 42
