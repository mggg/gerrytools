import pandas as pd
import pytest
from frozendict import frozendict

from gerrytools.data.uscensus.census_tables import (
    PL_TABLES_BY_YEAR,
    ACSAgeTableInfo,
    ACSCVAPTableInfo,
    ACSHispByRaceTableInfo,
    ACSNamedTableInfo,
    ACSRacePopTableInfo,
    ACSTableInfo,
    ACSTotPopTableInfo,
    ACSVAPTableInfo,
    PLBlockVAPTableInfo,
    PLTableInfo,
    census_column_name,
    format_acs_column_names,
    pl_table,
)
from gerrytools.data.uscensus.ptable_column_aliases import COLUMN_ALIASES_PTABLES

# ==================================
# ——————— PUBLIC COLUMN NAMES ——————
# ==================================


class TestCensusColumnName:
    def test_appends_normalized_source_and_two_digit_vintage(self):
        assert census_column_name("TOTAL_POP", source="ACS5", year=2023) == "total_pop_acs5_23"

    def test_decennial_name_can_carry_only_vintage(self):
        assert census_column_name("total_vap", year=2020) == "total_vap_20"


class TestFormatAcsColumnNames:
    def test_renames_in_place_and_appends_source(self):
        # Condensed frames now carry final base names directly (no _est round trip).
        frame = pd.DataFrame(columns=pd.Index(["total_vap", "white_cvap_moe"]))

        format_acs_column_names(frame, source="acs5", year=2023)

        assert list(frame.columns) == ["total_vap_acs5_23", "white_cvap_moe_acs5_23"]


# ==================================
# == ACS LONG NAMES + CONDENSE    ==
# ==================================


class TestACSLongNames:
    def test_unknown_group_has_curated_error(self):
        with pytest.raises(ValueError, match="groups_tup contains unknown.*Z"):
            ACSTableInfo(groups_tup=("Z",))

    def test_vap_total_group_male_estimate(self):
        names = ACSVAPTableInfo().construct_long_names(suffix="E")
        assert names["B05003_008E"] == "total_vap_est_male"

    def test_vap_white_alone_group(self):
        names = ACSVAPTableInfo().construct_long_names(suffix="E")
        assert names["B05003A_008E"] == "white_vap_est_male"

    def test_year_and_source_suffix_appended(self):
        names = ACSVAPTableInfo().construct_long_names(suffix="E", year=2022, source_suffix="acs1")
        assert names["B05003_008E"] == "total_vap_est_male_acs1_22"

    def test_totpop_index_with_empty_name_omits_suffix(self):
        names = ACSTotPopTableInfo().construct_long_names(suffix="E")
        assert names["B01001_001E"] == "total_pop_est"

    def test_moe_suffix_changes_abbreviation(self):
        names = ACSVAPTableInfo().construct_long_names(suffix="M")
        assert names["B05003_008M"] == "total_vap_moe_male"

    def test_age_cells_get_descriptive_full_names(self):
        names = ACSAgeTableInfo().construct_long_names(suffix="E", year=2023, source_suffix="acs5")
        assert names["B01001_035E"] == "female_25_to_29_pop_est_acs5_23"

    def test_unnamed_indices_fall_back_to_index_numbers(self):
        # A custom table whose indices have no shortener entries must not collapse every
        # variable onto one column name; unnamed indices keep the zero-padded index instead.
        custom = ACSTableInfo(
            table_name="commute",
            base_table_strings=("B08301",),
            table_indices=(1, 2),
            groups_tup=("",),
            index_to_name_dict=frozendict(),
        )
        names = custom.construct_long_names(suffix="E", year=2023, source_suffix="acs5")
        assert names["B08301_001E"] == "total_commute_est_001_acs5_23"
        assert names["B08301_002E"] == "total_commute_est_002_acs5_23"


class TestCondenseGroupDict:
    def test_vap_groups_sum_both_indices(self):
        groups = ACSVAPTableInfo().condense_group_dict
        assert groups["total_vap"] == ("B05003_008", "B05003_019")
        assert groups["white_vap"] == ("B05003A_008", "B05003A_019")

    def test_totpop_single_variable(self):
        groups = ACSTotPopTableInfo().condense_group_dict
        assert groups["total_pop"] == ("B01001_001",)

    def test_cvap_sums_four_citizen_indices(self):
        groups = ACSCVAPTableInfo().condense_group_dict
        assert groups["total_cvap"] == (
            "B05003_009",
            "B05003_011",
            "B05003_020",
            "B05003_022",
        )

    def test_hisp_by_race_override_uses_per_index_keys(self):
        groups = ACSHispByRaceTableInfo().condense_group_dict
        # The override maps each per-index name to a single source variable.
        assert groups["non_hispanic_white_pop"] == ("B03002_003",)
        assert groups["hispanic_white_pop"] == ("B03002_013",)

    def test_race_population_uses_semantic_names(self):
        groups = ACSRacePopTableInfo().condense_group_dict
        assert groups["amin_pop"] == ("B02001_004",)
        assert groups["two_or_more_races_pop"] == ("B02001_008",)

    def test_named_table_indices_must_match_named_variables(self):
        # Regression: table_indices could silently disagree with the variables a named
        # table actually queries; index_to_name_dict is now the single source of truth.
        with pytest.raises(ValueError, match="table_indices"):
            ACSNamedTableInfo(
                table_name="race_pop",
                base_table_strings=("B02001",),
                table_indices=(1, 2, 3),
                index_to_name_dict=frozendict({1: "total_pop", 2: "white_pop"}),
            )

    def test_named_table_rejects_group_suffixes(self):
        with pytest.raises(ValueError, match="groups_tup"):
            ACSNamedTableInfo(
                table_name="race_pop",
                base_table_strings=("B02001",),
                groups_tup=("A",),
                index_to_name_dict=frozendict({1: "total_pop"}),
            )

    def test_named_table_derives_indices_and_groups_from_names(self):
        info = ACSNamedTableInfo(
            table_name="race_pop",
            base_table_strings=("B02001",),
            index_to_name_dict=frozendict({2: "white_pop", 1: "total_pop"}),
        )
        assert info.table_indices == (1, 2)
        assert info.groups_tup == ("",)

    def test_named_table_coherent_construction_matches_concrete_table(self):
        concrete = ACSRacePopTableInfo()
        rebuilt = ACSNamedTableInfo(
            table_name=concrete.table_name,
            base_table_strings=concrete.base_table_strings,
            table_indices=concrete.table_indices,
            index_to_name_dict=concrete.index_to_name_dict,
        )
        assert rebuilt.construct_long_names(
            suffix="E", year=2023, source_suffix="acs5"
        ) == concrete.construct_long_names(suffix="E", year=2023, source_suffix="acs5")
        assert rebuilt.condense_group_dict == concrete.condense_group_dict

    def test_multiple_base_tables_are_rejected(self):
        # The condensation model keys output groups without the base table, so two base tables
        # would collide and only the last would survive; reject upfront instead.
        with pytest.raises(ValueError, match="one base table"):
            ACSTableInfo(
                table_name="pop",
                base_table_strings=("B01001", "B02001"),
                table_indices=(1,),
                groups_tup=("",),
            )

    def test_named_table_also_rejects_multiple_base_tables(self):
        with pytest.raises(ValueError, match="one base table"):
            ACSNamedTableInfo(
                table_name="race_pop",
                base_table_strings=("B02001", "B03002"),
                index_to_name_dict=frozendict({1: "total_pop"}),
            )

    def test_age_table_has_sex_cells_and_combined_bands(self):
        groups = ACSAgeTableInfo().condense_group_dict
        assert groups["male_under_5_pop"] == ("B01001_003",)
        assert groups["female_25_to_29_pop"] == ("B01001_035",)
        assert groups["under_18_pop"] == (
            "B01001_003",
            "B01001_004",
            "B01001_005",
            "B01001_006",
            "B01001_027",
            "B01001_028",
            "B01001_029",
            "B01001_030",
        )


# ==================================
# == DECENNIAL PL TABLES          ==
# ==================================


class TestPlTable:
    @pytest.mark.parametrize(
        "year,table_name",
        [
            (year, table_name)
            for year, table_names in PL_TABLES_BY_YEAR.items()
            for table_name in table_names
        ],
    )
    def test_builds_for_every_supported_table_and_year(self, year, table_name):
        table = pl_table(table_name, year)
        assert table.table_name == f"{table_name}_{year}"
        assert table.year == year
        assert len(table.construct_variable_names()) > 0

    def test_hand_built_table_requires_a_year(self):
        # Raw PL variable names are vintage-specific, so a table without a declared vintage
        # cannot be checked against the year a getter queries.
        with pytest.raises(ValueError, match="requires a year"):
            PLTableInfo(table_name="custom")

    def test_rejects_unknown_year(self):
        with pytest.raises(ValueError, match="PL year"):
            pl_table("P1", 1999)

    def test_rejects_unknown_table(self):
        with pytest.raises(ValueError, match="PL table"):
            pl_table("P9", 2020)

    def test_rejects_p5_for_2010(self):
        with pytest.raises(ValueError, match="PL table for 2010"):
            pl_table("P5", 2010)

    @pytest.mark.parametrize(
        "year,variable,expected",
        [
            (2010, "P001001", "total_pop_10"),
            (2020, "P1_001N", "total_pop_20"),
        ],
    )
    def test_getter_vintage_replaces_alias_vintage(self, year, variable, expected):
        assert pl_table("P1", year).construct_rename_map(year=year)[variable] == expected

    def test_2010_aliases_use_only_unpadded_api_spellings(self):
        # The 2010 PL API serves only unpadded variable names (P001001); the padded spellings
        # (P0010001) do not exist and a request for them would 400.
        table = pl_table("P1", 2010)
        mapping = table.variable_to_short_name
        assert mapping["P001001"] == "total_pop"
        assert mapping["P001009"] == "two_or_more_races_pop"
        assert mapping["P001071"] == "white_black_amin_asian_nhpi_other_pop"
        assert "P0010001" not in mapping

        # construct_variable_names is what the PL getters put on the wire.
        variables = table.construct_variable_names()
        assert all(len(variable) == 7 for variable in variables)

    def test_2010_h1_keys(self):
        assert set(pl_table("H1", 2010).variable_to_short_name) == {
            "H001001",
            "H001002",
            "H001003",
        }

    def test_block_vap_mapping_follows_selected_race_categories(self):
        table = PLBlockVAPTableInfo(race_categories=("total", "white"))

        assert set(table.variable_to_short_name.values()) == {"total_vap", "white_vap"}

    def test_housing_and_group_quarters_names(self):
        assert pl_table("H1", 2010).construct_rename_map(year=2010)["H001001"] == (
            "total_housing_units_10"
        )
        assert pl_table("H1", 2020).construct_rename_map(year=2020) == {
            "H1_001N": "total_housing_units_20",
            "H1_002N": "occupied_housing_units_20",
            "H1_003N": "vacant_housing_units_20",
        }
        assert (
            pl_table("P5", 2020).construct_rename_map(year=2020)["P5_003N"]
            == "adult_correctional_facility_pop_20"
        )

    @pytest.mark.parametrize("year", [2010, 2020])
    def test_p3_aliases_are_p1_aliases_for_voting_age_population(self, year):
        aliases = COLUMN_ALIASES_PTABLES[year]
        assert list(aliases["P3"].values()) == [
            name.replace("_pop", "_vap") for name in aliases["P1"].values()
        ]

    @pytest.mark.parametrize("year", [2010, 2020])
    def test_p2_aliases_extend_p1_with_non_hispanic_population(self, year):
        aliases = COLUMN_ALIASES_PTABLES[year]
        assert list(aliases["P2"].values())[2:] == [
            f"non_hispanic_{name.replace('total_', '')}" for name in aliases["P1"].values()
        ]

    @pytest.mark.parametrize("year", [2010, 2020])
    def test_p4_aliases_extend_p3_with_non_hispanic_vap(self, year):
        aliases = COLUMN_ALIASES_PTABLES[year]
        assert list(aliases["P4"].values())[2:] == [
            f"non_hispanic_{name.replace('total_', '')}" for name in aliases["P3"].values()
        ]

    @pytest.mark.parametrize("table_name", ["P1", "P2", "P3", "P4"])
    def test_2010_and_2020_aliases_have_the_same_semantic_order(self, table_name):
        assert list(COLUMN_ALIASES_PTABLES[2010][table_name].values()) == list(
            COLUMN_ALIASES_PTABLES[2020][table_name].values()
        )


class TestPLBlockVAPTableInfo:
    def test_defaults_to_the_2020_vintage(self):
        # The derived P3/P4 variable names use the 2020 aliases.
        assert PLBlockVAPTableInfo().year == 2020

    def test_unknown_race_category_has_curated_error(self):
        with pytest.raises(ValueError, match="race_categories contains unknown.*klingon"):
            PLBlockVAPTableInfo(race_categories=("klingon",))

    def test_short_names_carry_vintage(self):
        short_names = PLBlockVAPTableInfo().construct_short_names(year=2020)
        assert "total_vap_20" in short_names
        assert "hispanic_vap_20" in short_names

    def test_rename_map_maps_raw_to_vintage_name(self):
        rename_map = PLBlockVAPTableInfo().construct_rename_map(year=2020)
        assert rename_map["P3_001N"] == "total_vap_20"
        assert rename_map["P4_002N"] == "hispanic_vap_20"

    @pytest.mark.parametrize(
        ("short", "expected"),
        [
            ("total_vap", "total_vap_20"),
            ("white_vap", "white_vap_20"),
            ("non_hispanic_white_vap", "non_hispanic_white_vap_20"),
            ("hispanic_vap", "hispanic_vap_20"),
        ],
    )
    def test_short_names_compose_with_census_column_name(self, short, expected):
        # The one naming path for composites: census_column_name over the table's short names.
        table = PLBlockVAPTableInfo()
        assert short in table.variable_to_short_name.values()
        assert census_column_name(short, year=2020) == expected


# ==================================
# == PUBLIC EXPORTS               ==
# ==================================

# Table definitions and helpers that acs()/census() document as accepted arguments and that
# should therefore be reachable without importing from the private census_tables module.
PUBLIC_NAMES = (
    "ACSTableInfo",
    "ACSNamedTableInfo",
    "ACSTotPopTableInfo",
    "ACSRacePopTableInfo",
    "ACSAgeTableInfo",
    "ACSVAPTableInfo",
    "ACSCVAPTableInfo",
    "ACSHispByRaceTableInfo",
    "PLTableInfo",
    "PLBlockVAPTableInfo",
    "pl_table",
    "census_column_name",
    "CensusRateLimitError",
)


class TestPublicExports:
    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_reachable_from_data_package(self, name):
        import gerrytools.data as data

        assert name in data.__all__
        assert hasattr(data, name)

    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_reachable_from_uscensus_package(self, name):
        import gerrytools.data.uscensus as uscensus_pkg

        assert name in uscensus_pkg.__all__
        assert hasattr(uscensus_pkg, name)
