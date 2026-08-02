from dataclasses import dataclass, field
from itertools import product
from typing import ClassVar

import pandas as pd
from frozendict import frozendict

from .ptable_column_aliases import COLUMN_ALIASES_PTABLES

# Race categories shared between ACS VAP/CVAP and decennial PL block VAP columns.
RACE_CATEGORIES: tuple[str, ...] = (
    "total",
    "white",
    "black",
    "amin",
    "asian",
    "nhpi",
    "other",
    "two_or_more_races",
    "non_hispanic_white",
    "hispanic",
)

# P3 is authoritative for total VAP, the one semantic alias shared with P4.
_PL_2020_VAP_BY_SHORT = {
    short_name: variable
    for table in ("P4", "P3")
    for variable, short_name in COLUMN_ALIASES_PTABLES[2020][table].items()
}


def census_column_name(
    name: str,
    *,
    source: str | None = None,
    year: int | None = None,
) -> str:
    """Build a lowercase Census column name with optional source and vintage suffixes.

    Args:
        name (str): Semantic base column name.
        source (str | None, optional): Census product, such as ``"acs5"`` or ``"acs1"``.
            Defaults to None, which appends no product suffix.
        year (int | None, optional): Four-digit data vintage. The final two digits are appended.
            Defaults to None, which appends no vintage suffix.

    Returns:
        str: Lowercase column name, such as ``"total_pop_acs5_23"``.

    Examples:
        ``census_column_name("total_pop", source="acs5", year=2023)`` returns
        ``"total_pop_acs5_23"``.
    """

    parts = [name.lower()]
    if source:
        parts.append(source.lower())
    if year is not None:
        parts.append(f"{year % 100:02d}")
    return "_".join(parts)


def format_acs_column_names(
    df: pd.DataFrame,
    source: str,
    year: int,
) -> None:
    """Apply semantic ACS product and vintage suffixes to condensed columns in place.

    Args:
        df (pd.DataFrame): DataFrame whose column names will be rewritten in place.
        source (str): Normalized ACS product, either ``"acs1"`` or ``"acs5"``.
        year (int): Four-digit ACS vintage.

    Warning:
        Modifies ``df`` in place.
    """

    df.rename(
        columns={
            col: census_column_name(col.lower(), source=source, year=year) for col in df.columns
        },
        inplace=True,
    )


PL_YEARS: tuple[int, ...] = (2010, 2020)
PL_TABLES_BY_YEAR: dict[int, tuple[str, ...]] = {
    2010: ("P1", "P2", "P3", "P4", "H1"),
    2020: ("P1", "P2", "P3", "P4", "P5", "H1"),
}


def pl_table(table: str, year: int) -> "PLTableInfo":
    """Build a semantic table definition for a decennial PL table and vintage.

    Args:
        table (str): PL table identifier supported for ``year``.
        year (int): ``2010`` or ``2020``.

    Returns:
        PLTableInfo: Table info mapping raw Census variables to semantic base names.

    Raises:
        ValueError: If the year or table is unsupported.
    """

    if year not in PL_YEARS:
        raise ValueError(f"PL year must be one of {PL_YEARS}; got {year}.")

    available_tables = PL_TABLES_BY_YEAR[year]
    if table not in available_tables:
        raise ValueError(f"PL table for {year} must be one of {available_tables}; got {table!r}.")

    return PLTableInfo(
        table_name=f"{table}_{year}",
        variable_to_short_name=frozendict(COLUMN_ALIASES_PTABLES[year][table]),
        api_table_code=table,
        year=year,
    )


@dataclass(frozen=True)
class PLTableInfo:
    """Base table definition for decennial Public Law 94-171 (PL) Census API tables.

    Unlike ACS table definitions, PL variables do not use estimate/MOE suffixes or grouped variants.
    A table definition maps Census variables to semantic base names; a data getter adds its queried
    vintage to public column names.

    Attributes:
        table_name (str): Human-readable name for the logical PL table. Defaults to ``""``.
        variable_to_short_name (frozendict): Mapping from a raw Census variable to its semantic
            base column name. Defaults to an empty mapping.
        api_table_code (str | None): The Census API group code this table can be fetched with via
            ``get=group(...)`` (e.g. ``"P1"``). None for tables that combine multiple API groups
            and therefore cannot be fetched with a single group request.
        year (int | None): Decennial PL vintage whose raw variable names this table carries.
            Required: raw variable spellings differ across vintages (2010 ``P001001`` vs 2020
            ``P1_001N``), so getters use it to reject cross-vintage requests upfront. Defaults to
            None, which is rejected unless a subclass supplies a preset.

    Raises:
        ValueError: If no decennial vintage is supplied.
    """

    table_name: str = ""
    variable_to_short_name: frozendict = field(default_factory=frozendict)
    api_table_code: str | None = None
    year: int | None = None

    def __post_init__(self) -> None:
        if self.year is None:
            raise ValueError(
                "PLTableInfo requires a year: raw PL variable names are vintage-specific. "
                "Build tables via pl_table(table, year) or pass year= explicitly."
            )

    def construct_variable_names(self) -> tuple[str, ...]:
        """Return the raw Census API variable names for this PL table.

        Returns:
            tuple[str, ...]: Census API variable names in definition order.
        """

        return tuple(self.variable_to_short_name.keys())

    def construct_short_names(self, year: int | None = None) -> tuple[str, ...]:
        """Return semantic local column names, optionally suffixed by vintage.

        Args:
            year (int | None, optional): Census vintage appended to names. Defaults to None.

        Returns:
            tuple[str, ...]: Local column names paired 1:1 with ``construct_variable_names``.
        """

        return tuple(self.construct_rename_map(year=year).values())

    def construct_rename_map(self, year: int | None = None) -> dict[str, str]:
        """Return the raw-variable-to-semantic-column rename mapping.

        Args:
            year (int | None, optional): Census vintage appended to names. Defaults to None.

        Returns:
            dict[str, str]: Mapping suitable for ``DataFrame.rename``.
        """

        return {
            variable: census_column_name(short_name, year=year)
            for variable, short_name in self.variable_to_short_name.items()
        }

    def rename_columns(self, df: pd.DataFrame, year: int | None = None) -> None:
        """Rename raw Census API variables in ``df`` to local short names.

        Args:
            df (pd.DataFrame): DataFrame whose columns will be renamed in place.
            year (int | None, optional): Census vintage appended to semantic names.
                Defaults to None.

        Warning:
            Modifies ``df`` in place.
        """

        df.rename(columns=self.construct_rename_map(year=year), inplace=True)


@dataclass(frozen=True)
class PLBlockVAPTableInfo(PLTableInfo):
    """Decennial PL block-level voting-age population (VAP) table definition.

    Wraps Census PL tables P3 (VAP by race) and P4 (VAP by Hispanic origin) at block geography.
    Its semantic names are shared with the ACS VAP/CVAP tables so the estimator can pair them.

    Attributes:
        table_name (str): ``"PLBlockVAP"`` by default.
        year (int | None): ``2020`` by default; the derived P3/P4 mapping uses 2020 spellings.
        race_categories (tuple[str, ...]): Race categories this table covers. Defaults to
            ``RACE_CATEGORIES``.
        variable_to_short_name (frozendict): Maps the derived P3/P4 variables to their semantic
            VAP names.

    Raises:
        ValueError: If a race category is unknown.
    """

    table_name: str = "PLBlockVAP"
    year: int | None = 2020
    race_categories: tuple[str, ...] = RACE_CATEGORIES
    variable_to_short_name: frozendict = field(default_factory=frozendict, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        unknown = [category for category in self.race_categories if category not in RACE_CATEGORIES]
        if unknown:
            raise ValueError(
                f"race_categories contains unknown values {unknown!r}; "
                f"expected values from {RACE_CATEGORIES!r}."
            )
        object.__setattr__(
            self,
            "variable_to_short_name",
            frozendict(
                {
                    _PL_2020_VAP_BY_SHORT[f"{category}_vap"]: f"{category}_vap"
                    for category in self.race_categories
                }
            ),
        )


@dataclass(frozen=True)
class ACSTableInfo:
    """Base class for American Community Survey (ACS) table definitions.

    Subclasses specify the Census base tables, indices, and demographic groups that make up one
    logical "table" (e.g. VAP, CVAP). The base class derives both the long
    Census-variable-to-descriptive-name mapping and the ``condense_group_dict`` (raw variable
    groupings to sum) from those fields.

    Subclasses override ``condense_group_dict`` only when the standard "group letter × indices →
    per-race sum" derivation does not fit, as is the case for ``ACSHispByRaceTableInfo``.

    Attributes:
        table_name (str): Semantic measure name for the logical ACS table. Defaults to the
            concrete table class's preset, or ``""`` on this base class.
        base_table_strings (tuple[str, ...]): Census base table code used (e.g. ``("B05003",)``).
            At most one entry: condensed output names do not carry the base table, so a second
            base table would collide with the first. Defaults to the concrete class's preset, or
            an empty tuple on this base class.
        table_indices (tuple[int, ...]): Numeric indices within each base table that correspond to
            the variables this table cares about. Defaults to the concrete class's preset, or an
            empty tuple on this base class.
        groups_tup (tuple[str, ...]): Single-letter Census group suffixes (e.g. ``""`` for all
            races, ``"A"`` for White alone) that this table requests. Defaults to the concrete
            class's preset, or an empty tuple on this base class.
        index_to_name_dict (frozendict): Mapping from a Census index to the descriptive suffix
            appended to the long column name. Defaults to the concrete class's preset, or an empty
            mapping on this base class.
        table_to_group_dict (frozendict): Class attribute; semantic name for each Census
            group-suffix letter.
        suffix_to_abbrev_dict (frozendict): Class attribute; Census variable suffixes
            (``"E"``, ``"M"``) mapped to their short forms.

    Raises:
        ValueError: If table codes, indices, groups, or semantic names are invalid.
    """

    table_name: str = ""
    base_table_strings: tuple[str, ...] = ()
    table_indices: tuple[int, ...] = ()
    groups_tup: tuple[str, ...] = ()
    index_to_name_dict: frozendict = field(default_factory=frozendict)

    table_to_group_dict: ClassVar[frozendict] = frozendict(
        {
            "": "total",
            "A": "white",
            "B": "black",
            "C": "amin",
            "D": "asian",
            "E": "nhpi",
            "F": "other",
            "G": "two_or_more_races",
            "H": "non_hispanic_white",
            "I": "hispanic",
        }
    )

    suffix_to_abbrev_dict: ClassVar[frozendict] = frozendict(
        {
            "E": "est",
            "M": "moe",
        }
    )

    def __post_init__(self) -> None:
        if len(self.base_table_strings) > 1:
            raise ValueError(
                f"ACSTableInfo supports one base table per info object; got "
                f"{self.base_table_strings}. Condensed output names do not carry the base "
                "table, so multiple base tables would collide on the same column names. "
                "Build one info object per base table instead."
            )
        unknown_groups = [
            group for group in self.groups_tup if group not in self.table_to_group_dict
        ]
        if unknown_groups:
            raise ValueError(
                f"groups_tup contains unknown values {unknown_groups!r}; "
                f"expected values from {tuple(self.table_to_group_dict)!r}."
            )

    def construct_long_names(
        self,
        suffix: str = "E",
        year: int | None = None,
        source_suffix: str | None = None,
    ) -> dict[str, str]:
        """Construct descriptive ACS column names for this table definition.

        Args:
            suffix (str): Census variable suffix: ``"E"`` for estimates, ``"M"`` for margins of
                error. Defaults to ``"E"``.
            year (int | None): ACS year to append to the descriptive column names. When omitted,
                generated names do not include a year suffix.
            source_suffix (str | None): ACS product appended to the descriptive column name, such
                as ``"acs5"`` or ``"acs1"``. When omitted, no source suffix is appended.

        Returns:
            dict[str, str]: Mapping from Census API variables to descriptive column names.
        """

        long_name_dict = {}

        for table, group, index in product(
            self.base_table_strings, self.groups_tup, self.table_indices
        ):
            variable = f"{table}{group}_{index:03d}{suffix}"
            parts = [
                self.table_to_group_dict[group],
                self.table_name,
                self.suffix_to_abbrev_dict[suffix],
            ]
            index_name = self.index_to_name_dict.get(index, "")
            # An unnamed index is fine for a single-variable group (the group name says it all),
            # but with several indices every variable would collapse onto the same column name;
            # fall back to the zero-padded index so the names stay unique and traceable.
            if not index_name and len(self.table_indices) > 1:
                index_name = f"{index:03d}"
            if index_name:
                parts.append(index_name)
            long_name_dict[variable] = census_column_name(
                "_".join(parts),
                source=source_suffix,
                year=year,
            )

        return long_name_dict

    @property
    def condense_group_dict(self) -> frozendict:
        """Output group name to raw Census variable names to sum into it.

        Default behaviour: for each Census base table × group-suffix letter, produce one output
        group ``{long_group_name}_{table_name}`` that sums all ``table_indices`` for that (base
        table, group) pair. Subclasses whose structure does not fit (e.g.
        ``ACSHispByRaceTableInfo``, which uses the per-index name as the output key) override this
        property.

        Returns:
            frozendict: Mapping from output group name to a tuple of raw Census variable names
            (without estimate/MOE suffix) whose values should be summed.
        """

        result = {}
        for base_table in self.base_table_strings:
            for group in self.groups_tup:
                key = f"{self.table_to_group_dict[group]}_{self.table_name}"
                result[key] = tuple(
                    f"{base_table}{group}_{index:03d}" for index in self.table_indices
                )
        return frozendict(result)


@dataclass(frozen=True)
class ACSNamedTableInfo(ACSTableInfo):
    """ACS table whose variables each map directly to one semantic output name.

    ``index_to_name_dict`` is the source of truth: ``table_indices`` is derived from (or
    validated against) its keys, and ``groups_tup`` is pinned to the ungrouped spelling
    ``("",)``, since named tables query only plain ``{table}_{index}`` variables.

    Raises:
        ValueError: If configured indices or groups conflict with ``index_to_name_dict``.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        derived_indices = tuple(sorted(self.index_to_name_dict))
        if self.table_indices and tuple(sorted(self.table_indices)) != derived_indices:
            raise ValueError(
                f"table_indices {self.table_indices} disagree with the index_to_name_dict "
                f"keys {derived_indices}; a named table queries exactly the named indices."
            )
        object.__setattr__(self, "table_indices", derived_indices)
        if self.groups_tup not in ((), ("",)):
            raise ValueError(
                f"groups_tup {self.groups_tup} is invalid for a named table; "
                "index_to_name_dict names ungrouped variables only, so groups_tup "
                'must be left unset or ("",).'
            )
        object.__setattr__(self, "groups_tup", ("",))

    def construct_long_names(
        self,
        suffix: str = "E",
        year: int | None = None,
        source_suffix: str | None = None,
    ) -> dict[str, str]:
        """Map raw ACS variables to named statistic columns.

        Args:
            suffix (str, optional): ACS statistic suffix, normally ``"E"`` or ``"M"``. Defaults
                to ``"E"``.
            year (int | None, optional): Census vintage appended to semantic names. Defaults to
                None, which appends no vintage.
            source_suffix (str | None, optional): Source token appended before the year. Defaults
                to None, which appends no source token.

        Returns:
            dict[str, str]: Raw ACS variable names mapped to semantic output names.
        """
        statistic = self.suffix_to_abbrev_dict[suffix]
        return {
            f"{table}_{index:03d}{suffix}": census_column_name(
                f"{name}_{statistic}",
                source=source_suffix,
                year=year,
            )
            for table in self.base_table_strings
            for index, name in self.index_to_name_dict.items()
        }

    @property
    def condense_group_dict(self) -> frozendict:
        """Map each semantic name to its single raw ACS variable."""
        return frozendict(
            {
                name: (f"{table}_{index:03d}",)
                for table in self.base_table_strings
                for index, name in self.index_to_name_dict.items()
            }
        )


@dataclass(frozen=True)
class ACSCVAPTableInfo(ACSTableInfo):
    """ACS Citizen Voting-Age Population (CVAP) table definition.

    Sources variables from Census base table B05003 and sums male/female native and naturalized
    citizen counts (indices 9, 11, 20, 22) into per-race CVAP totals.
    """

    table_name: str = "cvap"
    base_table_strings: tuple[str, ...] = ("B05003",)
    table_indices: tuple[int, ...] = (9, 11, 20, 22)
    groups_tup: tuple[str, ...] = ("", "A", "B", "C", "D", "E", "F", "G", "H", "I")
    index_to_name_dict: frozendict = field(
        default_factory=lambda: frozendict(
            {
                9: "male_native",
                11: "male_naturalized",
                20: "female_native",
                22: "female_naturalized",
            }
        )
    )


@dataclass(frozen=True)
class ACSVAPTableInfo(ACSTableInfo):
    """ACS Voting-Age Population (VAP) table definition.

    Sources variables from Census base table B05003 and sums the male/female voting-age subtotals
    (indices 8 and 19) into per-race VAP totals.
    """

    table_name: str = "vap"
    base_table_strings: tuple[str, ...] = ("B05003",)
    table_indices: tuple[int, ...] = (8, 19)
    groups_tup: tuple[str, ...] = ("", "A", "B", "C", "D", "E", "F", "G", "H", "I")
    index_to_name_dict: frozendict = field(
        default_factory=lambda: frozendict(
            {
                8: "male",
                19: "female",
            }
        )
    )


@dataclass(frozen=True)
class ACSTotPopTableInfo(ACSTableInfo):
    """ACS total-population table definition.

    Pulls the total-population row (index 1) from Census base table B01001 into a single
    ``total_pop`` condensed group.
    """

    table_name: str = "pop"
    base_table_strings: tuple[str, ...] = ("B01001",)
    table_indices: tuple[int, ...] = (1,)
    groups_tup: tuple[str, ...] = ("",)
    index_to_name_dict: frozendict = field(default_factory=lambda: frozendict({1: ""}))


@dataclass(frozen=True)
class ACSRacePopTableInfo(ACSNamedTableInfo):
    """ACS race-alone population from detailed table B02001."""

    table_name: str = "race_pop"
    base_table_strings: tuple[str, ...] = ("B02001",)
    table_indices: tuple[int, ...] = tuple(range(1, 11))
    index_to_name_dict: frozendict = field(
        default_factory=lambda: frozendict(
            {
                1: "total_pop",
                2: "white_pop",
                3: "black_pop",
                4: "amin_pop",
                5: "asian_pop",
                6: "nhpi_pop",
                7: "other_pop",
                8: "two_or_more_races_pop",
                9: "two_races_including_other_pop",
                10: "two_races_excluding_other_or_three_plus_races_pop",
            }
        )
    )


ACS_AGE_BANDS: tuple[str, ...] = (
    "under_5",
    "5_to_9",
    "10_to_14",
    "15_to_17",
    "18_to_19",
    "20",
    "21",
    "22_to_24",
    "25_to_29",
    "30_to_34",
    "35_to_39",
    "40_to_44",
    "45_to_49",
    "50_to_54",
    "55_to_59",
    "60_to_61",
    "62_to_64",
    "65_to_66",
    "67_to_69",
    "70_to_74",
    "75_to_79",
    "80_to_84",
    "85_plus",
)


def _acs_age_names() -> frozendict:
    names = {1: "total_pop", 2: "male_pop", 26: "female_pop"}
    for offset, age_band in enumerate(ACS_AGE_BANDS):
        names[3 + offset] = f"male_{age_band}_pop"
        names[27 + offset] = f"female_{age_band}_pop"
    return frozendict(names)


@dataclass(frozen=True)
class ACSAgeTableInfo(ACSNamedTableInfo):
    """ACS sex-by-age population cells and commonly used combined age bands from B01001."""

    table_name: str = "age_pop"
    base_table_strings: tuple[str, ...] = ("B01001",)
    table_indices: tuple[int, ...] = tuple(range(1, 50))
    index_to_name_dict: frozendict = field(default_factory=_acs_age_names)

    @property
    def condense_group_dict(self) -> frozendict:
        """Include combined age bands alongside the individual B01001 cells."""
        groups = dict(super().condense_group_dict)
        groups.update(
            {
                "under_18_pop": tuple(
                    f"B01001_{index:03d}" for index in (*range(3, 7), *range(27, 31))
                ),
                "18_plus_pop": tuple(
                    f"B01001_{index:03d}" for index in (*range(7, 26), *range(31, 50))
                ),
                "18_to_64_pop": tuple(
                    f"B01001_{index:03d}" for index in (*range(7, 20), *range(31, 44))
                ),
                "65_plus_pop": tuple(
                    f"B01001_{index:03d}" for index in (*range(20, 26), *range(44, 50))
                ),
            }
        )
        return frozendict(groups)


@dataclass(frozen=True)
class ACSHispByRaceTableInfo(ACSNamedTableInfo):
    """ACS Hispanic-by-Race table definition.

    Sources variables from Census base table B03002, which cross-tabulates Hispanic origin with
    race. Condenses into ``non_hispanic_*_pop`` and ``hispanic_*_pop`` groups for each race
    category. Because the output keys are per-index rather than per-group, this subclass overrides
    ``condense_group_dict``.
    """

    table_name: str = "hispanic_by_race"
    base_table_strings: tuple[str, ...] = ("B03002",)
    table_indices: tuple[int, ...] = tuple(range(2, 22))
    groups_tup: tuple[str, ...] = ("",)
    index_to_name_dict: frozendict = field(
        default_factory=lambda: frozendict(
            {
                2: "non_hispanic_pop",
                3: "non_hispanic_white_pop",
                4: "non_hispanic_black_pop",
                5: "non_hispanic_amin_pop",
                6: "non_hispanic_asian_pop",
                7: "non_hispanic_nhpi_pop",
                8: "non_hispanic_other_pop",
                9: "non_hispanic_two_or_more_races_pop",
                10: "non_hispanic_two_races_including_other_pop",
                11: "non_hispanic_two_races_excluding_other_or_three_plus_races_pop",
                12: "hispanic_pop",
                13: "hispanic_white_pop",
                14: "hispanic_black_pop",
                15: "hispanic_amin_pop",
                16: "hispanic_asian_pop",
                17: "hispanic_nhpi_pop",
                18: "hispanic_other_pop",
                19: "hispanic_two_or_more_races_pop",
                20: "hispanic_two_races_including_other_pop",
                21: "hispanic_two_races_excluding_other_or_three_plus_races_pop",
            }
        )
    )
