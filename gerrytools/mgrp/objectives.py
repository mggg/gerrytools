"""Objective builders for the rustrecom optimizer runs.

The short-bursts and tilted optimizers steer the chain with a single scoring function, configured
by an *objective spec*: a small dict with an ``"objective"`` key naming the scorer plus its
parameters. :class:`Objective` provides one documented builder per scorer so specs can be
constructed without memorizing the JSON schema; each builder returns a plain dict, so hand-written
spec dicts remain interchangeable with builder output.

Every run needs exactly one objective, passed as the ``objective`` argument of
:class:`~gerrytools.mgrp.ShortBurstsRunInfo` or :class:`~gerrytools.mgrp.TiltedRunInfo`. Whether
the optimizer drives the score up or down is controlled by the run's ``maximize`` flag, not by the
objective itself: for example, ``by_district_abs_deviation`` measures a distance and is normally
*minimized*, while ``gingles_partial`` counts districts and is normally *maximized*.

Example::

    from gerrytools.mgrp import Objective, ShortBurstsRunInfo

    run_info = ShortBurstsRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        objective=Objective.gingles_partial(
            threshold=0.5, min_pop="BVAP", total_pop="VAP"
        ),
        burst_length=25,
        n_steps=5000,
    )
"""

import math
from collections.abc import Sequence
from typing import Literal, NotRequired, Required, TypedDict, cast, get_args, get_type_hints

Aggregation = Literal["mean", "min", "sum"]
"""How per-district or per-election scores are combined into one value."""


class ByDistrictAbsDeviationSpec(TypedDict, total=False):
    """Serialized target-share deviation objective with per-district targets.

    Attributes:
        objective (Literal["by_district_abs_deviation"]): Objective discriminator.
        target_values (list[float]): Target shares matched to the closest districts.
        pov_counts_col (str): Node column containing population-of-interest counts.
        total_counts_col (str, optional): Node column containing denominator counts.
        total_count (float, optional): Global denominator used for every district.

    Exactly one of ``total_counts_col`` and ``total_count`` is required.
    """

    objective: Required[Literal["by_district_abs_deviation"]]
    target_values: Required[list[float]]
    pov_counts_col: Required[str]
    total_counts_col: NotRequired[str]
    total_count: NotRequired[float]


class AbsDeviationSpec(TypedDict, total=False):
    """Serialized target-share deviation objective with a district-count target.

    Attributes:
        objective (Literal["abs_deviation"]): Objective discriminator.
        target (float): Share sought in the closest districts.
        n_target_districts (int): Number of districts matched to ``target``.
        pov_counts_col (str): Node column containing population-of-interest counts.
        total_counts_col (str, optional): Node column containing denominator counts.
        total_count (float, optional): Global denominator used for every district.

    Exactly one of ``total_counts_col`` and ``total_count`` is required.
    """

    objective: Required[Literal["abs_deviation"]]
    target: Required[float]
    n_target_districts: Required[int]
    pov_counts_col: Required[str]
    total_counts_col: NotRequired[str]
    total_count: NotRequired[float]


class GinglesPartialSpec(TypedDict):
    """Serialized single-threshold Gingles objective.

    Attributes:
        objective (Literal["gingles_partial"]): Objective discriminator.
        threshold (float): Minority-share threshold strictly between zero and one.
        min_pop (str): Node column containing minority-population counts.
        total_pop (str): Node column containing total-population counts.
    """

    objective: Literal["gingles_partial"]
    threshold: float
    min_pop: str
    total_pop: str


class BandedGinglesPartialSpec(TypedDict):
    """Serialized banded Gingles objective.

    Attributes:
        objective (Literal["banded_gingles_partial"]): Objective discriminator.
        lower_threshold (float): Inclusive lower edge of the target share band.
        upper_threshold (float): Inclusive upper edge of the target share band.
        min_pop (str): Node column containing minority-population counts.
        total_pop (str): Node column containing total-population counts.
    """

    objective: Literal["banded_gingles_partial"]
    lower_threshold: float
    upper_threshold: float
    min_pop: str
    total_pop: str


class ElectionColumns(TypedDict):
    """Vote-column pair for one election in an election-wins objective.

    Attributes:
        votes_a (str): Node column containing party A vote counts.
        votes_b (str): Node column containing party B vote counts.
    """

    votes_a: str
    votes_b: str


class ElectionWinsSpec(TypedDict):
    """Serialized election-wins objective.

    Attributes:
        objective (Literal["election_wins"]): Objective discriminator.
        elections (list[ElectionColumns]): Vote-column pairs for the elections being combined.
        target (Literal["a", "b"]): Party whose district wins are counted.
        aggregation (Aggregation): Operation combining the per-election scores.
    """

    objective: Literal["election_wins"]
    elections: list[ElectionColumns]
    target: Literal["a", "b"]
    aggregation: Aggregation


class PolsbyPopperSpec(TypedDict, total=False):
    """Serialized Polsby-Popper objective.

    Attributes:
        objective (Literal["polsby_popper"]): Objective discriminator.
        area_col (str): Node column containing unit areas.
        shared_perim_col (str): Edge column containing shared boundary lengths.
        aggregation (Aggregation): Operation combining district scores.
        perim_col (str, optional): Node column containing total unit perimeters.
        boundary_perim_col (str, optional): Node column containing external boundary lengths.

    At least one of ``perim_col`` and ``boundary_perim_col`` is required. When both are present,
    rustrecom derives total perimeters from ``boundary_perim_col``.
    """

    objective: Required[Literal["polsby_popper"]]
    area_col: Required[str]
    shared_perim_col: Required[str]
    aggregation: Required[Aggregation]
    perim_col: NotRequired[str]
    boundary_perim_col: NotRequired[str]


ObjectiveSpec = (
    ByDistrictAbsDeviationSpec
    | AbsDeviationSpec
    | GinglesPartialSpec
    | BandedGinglesPartialSpec
    | ElectionWinsSpec
    | PolsbyPopperSpec
)
"""A discriminated rustrecom objective specification."""

_AGGREGATIONS: tuple[str, ...] = get_args(Aggregation)


def _check_share(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number; got {value!r}.")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]; got {value!r}.")


def _check_positive_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number; got {value!r}.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value!r}.")


def _check_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string; got {value!r}.")
    if not value:
        raise ValueError(f"{name} must be nonempty.")
    return value


def _check_aggregation(aggregation: str) -> None:
    if aggregation not in _AGGREGATIONS:
        raise ValueError(
            f"Unknown aggregation {aggregation!r}. Choose one of: {', '.join(_AGGREGATIONS)}."
        )


class Objective:
    """Builders for rustrecom optimizer objective specs.

    Each static method documents one scorer and returns its spec dict. The definitive scoring
    semantics live in the
    `rustrecom objectives module
    <https://github.com/mggg/rustrecom/blob/bda3898/src/objectives/mod.rs>`_;
    each builder links to the matching definition.

    Note:
        Column names refer to node attributes in the dual graph JSON. The optimizer loads any
        columns an objective requires automatically; they do not need to be listed in
        ``sum_cols``.
    """

    @staticmethod
    def by_district_abs_deviation(
        target_values: Sequence[float],
        pov_counts_col: str,
        total_counts_col: str | None = None,
        total_count: float | None = None,
    ) -> ByDistrictAbsDeviationSpec:
        """Total deviation of district shares from a list of targets.

        Each district's *share* is its sum of ``pov_counts_col`` divided by its sum of
        ``total_counts_col`` (or by the global constant ``total_count``). Given ``k`` target
        shares, the score is the minimum total ``|share - target|`` over all matchings of the
        ``k`` targets to ``k`` distinct districts, so each target claims its own district. Two
        shapes have a simple reading: ``k`` copies of one value ``t`` reward the ``k`` districts
        closest to ``t``, and one value per district scores the full sorted bijection.

        This objective measures a distance, so runs normally set ``maximize=False``.

        Args:
            target_values (Sequence[float]): Target shares, each in ``[0, 1]``. The length must be
                between 1 and the district count (rustrecom validates the upper bound when the
                chain starts).
            pov_counts_col (str): Node-attribute column with the population-of-interest counts
                (integer-valued).
            total_counts_col (str | None, optional): Node-attribute column with the total counts.
                Exactly one of this and ``total_count`` must be given. Defaults to None.
            total_count (float | None, optional): Positive global denominator applied to every
                district instead of a per-district column sum. Defaults to None.

        Returns:
            dict: The objective spec.

        Raises:
            ValueError: If targets, denominator selection, or column names are invalid.

        Note:
            Definition:
            `ByDistrictAbsDeviation
            <https://github.com/mggg/rustrecom/blob/bda3898/src/objectives/mod.rs#L147>`_.
            For many districts sharing one target, :meth:`Objective.abs_deviation` is a shorthand
            for the same scorer.
        """
        if not target_values:
            raise ValueError("target_values must be nonempty.")
        for value in target_values:
            _check_share(value, "each entry of target_values")
        if (total_counts_col is None) == (total_count is None):
            raise ValueError(
                "Provide exactly one of total_counts_col or total_count as the share denominator."
            )
        if total_count is not None:
            _check_positive_number(total_count, "total_count")
        _check_text(pov_counts_col, "pov_counts_col")
        if total_counts_col is not None:
            _check_text(total_counts_col, "total_counts_col")
        spec: ByDistrictAbsDeviationSpec = {
            "objective": "by_district_abs_deviation",
            "target_values": list(target_values),
            "pov_counts_col": pov_counts_col,
        }
        if total_counts_col is not None:
            spec["total_counts_col"] = total_counts_col
        else:
            assert total_count is not None
            spec["total_count"] = total_count
        return spec

    @staticmethod
    def abs_deviation(
        target: float,
        n_target_districts: int,
        pov_counts_col: str,
        total_counts_col: str | None = None,
        total_count: float | None = None,
    ) -> AbsDeviationSpec:
        """Shorthand for ``by_district_abs_deviation`` with one repeated target.

        Equivalent to :meth:`Objective.by_district_abs_deviation` with ``n_target_districts``
        copies of ``target``: the score sums ``|share - target|`` over the ``n_target_districts``
        districts closest to the target. Useful for "get *n* districts near share *t*" goals
        without writing out the list.

        This objective measures a distance, so runs normally set ``maximize=False``.

        Args:
            target (float): The shared target share, in ``[0, 1]``.
            n_target_districts (int): How many districts should chase the target; must be positive
                and at most the district count.
            pov_counts_col (str): Node-attribute column with the population-of-interest counts
                (integer-valued).
            total_counts_col (str | None, optional): Node-attribute column with the total counts.
                Exactly one of this and ``total_count`` must be given. Defaults to None.
            total_count (float | None, optional): Positive global denominator applied to every
                district instead of a per-district column sum. Defaults to None.

        Returns:
            dict: The objective spec.

        Raises:
            TypeError: If ``n_target_districts`` is not an integer.
            ValueError: If targets, denominator selection, or column names are invalid.

        Note:
            Definition:
            `ByDistrictAbsDeviation
            <https://github.com/mggg/rustrecom/blob/bda3898/src/objectives/mod.rs#L147>`_
            (the ``abs_deviation`` alias expands to it).
        """
        _check_share(target, "target")
        if isinstance(n_target_districts, bool) or not isinstance(n_target_districts, int):
            raise TypeError(
                f"n_target_districts must be a positive integer; got {n_target_districts!r}."
            )
        if n_target_districts < 1:
            raise ValueError("n_target_districts must be a positive integer.")
        if (total_counts_col is None) == (total_count is None):
            raise ValueError(
                "Provide exactly one of total_counts_col or total_count as the share denominator."
            )
        if total_count is not None:
            _check_positive_number(total_count, "total_count")
        _check_text(pov_counts_col, "pov_counts_col")
        if total_counts_col is not None:
            _check_text(total_counts_col, "total_counts_col")
        spec: AbsDeviationSpec = {
            "objective": "abs_deviation",
            "target": target,
            "n_target_districts": n_target_districts,
            "pov_counts_col": pov_counts_col,
        }
        if total_counts_col is not None:
            spec["total_counts_col"] = total_counts_col
        else:
            assert total_count is not None
            spec["total_count"] = total_count
        return spec

    @staticmethod
    def gingles_partial(threshold: float, min_pop: str, total_pop: str) -> GinglesPartialSpec:
        """Count Gingles opportunity districts, with a partial tiebreaker.

        The score is the number of districts whose minority share (district sum of ``min_pop``
        over district sum of ``total_pop``) exceeds ``threshold``, plus a fractional reward from
        the single highest sub-threshold district. The fraction gives the optimizer gradient
        signal between integer counts, so plans closer to gaining a new opportunity district
        score higher.

        This objective counts districts, so runs normally set ``maximize=True``.

        Args:
            threshold (float): Minority share threshold, strictly between 0 and 1.
            min_pop (str): Node-attribute column with the minority population (integer-valued).
            total_pop (str): Node-attribute column with the total population (integer-valued).

        Returns:
            dict: The objective spec.

        Raises:
            ValueError: If the threshold or column names are invalid.

        Note:
            Definition: `GinglesPartial
            <https://github.com/mggg/rustrecom/blob/bda3898/src/objectives/mod.rs#L176>`_.
        """
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be strictly between 0 and 1; got {threshold!r}.")
        _check_text(min_pop, "min_pop")
        _check_text(total_pop, "total_pop")
        return {
            "objective": "gingles_partial",
            "threshold": threshold,
            "min_pop": min_pop,
            "total_pop": total_pop,
        }

    @staticmethod
    def banded_gingles_partial(
        lower_threshold: float,
        upper_threshold: float,
        min_pop: str,
        total_pop: str,
    ) -> BandedGinglesPartialSpec:
        """Count opportunity districts inside a share band, penalizing overshoot.

        Each district whose minority share lands in ``[lower_threshold, upper_threshold]``
        contributes a full ``1.0``. The single highest district strictly below the band adds a
        fractional reward ``share / lower_threshold`` (gradient toward gaining a new in-band
        district), and *every* district strictly above the band contributes a penalized
        ``upper_threshold / share`` in ``(0, 1)``, so the demerit grows the further a district
        overshoots. Compared to :meth:`Objective.gingles_partial`, this discourages packing:
        pushing a district far past the band costs score instead of counting the same.

        This objective counts districts, so runs normally set ``maximize=True``.

        Args:
            lower_threshold (float): Lower edge of the target band, strictly between 0 and 1.
            upper_threshold (float): Upper edge of the target band, strictly between 0 and 1 and
                at least ``lower_threshold``.
            min_pop (str): Node-attribute column with the minority population (integer-valued).
            total_pop (str): Node-attribute column with the total population (integer-valued).

        Returns:
            dict: The objective spec.

        Raises:
            ValueError: If the thresholds or column names are invalid.

        Note:
            Definition: `BandedGinglesPartial
            <https://github.com/mggg/rustrecom/blob/bda3898/src/objectives/mod.rs#L217>`_.
        """
        for name, value in (
            ("lower_threshold", lower_threshold),
            ("upper_threshold", upper_threshold),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must be strictly between 0 and 1; got {value!r}.")
        if upper_threshold < lower_threshold:
            raise ValueError("upper_threshold must be at least lower_threshold.")
        _check_text(min_pop, "min_pop")
        _check_text(total_pop, "total_pop")
        return {
            "objective": "banded_gingles_partial",
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
            "min_pop": min_pop,
            "total_pop": total_pop,
        }

    @staticmethod
    def election_wins(
        elections: Sequence[ElectionColumns],
        target: Literal["a", "b"] = "a",
        aggregation: Aggregation = "mean",
    ) -> ElectionWinsSpec:
        """Count districts won by a target party across a set of elections.

        For each election, the score is the number of districts where the target party's vote
        total exceeds the other party's, plus a fractional tiebreaker from the closest losing
        district (gradient signal between integer win counts). Per-election scores are combined
        with ``aggregation``: ``"mean"`` for a typical-performance goal, ``"min"`` to optimize the
        worst election, ``"sum"`` for the total across elections.

        Use ``maximize=True`` to seek plans favoring the target party, or ``maximize=False`` to
        seek plans disfavoring it.

        Args:
            elections (Sequence[dict]): One dict per election, each with ``"votes_a"`` and
                ``"votes_b"`` naming node-attribute columns of integer vote counts, e.g.
                ``[{"votes_a": "DEM_GOV_18", "votes_b": "REP_GOV_18"}]``.
            target (str, optional): Party whose wins are counted. Defaults to ``"a"``.
            aggregation (str, optional): Election aggregation. Defaults to ``"mean"``; also
                accepts ``"min"`` and ``"sum"``.

        Returns:
            dict: The objective spec.

        Raises:
            ValueError: If elections are empty or malformed, or an option is unknown.

        Note:
            Definition: `ElectionWins
            <https://github.com/mggg/rustrecom/blob/bda3898/src/objectives/mod.rs#L251>`_.
        """
        if not elections:
            raise ValueError("elections must be a nonempty sequence.")
        for election in elections:
            if set(election) != {"votes_a", "votes_b"}:
                raise ValueError(
                    "Each election must be a dict with exactly the keys 'votes_a' and "
                    f"'votes_b'; got {election!r}."
                )
            _check_text(election["votes_a"], "votes_a")
            _check_text(election["votes_b"], "votes_b")
        if target not in ("a", "b"):
            raise ValueError(f"target must be 'a' or 'b'; got {target!r}.")
        _check_aggregation(aggregation)
        return {
            "objective": "election_wins",
            "elections": [
                ElectionColumns(votes_a=election["votes_a"], votes_b=election["votes_b"])
                for election in elections
            ],
            "target": target,
            "aggregation": aggregation,
        }

    @staticmethod
    def polsby_popper(
        area_col: str,
        shared_perim_col: str,
        perim_col: str | None = None,
        boundary_perim_col: str | None = None,
        aggregation: Aggregation = "mean",
    ) -> PolsbyPopperSpec:
        """Aggregate Polsby-Popper compactness across districts.

        The Polsby-Popper score of a district is ``4 * pi * area / perimeter^2`` (1.0 for a disc,
        approaching 0 for winding shapes). A district's area is the sum of ``area_col``; its
        perimeter combines the ``shared_perim_col`` edge attribute on cut edges with each
        precinct's total perimeter. Per-district scores are combined with ``aggregation``.

        Perimeter data can be supplied two ways: ``perim_col`` naming each precinct's total
        perimeter directly, or ``boundary_perim_col`` naming only the outer-hull contribution
        (nonzero on boundary precincts), from which rustrecom derives total perimeters using the
        shared-perimeter edges. At least one of the two must be given; when both are given the
        derived values overwrite ``perim_col``.

        Compactness is a quality score, so runs normally set ``maximize=True``.

        Args:
            area_col (str): Node-attribute column with precinct areas (float-valued).
            shared_perim_col (str): Edge-attribute column with the shared perimeter between
                adjacent precincts (float-valued).
            perim_col (str | None, optional): Node-attribute column with each precinct's total
                perimeter, including shared boundaries. Defaults to None.
            boundary_perim_col (str | None, optional): Node-attribute column with each precinct's
                outer-hull perimeter contribution. Defaults to None.
            aggregation (str, optional): District aggregation. Defaults to ``"mean"``; also accepts
                ``"min"`` and ``"sum"``.

        Returns:
            dict: The objective spec.

        Raises:
            ValueError: If no perimeter column is supplied or a column or option is invalid.

        Note:
            Definition: `PolsbyPopper
            <https://github.com/mggg/rustrecom/blob/bda3898/src/objectives/mod.rs#L291>`_.
        """
        if perim_col is None and boundary_perim_col is None:
            raise ValueError("Provide at least one of perim_col or boundary_perim_col.")
        _check_text(area_col, "area_col")
        _check_text(shared_perim_col, "shared_perim_col")
        if perim_col is not None:
            _check_text(perim_col, "perim_col")
        if boundary_perim_col is not None:
            _check_text(boundary_perim_col, "boundary_perim_col")
        _check_aggregation(aggregation)
        spec: PolsbyPopperSpec = {
            "objective": "polsby_popper",
            "area_col": area_col,
            "shared_perim_col": shared_perim_col,
            "aggregation": aggregation,
        }
        if perim_col is not None:
            spec["perim_col"] = perim_col
        if boundary_perim_col is not None:
            spec["boundary_perim_col"] = boundary_perim_col
        return spec


_OBJECTIVE_KEYS = {
    get_args(get_type_hints(spec_type)["objective"])[0]: (
        set(spec_type.__required_keys__),
        set(spec_type.__optional_keys__),
    )
    for spec_type in get_args(ObjectiveSpec)
}


def _validate_keys(spec: dict[str, object], kind: str) -> None:
    required, optional = _OBJECTIVE_KEYS[kind]
    missing = required - spec.keys()
    extra = spec.keys() - required - optional
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if extra:
            details.append(f"unexpected {sorted(extra)!r}")
        raise ValueError(f"Invalid {kind!r} objective fields: {', '.join(details)}.")


def _text(spec: dict[str, object], key: str) -> str:
    return _check_text(spec[key], f"objective field {key!r}")


def _optional_text(spec: dict[str, object], key: str) -> str | None:
    return _text(spec, key) if key in spec else None


def _number(spec: dict[str, object], key: str) -> float:
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"objective field {key!r} must be a number; got {value!r}.")
    return value


def _optional_number(spec: dict[str, object], key: str) -> float | None:
    return _number(spec, key) if key in spec else None


def _positive_integer(spec: dict[str, object], key: str) -> int:
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"objective field {key!r} must be an integer; got {value!r}.")
    if value <= 0:
        raise ValueError(f"objective field {key!r} must be positive; got {value!r}.")
    return value


def _number_list(spec: dict[str, object], key: str) -> list[float]:
    values = spec[key]
    if not isinstance(values, list):
        raise TypeError(f"objective field {key!r} must be a list of numbers; got {values!r}.")
    return [_number({key: value}, key) for value in values]


def _elections(spec: dict[str, object]) -> list[ElectionColumns]:
    values = spec["elections"]
    if not isinstance(values, list):
        raise TypeError(f"objective field 'elections' must be a list; got {values!r}.")
    elections: list[ElectionColumns] = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"votes_a", "votes_b"}:
            raise ValueError(
                "Each election must be a dict with exactly the keys 'votes_a' and "
                f"'votes_b'; got {value!r}."
            )
        election = cast(dict[str, object], value)
        elections.append(
            {
                "votes_a": _text(election, "votes_a"),
                "votes_b": _text(election, "votes_b"),
            }
        )
    return elections


def validate_objective_spec(value: object) -> ObjectiveSpec:
    """Copy and validate a raw objective document before container construction.

    Args:
        value (object): Raw objective dictionary.

    Returns:
        ObjectiveSpec: Copied, validated objective specification.

    Raises:
        TypeError: If the document or one of its values has the wrong type.
        ValueError: If its name, fields, or values are invalid.
    """
    if not isinstance(value, dict):
        raise TypeError(f"objective must be a dictionary; got {type(value).__name__}.")
    spec = cast(dict[str, object], dict(value))
    kind = spec.get("objective")
    if not isinstance(kind, str) or kind not in _OBJECTIVE_KEYS:
        raise ValueError(
            f"Unknown objective {kind!r}. Known objectives: {', '.join(sorted(_OBJECTIVE_KEYS))}."
        )
    _validate_keys(spec, kind)

    if kind == "by_district_abs_deviation":
        return Objective.by_district_abs_deviation(
            target_values=_number_list(spec, "target_values"),
            pov_counts_col=_text(spec, "pov_counts_col"),
            total_counts_col=_optional_text(spec, "total_counts_col"),
            total_count=_optional_number(spec, "total_count"),
        )
    if kind == "abs_deviation":
        return Objective.abs_deviation(
            target=_number(spec, "target"),
            n_target_districts=_positive_integer(spec, "n_target_districts"),
            pov_counts_col=_text(spec, "pov_counts_col"),
            total_counts_col=_optional_text(spec, "total_counts_col"),
            total_count=_optional_number(spec, "total_count"),
        )
    if kind == "gingles_partial":
        return Objective.gingles_partial(
            threshold=_number(spec, "threshold"),
            min_pop=_text(spec, "min_pop"),
            total_pop=_text(spec, "total_pop"),
        )
    if kind == "banded_gingles_partial":
        return Objective.banded_gingles_partial(
            lower_threshold=_number(spec, "lower_threshold"),
            upper_threshold=_number(spec, "upper_threshold"),
            min_pop=_text(spec, "min_pop"),
            total_pop=_text(spec, "total_pop"),
        )
    if kind == "election_wins":
        target = _text(spec, "target")
        aggregation = _text(spec, "aggregation")
        return Objective.election_wins(
            elections=_elections(spec),
            target=cast(Literal["a", "b"], target),
            aggregation=cast(Aggregation, aggregation),
        )
    return Objective.polsby_popper(
        area_col=_text(spec, "area_col"),
        shared_perim_col=_text(spec, "shared_perim_col"),
        perim_col=_optional_text(spec, "perim_col"),
        boundary_perim_col=_optional_text(spec, "boundary_perim_col"),
        aggregation=cast(Aggregation, _text(spec, "aggregation")),
    )
