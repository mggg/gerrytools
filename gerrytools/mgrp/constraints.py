"""Build sampler constraints through a shared Python interface.

The ReCom, SMC, and Forest runners use different constraint implementations, but each can be
configured from scalar values and graph or map column names. :class:`Constraints` collects those
settings as plain dictionaries that the selected runner translates for its engine.

The word "constraint" does not imply the same behavior for every engine. ReCom and Forest use hard
feasibility rules that reject invalid states or proposals. The SMC methods wrap ``redist`` Gibbs
penalties, which change plan weights rather than making a plan categorically invalid.

This module validates each specification's shape, value domains, and engine compatibility. It
does not inspect input data or confirm that named columns exist; those checks remain with the
selected engine.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Literal, TypedDict, cast, get_args, get_type_hints


class DistrictShareFloorSpec(TypedDict):
    """Serialized rustrecom district-share-floor constraint.

    Attributes:
        constraint (Literal["district_share_floor"]): Constraint discriminator.
        numerator_col (str): Node column summed for the share numerator.
        denominator_cols (list[str]): Node columns whose district sums form the denominator.
        threshold (float): Inclusive minimum permitted share in every district.
    """

    constraint: Literal["district_share_floor"]
    numerator_col: str
    denominator_cols: list[str]
    threshold: float


class GroupHingeSpec(TypedDict):
    """Serialized redist group-hinge penalty.

    Attributes:
        constraint (Literal["group_hinge"]): Constraint discriminator.
        strength (float): Gibbs penalty coefficient.
        group_pop_col (str): Map column containing the focal-group population.
        total_pop_col (str | None): Denominator column, or None to use the SMC population column.
        targets (list[float]): Candidate target shares used independently for each district.
    """

    constraint: Literal["group_hinge"]
    strength: float
    group_pop_col: str
    total_pop_col: str | None
    targets: list[float]


class GroupPowerSpec(TypedDict):
    """Serialized redist group-power penalty.

    Attributes:
        constraint (Literal["group_power"]): Constraint discriminator.
        strength (float): Gibbs penalty coefficient.
        group_pop_col (str): Map column containing the focal-group population.
        total_pop_col (str): Map column containing the denominator population.
        target_group (float): First target for the focal-group share.
        target_other (float): Second target for the same focal-group share.
        pow (float): Exponent applied to the product of distances from the targets.
    """

    constraint: Literal["group_power"]
    strength: float
    group_pop_col: str
    total_pop_col: str
    target_group: float
    target_other: float
    pow: float


class StatusQuoSpec(TypedDict):
    """Serialized redist status-quo penalty.

    Attributes:
        constraint (Literal["status_quo"]): Constraint discriminator.
        strength (float): Gibbs penalty coefficient.
        plan_col (str): Map column containing the reference district assignment.
    """

    constraint: Literal["status_quo"]
    strength: float
    plan_col: str


class SplitsSpec(TypedDict):
    """Serialized redist administrative-splits penalty.

    Attributes:
        constraint (Literal["splits"]): Constraint discriminator.
        strength (float): Gibbs penalty coefficient.
        admin_col (str): Map column identifying administrative-unit membership.
    """

    constraint: Literal["splits"]
    strength: float
    admin_col: str


class PackNodesSpec(TypedDict):
    """Serialized MSMS packed-node constraint.

    Attributes:
        constraint (Literal["pack_nodes"]): Constraint discriminator.
        unpack (int): Number subtracted from each hierarchy node's packed-district target.
    """

    constraint: Literal["pack_nodes"]
    unpack: int


class MaxCoarseNodeSplitsSpec(TypedDict):
    """Serialized MSMS coarse-node split-budget constraint.

    Attributes:
        constraint (Literal["max_coarse_node_splits"]): Constraint discriminator.
        max_splits (int): Maximum total excess district intersections across coarse nodes.
    """

    constraint: Literal["max_coarse_node_splits"]
    max_splits: int


class AllowedExcessDistrictsSpec(TypedDict):
    """Serialized MSMS per-coarse-node excess-district constraint.

    Attributes:
        constraint (Literal["allowed_excess_dists_in_coarse_nodes"]): Constraint discriminator.
        allowable_excess (int): Additional district intersections permitted per coarse node.
    """

    constraint: Literal["allowed_excess_dists_in_coarse_nodes"]
    allowable_excess: int


class MaxDiscontinuousTraversalSegmentsSpec(TypedDict):
    """Serialized MSMS traversal-connectivity constraint.

    Attributes:
        constraint (Literal["max_discontinuous_traversal_segments"]): Constraint discriminator.
        max_line_segments (int): Maximum traversal components; MSMS currently supports only one.
    """

    constraint: Literal["max_discontinuous_traversal_segments"]
    max_line_segments: int


ConstraintSpec = (
    DistrictShareFloorSpec
    | GroupHingeSpec
    | GroupPowerSpec
    | StatusQuoSpec
    | SplitsSpec
    | PackNodesSpec
    | MaxCoarseNodeSplitsSpec
    | AllowedExcessDistrictsSpec
    | MaxDiscontinuousTraversalSegmentsSpec
)

# Which runners accept each constraint type. The builder methods document the engine semantics;
# this table drives compatibility validation and error messages.
ENGINE_SUPPORT = {
    "district_share_floor": ("recom",),
    "group_hinge": ("smc",),
    "group_power": ("smc",),
    "status_quo": ("smc",),
    "splits": ("smc",),
    "pack_nodes": ("forest",),
    "max_coarse_node_splits": ("forest",),
    "allowed_excess_dists_in_coarse_nodes": ("forest",),
    "max_discontinuous_traversal_segments": ("forest",),
}


class Constraints:
    """Collect constraints for a ReCom, SMC, or Forest ensemble run.

    Each public method appends one plain-dictionary specification and returns this builder, allowing
    several calls to be chained. A runner validates the accumulated specifications against its
    engine before starting the container.

    Constraint types from different engines may be placed in one builder, but no individual runner
    can execute that mixed collection. For example, an SMC runner rejects a Forest constraint and
    reports which engine supports it.

    Examples:
        Build two soft SMC penalties::

            constraints = (
                Constraints()
                .group_hinge(
                    strength=1500.0,
                    group_pop_col="BVAP",
                    total_pop_col="VAP",
                )
                .splits(strength=100.0, admin_col="COUNTY")
            )

        Build a Forest constraint that replaces the runner's default coarse-node split budget::

            constraints = Constraints().max_coarse_node_splits(max_splits=8)

    Note:
        :meth:`specs` returns deep copies of the stored dictionaries. Modifying its return value,
        including nested lists, does not modify this builder.

        Forest stores one constraint per underlying MSMS constraint type, so repeated calls of the
        same Forest method do not stack when executed; the last specification of that type wins.
        ReCom currently accepts only one constraint specification per run.
    """

    def __init__(self):
        """Initialize an empty constraint builder."""
        self._specs: list[ConstraintSpec] = []

    def specs(self) -> list[dict[str, object]]:
        """Return deep copies of the accumulated constraint specifications.

        Returns:
            list[dict]: Constraint dictionaries in the order they were added.
        """
        return [cast(dict[str, object], copy.deepcopy(spec)) for spec in self._specs]

    def _add(self, spec: ConstraintSpec) -> Constraints:
        """Append one specification and return this builder."""
        self._specs.append(validate_constraint_spec(spec))
        return self

    # --- recom (rustrecom) ---------------------------------------------------------

    def district_share_floor(
        self,
        numerator_col: str,
        denominator_cols: Sequence[str],
        threshold: float,
    ) -> Constraints:
        """(RustReCom) Require every district to meet a minimum population share.

        For each district, ``rustrecom`` divides the district sum of ``numerator_col`` by the sum
        of all district totals named in ``denominator_cols``. The initial assignment must satisfy
        ``share >= threshold`` in every district, and a proposal is rejected if either changed
        district falls below the floor. Equality with the threshold is allowed.

        A zero denominator is treated as a share of ``0.0``. The numerator is not automatically
        included in the denominator, and repeated denominator column names are counted repeatedly.

        Args:
            numerator_col (str): Numeric node-attribute column used as the numerator.
            denominator_cols (Sequence[str]): Nonempty sequence of numeric node-attribute columns
                whose district sums form the denominator. Pass a sequence such as ``["VAP"]``;
                passing a bare string would be interpreted as a sequence of characters.
            threshold (float): Minimum permitted district share. ``rustrecom`` expects a finite
                value in the inclusive range ``[0, 1]``.

        Returns:
            Constraints: This builder instance, allowing additional methods to be chained.

        Note:
            Missing or nonnumeric graph columns are reported by ``rustrecom`` after the container
            starts.
            ReCom currently accepts only one constraint specification per run.
        """
        return self._add(
            {
                "constraint": "district_share_floor",
                "numerator_col": numerator_col,
                "denominator_cols": list(denominator_cols),
                "threshold": threshold,
            }
        )

    # --- smc (redist) ---------------------------------------------------------

    def group_hinge(
        self,
        strength: float,
        group_pop_col: str,
        total_pop_col: str | None = None,
        targets: Sequence[float] = (0.55,),
    ) -> Constraints:
        """(SMC) Penalize districts whose group share falls below a target.

        See `add_constr_grp_hinge() in the ALARM redist documentation
        <https://alarm-redist.org/redist/reference/constraints.html>`_.

        For each district, ``redist`` computes ``group_pop_col / total_pop_col``, selects the
        nearest value in ``targets``, and adds ``sqrt(max(0, target - share))`` to the plan
        statistic. A district above its selected target receives no penalty. Targets are selected
        independently by district; they do not impose a quota on how many districts must reach each
        target.

        SMC constraints are soft Gibbs penalties. With positive ``strength``, larger statistics
        receive less weight; ``strength=0`` has no effect, and a negative value reverses the
        preference. Very large strengths can reduce SMC efficiency and should be calibrated with
        sampler diagnostics.

        Args:
            strength (float): Coefficient multiplying the penalty statistic.
            group_pop_col (str): Map column containing the focal-group population.
            total_pop_col (str | None, optional): Denominator population column. If ``None``,
                ``redist`` uses the population column supplied through ``SMCRunSpec.pop_col``.
                Defaults to None.
            targets (Sequence[float], optional): Nonempty target-share sequence. Values should be
                finite and in ``[0, 1]``. Defaults to ``(0.55,)``.

        Returns:
            Constraints: This builder instance, allowing additional methods to be chained.

        Note:
            Zero district denominators are handled by ``redist`` when the sampler runs.
        """
        return self._add(
            {
                "constraint": "group_hinge",
                "strength": strength,
                "group_pop_col": group_pop_col,
                "total_pop_col": total_pop_col,
                "targets": list(targets),
            }
        )

    def group_power(
        self,
        strength: float,
        group_pop_col: str,
        total_pop_col: str,
        target_group: float = 0.5,
        target_other: float = 0.5,
        pow: float = 1.0,
    ) -> Constraints:
        """(SMC) Apply ``redist``'s power penalty to district group shares.

        See `add_constr_grp_pow() in the ALARM redist documentation
        <https://alarm-redist.org/redist/reference/constraints.html>`_.

        For a district group share ``s``, the statistic is
        ``(|s - target_group| * |s - target_other|) ** pow``. The plan statistic is the sum across
        districts. With the default equal targets and exponent, this becomes ``|s - 0.5| ** 2``.

        ``target_group`` and ``target_other`` are two alternative targets for the same focal-group
        share; ``target_other`` does not refer to a second population group. Positive ``strength``
        and ``pow`` favor shares near either target.

        Args:
            strength (float): Coefficient multiplying the penalty statistic.
            group_pop_col (str): Map column containing the focal-group population.
            total_pop_col (str): Map column containing the denominator population.
            target_group (float, optional): First target share. Defaults to ``0.5``.
            target_other (float, optional): Second target share for the same group. Defaults to
                ``0.5``.
            pow (float, optional): Exponent applied to the product of distances. Positive values are
                the intended domain. Defaults to ``1.0``.

        Returns:
            Constraints: This builder instance, allowing additional methods to be chained.

        Note:
            ``redist`` describes this as an expert-use penalty because its scale depends strongly on
            the targets and exponent. Plot or otherwise inspect the statistic before choosing a
            large strength.
        """
        return self._add(
            {
                "constraint": "group_power",
                "strength": strength,
                "group_pop_col": group_pop_col,
                "total_pop_col": total_pop_col,
                "target_group": target_group,
                "target_other": target_other,
                "pow": pow,
            }
        )

    def status_quo(self, strength: float, plan_col: str) -> Constraints:
        """(SMC) Penalize plans that depart from a reference districting plan.

        See `add_constr_status_quo() in the ALARM redist documentation
        <https://alarm-redist.org/redist/reference/constraints.html>`_.

        This wraps ``redist``'s population-weighted, rescaled status-quo penalty. It compares how
        reference-district population is distributed across the sampled districts, rather than
        counting units whose numeric district label changed. Relabeling otherwise identical
        districts therefore does not by itself represent a substantive change.

        Args:
            strength (float): Coefficient multiplying the status-quo penalty.
            plan_col (str): Map column containing the reference district assignment. Positive,
                contiguous, one-based district labels are the safe input convention, and the
                reference plan should have the same number of districts as the sampled plans.

        Returns:
            Constraints: This builder instance, allowing additional methods to be chained.
        """
        return self._add({"constraint": "status_quo", "strength": strength, "plan_col": plan_col})

    def splits(self, strength: float, admin_col: str) -> Constraints:
        """(SMC) Penalize administrative units split across multiple districts.

        See `add_constr_splits() in the ALARM redist documentation
        <https://alarm-redist.org/redist/reference/constraints.html>`_.

        The statistic counts administrative units that appear in at least two districts. Each unit
        contributes at most ``1``, even if it is divided among three or more districts; this is not
        a count of pieces or excess district-unit intersections.

        Args:
            strength (float): Coefficient multiplying the split-unit count. Because the statistic
                is a count, ``redist`` generally recommends starting with a small positive value.
            admin_col (str): Map column identifying administrative-unit membership. The values need
                not represent counties, but they must not be missing.

        Returns:
            Constraints: This builder instance, allowing additional methods to be chained.
        """
        return self._add({"constraint": "splits", "strength": strength, "admin_col": admin_col})

    # An incumbency builder for redist's add_constr_incumbency() is deliberately absent: the
    # Docker translation forwards a whole map column where redist expects one-based incumbent
    # row indices, so no column shape produces a working configuration. Reintroduce it once the
    # translation converts an indicator column to indices.

    # --- forest (MSMS) --------------------------------------------------------

    def pack_nodes(self, unpack: int = 0) -> Constraints:
        """(MSMS) Require population-heavy nodes to meet packed-district targets.

        See the `MSMS PackNodeConstraint implementation
        <https://github.com/mggg/Multi-Scale-Map-Sampler/blob/3f4a6c8/src/constraints.jl#L38-L64>`_.

        At every hierarchy level, MSMS computes
        ``floor(node_population / ideal_population) - unpack`` for each node and omits nodes whose
        target is not positive. The remaining nodes must satisfy MSMS's packed-district feasibility
        check. The ideal population is total population divided by ``num_dists``.

        Args:
            unpack (int, optional): Number of districts subtracted from every node's target.
                Increasing it loosens the constraint and may reduce a target to zero, removing that
                node from the constraint. Defaults to ``0``.

        Returns:
            Constraints: This builder instance, allowing additional methods to be chained.

        Note:
            The rule applies independently to nodes at every hierarchy level, including nodes whose
            population equals one ideal district. Negative ``unpack`` values strengthen the rule and
            are not rejected by this builder.
        """
        return self._add({"constraint": "pack_nodes", "unpack": unpack})

    def max_coarse_node_splits(self, max_splits: int) -> Constraints:
        """(MSMS) Set the budget for excess district intersections with coarse nodes.

        See the `MSMS MaxCoarseNodeSplits implementation
        <https://github.com/mggg/Multi-Scale-Map-Sampler/blob/3f4a6c8/src/constraints.jl#L220>`_.

        For every top-level coarse node ``N``, let ``k_N`` be the number of districts intersecting
        it. MSMS requires ``sum(k_N - 1) <= max_splits``. A node intersected by three districts thus
        consumes two units of the budget; this is not a binary count of how many nodes are split.

        Args:
            max_splits (int): Maximum total excess district intersections across top-level coarse
                nodes. A user specification replaces the Forest runner default of
                ``num_dists + 1``.

        Returns:
            Constraints: This builder instance, allowing additional methods to be chained.
        """
        return self._add({"constraint": "max_coarse_node_splits", "max_splits": max_splits})

    def allowed_excess_dists_in_coarse_nodes(self, allowable_excess: int = 0) -> Constraints:
        """(MSMS) Limit district intersections separately within each coarse node.

        See the `MSMS AllowedExcessDistsInCoarseNodes implementation
        <https://github.com/mggg/Multi-Scale-Map-Sampler/blob/3f4a6c8/src/constraints.jl#L296>`_.

        For a top-level coarse node at or above one ideal district of population, the baseline is
        ``ceil(node_population / ideal_population)`` intersecting districts. A node below one ideal
        district instead uses a baseline of two. ``allowable_excess`` is added to that per-node
        baseline.

        Args:
            allowable_excess (int, optional): Additional intersecting districts permitted per
                top-level coarse node. Defaults to ``0``.

        Returns:
            Constraints: This builder instance, allowing additional methods to be chained.

        Note:
            The calculation uses ideal population but not the Forest population-deviation tolerance.
            At exactly one ideal district of population, the first formula applies and the baseline
            is one; immediately below ideal population, the special-case baseline is two.
        """
        return self._add(
            {
                "constraint": "allowed_excess_dists_in_coarse_nodes",
                "allowable_excess": allowable_excess,
            }
        )

    def max_discontinuous_traversal_segments(self, max_line_segments: int = 1) -> Constraints:
        """(MSMS) Require connected traversal within partially included hierarchy nodes.

        See the `MSMS ConstrainDiscontinuousTraversals implementation
        <https://github.com/mggg/Multi-Scale-Map-Sampler/blob/3f4a6c8/src/constraints.jl#L127>`_.

        Before a Forest ReCom cut, MSMS examines the merged region selected for recombination. In
        every hierarchy node that the region only partially occupies, the included child-level nodes
        must induce one connected graph component. This is a graph-connectivity rule, not a count of
        visual or geometric line segments in each resulting district.

        Args:
            max_line_segments (int, optional): Intended maximum number of traversal components.
                Defaults to ``1``.

        Returns:
            Constraints: This builder instance, allowing additional methods to be chained.

        Raises:
            ValueError: If ``max_line_segments`` is not ``1``. MSMS implements only that value
                and asserts at runtime for any other, so unsupported values are rejected here
                instead of failing mid-run.
        """
        if max_line_segments != 1:
            raise ValueError(
                "MSMS implements only max_line_segments=1; other values fail its runtime assert."
            )
        return self._add(
            {
                "constraint": "max_discontinuous_traversal_segments",
                "max_line_segments": max_line_segments,
            }
        )


_CONSTRAINT_KEYS = {
    get_args(get_type_hints(spec_type)["constraint"])[0]: set(spec_type.__required_keys__)
    | set(spec_type.__optional_keys__)
    for spec_type in get_args(ConstraintSpec)
}


def _constraint_text(spec: dict[str, object], key: str) -> str:
    value = spec[key]
    if not isinstance(value, str):
        raise TypeError(f"constraint field {key!r} must be a string; got {value!r}.")
    if not value:
        raise ValueError(f"constraint field {key!r} must be nonempty.")
    return value


def _constraint_number(spec: dict[str, object], key: str) -> float:
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"constraint field {key!r} must be a finite number; got {value!r}.")
    if not math.isfinite(value):
        raise ValueError(f"constraint field {key!r} must be a finite number; got {value!r}.")
    return value


def _constraint_integer(spec: dict[str, object], key: str) -> int:
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"constraint field {key!r} must be an integer; got {value!r}.")
    return value


def _constraint_string_list(spec: dict[str, object], key: str) -> list[str]:
    values = spec[key]
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(item, str) for item in values)
    ):
        raise TypeError(
            f"constraint field {key!r} must be a nonempty list of strings; got {values!r}."
        )
    if not all(values):
        raise ValueError(f"constraint field {key!r} cannot contain empty strings.")
    return [cast(str, item) for item in values]


def _constraint_number_list(spec: dict[str, object], key: str) -> list[float]:
    values = spec[key]
    if not isinstance(values, list) or not values:
        raise TypeError(
            f"constraint field {key!r} must be a nonempty list of numbers; got {values!r}."
        )
    return [_constraint_number({key: item}, key) for item in values]


def _check_unit_interval(value: float, key: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"constraint field {key!r} must be in [0, 1]; got {value!r}.")


def validate_constraint_spec(value: object) -> ConstraintSpec:
    """Copy and validate one constraint document before container construction.

    The copy is deep so nested values (e.g. ``denominator_cols``) never alias the input.

    Args:
        value (object): Raw constraint dictionary.

    Returns:
        ConstraintSpec: Deep-copied, validated constraint specification.

    Raises:
        TypeError: If the document or one of its values has the wrong type.
        ValueError: If its name, fields, or values are invalid.
    """
    if not isinstance(value, dict):
        raise TypeError(f"constraint must be a dictionary; got {type(value).__name__}.")
    spec = cast(dict[str, object], copy.deepcopy(value))
    kind = spec.get("constraint")
    if not isinstance(kind, str) or kind not in _CONSTRAINT_KEYS:
        raise ValueError(
            f"Unknown constraint {kind!r}. Known constraints: "
            f"{', '.join(sorted(_CONSTRAINT_KEYS))}."
        )
    expected = _CONSTRAINT_KEYS[kind]
    if set(spec) != expected:
        missing = expected - spec.keys()
        extra = spec.keys() - expected
        details = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if extra:
            details.append(f"unexpected {sorted(extra)!r}")
        raise ValueError(f"Invalid {kind!r} constraint fields: {', '.join(details)}.")

    if kind == "district_share_floor":
        _constraint_text(spec, "numerator_col")
        _constraint_string_list(spec, "denominator_cols")
        _check_unit_interval(_constraint_number(spec, "threshold"), "threshold")
    elif kind == "group_hinge":
        _constraint_number(spec, "strength")
        _constraint_text(spec, "group_pop_col")
        total_pop_col = spec["total_pop_col"]
        if total_pop_col is not None:
            _constraint_text(spec, "total_pop_col")
        for target in _constraint_number_list(spec, "targets"):
            _check_unit_interval(target, "targets")
    elif kind == "group_power":
        _constraint_number(spec, "strength")
        _constraint_text(spec, "group_pop_col")
        _constraint_text(spec, "total_pop_col")
        _check_unit_interval(_constraint_number(spec, "target_group"), "target_group")
        _check_unit_interval(_constraint_number(spec, "target_other"), "target_other")
        if _constraint_number(spec, "pow") <= 0:
            raise ValueError("constraint field 'pow' must be positive.")
    elif kind in ("status_quo", "splits"):
        _constraint_number(spec, "strength")
        _constraint_text(spec, "plan_col" if kind == "status_quo" else "admin_col")
    elif kind == "max_discontinuous_traversal_segments":
        if _constraint_integer(spec, "max_line_segments") != 1:
            raise ValueError("constraint field 'max_line_segments' must be 1.")
    else:
        key = {
            "pack_nodes": "unpack",
            "max_coarse_node_splits": "max_splits",
            "allowed_excess_dists_in_coarse_nodes": "allowable_excess",
        }[kind]
        number = _constraint_integer(spec, key)
        if kind != "pack_nodes" and number < 0:
            raise ValueError(f"constraint field {key!r} must be non-negative.")
    return cast(ConstraintSpec, spec)


# Accepted constraint input forms for each RunSpec dataclass.
ConstraintsLike = Constraints | Sequence[ConstraintSpec] | ConstraintSpec | None


def constraint_specs(value: ConstraintsLike, engine: str) -> list[ConstraintSpec]:
    """Normalize and validate constraint specifications for one runner engine.

    ``Constraints`` builders, individual dictionaries, dictionary sequences, and ``None`` all pass
    through this function before a run starts. Returned dictionaries are deep copies, so
    normalization neither mutates nor aliases caller-owned data.

    Args:
        value (Constraints | Sequence[dict] | dict | None): Constraint input to normalize. ``None``
            and an empty dictionary produce an empty list.
        engine (str): Runner engine name: ``"recom"``, ``"smc"``, or ``"forest"``.

    Returns:
        list[dict]: Copied constraint specifications in their original order.

    Raises:
        ValueError: If a specification has an unknown constraint name or the selected engine does
            not support that constraint type.

    Note:
        Validation here covers document shape, value domains, and engine compatibility. Input-data
        columns are still checked by the selected engine.
    """
    if value is None:
        return []
    if isinstance(value, Constraints):
        raw_items = value.specs()
    elif isinstance(value, dict):
        raw_items = [value] if value else []
    else:
        raw_items = value
    items = [validate_constraint_spec(item) for item in raw_items]

    supported = sorted(name for name, engines in ENGINE_SUPPORT.items() if engine in engines)
    for item in items:
        name = item["constraint"]
        if engine not in ENGINE_SUPPORT[name]:
            raise ValueError(
                f"The {engine} runner does not support the {name!r} constraint "
                f"(available on: {', '.join(ENGINE_SUPPORT[name])}). The {engine} "
                f"runner supports: {', '.join(supported)}."
            )
    return items
