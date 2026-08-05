import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast, get_args, overload

from ..constraints import ConstraintsLike, ConstraintSpec, constraint_specs
from ..objectives import ObjectiveSpec, validate_objective_spec
from ..run_config import (
    check_boolean,
    check_finite_nonnegative,
    check_finite_number,
    check_nonempty_string,
    check_positive_int,
    check_rng_seed,
    check_string_list,
    check_unit_interval,
)
from ..run_container import (
    RunnerConfig,
    RunSpec,
    _resolve_output_name,
    _validate_output_file_name,
)

ChainVariant = Literal[
    "A",
    "B",
    "C",
    "D",
    "R",
    "AW",
    "BW",
    "cut-edges-mst",
    "district-pairs-mst",
    "cut-edges-ust",
    "district-pairs-ust",
    "reversible",
    "cut-edges-region-aware",
    "district-pairs-region-aware",
]
"""A chain proposal variant: a letter code or its rustrecom spelling (normalized to the letter)."""

OptimizerVariant = Literal[
    "A",
    "B",
    "C",
    "D",
    "cut-edges-mst",
    "district-pairs-mst",
    "cut-edges-ust",
    "district-pairs-ust",
]
"""An optimizer proposal variant; the optimizers support neither reversible nor the region-aware
spellings (region-aware sampling comes from supplying ``region_weights`` with A or B)."""

Writer = Literal[
    "tsv",
    "jsonl",
    "pcompress",
    "jsonl-full",
    "assignments",
    "canonicalized-assignments",
    "canonical",
    "ben",
    "bendl",
]
"""A rustrecom output writer name."""

AcceptRule = Literal["fixed", "linear", "exponential"]
"""A tilted-run acceptance rule for worsening proposals."""

# Letter codes are the established shorthand; the long names match the rustrecom CLI.
VARIANT_TO_RUSTRECOM = {
    "A": "cut-edges-mst",
    "B": "district-pairs-mst",
    "C": "cut-edges-ust",
    "D": "district-pairs-ust",
    "R": "reversible",
    "AW": "cut-edges-region-aware",
    "BW": "district-pairs-region-aware",
}
RUSTRECOM_TO_VARIANT = {name: letter for letter, name in VARIANT_TO_RUSTRECOM.items()}

WRITERS: tuple[str, ...] = get_args(Writer)


def _checked_chain_variant(variant: str) -> ChainVariant:
    """Normalize a chain variant to its letter code, rejecting unknown spellings.

    Applied at construction and again at config build, so a post-construction ``variant``
    assignment cannot bypass validation.
    """
    normalized = RUSTRECOM_TO_VARIANT.get(variant, variant)
    if normalized not in VARIANT_TO_RUSTRECOM:
        raise ValueError(
            f"Unknown variant {variant!r}. Choose one of A, B, C, D, "
            "R (reversible), AW (cut edges region aware), BW (district pairs "
            f"region aware), or the rustrecom names: {', '.join(RUSTRECOM_TO_VARIANT)}."
        )
    return cast(ChainVariant, normalized)


def _checked_optimizer_variant(variant: str) -> OptimizerVariant:
    """Normalize an optimizer variant to its letter code, rejecting unknown spellings.

    Applied at construction and again at config build, so a post-construction ``variant``
    assignment cannot bypass validation.
    """
    normalized = RUSTRECOM_TO_VARIANT.get(variant, variant)
    if normalized not in OPTIMIZER_VARIANTS:
        raise ValueError(
            f"Unknown optimizer variant {variant!r}. Choose one of A, B, C, D "
            "(region-aware sampling comes from supplying 'region_weights' with A or B), "
            f"or the rustrecom names: "
            f"{', '.join(VARIANT_TO_RUSTRECOM[letter] for letter in OPTIMIZER_VARIANTS)}."
        )
    return cast(OptimizerVariant, normalized)


def _checked_writer(writer: str) -> Writer:
    """Reject unknown writers, at construction and again at config build."""
    if writer not in WRITERS:
        raise ValueError(f"Unknown writer {writer!r}. Choose one of: {', '.join(WRITERS)}.")
    return cast(Writer, writer)


class RecomChainConfig(TypedDict):
    """Validated rustrecom chain configuration sent to the container.

    Attributes:
        version (Literal[1]): Native configuration schema version.
        command (str): Rustrecom subcommand, currently ``"chain"``.
        graph_json (str): Container path to the dual-graph JSON file.
        n_steps (int): Number of chain steps.
        tol (float): Permitted relative population deviation.
        pop_col (str): Node column containing districting population.
        assignment_col (str): Node column containing the initial assignment.
        rng_seed (int): Random-number-generator seed.
        variant (str): Native rustrecom proposal-variant name.
        target_pop (int | None): Explicit ideal district population, or None to derive it.
        balance_ub (int): Reversible-variant balance upper bound.
        n_threads (int): Number of proposal-generation threads.
        batch_size (int): Number of proposals generated per batch.
        writer (Writer): Chain-record output format.
        sum_cols (list[str]): Node columns to aggregate in JSON outputs.
        region_weights (dict[str, float]): Edge surcharges keyed by region column.
        edge_weight_keys (list[str]): Edge columns added to spanning-tree weights.
        cut_edges_count (bool): Whether to include the cut-edge count in output.
        output_file (str | None): Container output path, or None to write to stdout.
        bendl_graph_order (str): Graph ordering used before BENDL encoding.
        show_progress (bool): Whether rustrecom writes progress to stderr.
        constraint (ConstraintSpec, optional): Validated rustrecom constraint.
    """

    version: Literal[1]
    command: str
    graph_json: str
    n_steps: int
    tol: float
    pop_col: str
    assignment_col: str
    rng_seed: int
    variant: str
    target_pop: int | None
    balance_ub: int
    n_threads: int
    batch_size: int
    writer: Writer
    sum_cols: list[str]
    region_weights: dict[str, float]
    edge_weight_keys: list[str]
    cut_edges_count: bool
    output_file: str | None
    bendl_graph_order: str
    show_progress: bool
    constraint: NotRequired[ConstraintSpec]


class OptimizerConfig(TypedDict):
    """Validated rustrecom optimizer configuration sent to the container.

    Attributes:
        version (Literal[1]): Native configuration schema version.
        command (str): Optimizer subcommand, ``"short-bursts"`` or ``"tilted"``.
        graph_json (str): Container path to the dual-graph JSON file.
        n_steps (int): Total number of proposals.
        tol (float): Permitted relative population deviation.
        pop_col (str): Node column containing districting population.
        assignment_col (str): Node column containing the initial assignment.
        rng_seed (int): Random-number-generator seed.
        objective (ObjectiveSpec): Validated objective driving the optimizer.
        maximize (bool): Whether larger objective values are preferred.
        n_threads (int): Number of proposal-generation threads.
        variant (str): Native rustrecom proposal-variant name.
        writer (Writer): Chain-record output format.
        sum_cols (list[str]): Node columns aggregated over districts.
        partial_sum_cols (list[str]): Sum columns whose missing node values count as zero.
        region_weights (dict[str, float]): Edge surcharges keyed by region column.
        edge_weight_keys (list[str]): Edge columns added to spanning-tree weights.
        output_file (str): Container path for chain records.
        scores_output_file (str): Container path for optimizer scores.
        show_progress (bool): Whether rustrecom writes progress to stderr.
        write_improved_scores_only (bool): Whether to record only new global-best scores.
        burst_length (int, optional): Accepted steps per short burst.
        accept_rule (AcceptRule, optional): Tilted-run rule for worsening proposals.
        accept_worse_prob (float, optional): Fixed-rule probability of accepting a worse proposal.
        acceptance_beta (float, optional): Linear or exponential tilt strength.
    """

    version: Literal[1]
    command: str
    graph_json: str
    n_steps: int
    tol: float
    pop_col: str
    assignment_col: str
    rng_seed: int
    objective: ObjectiveSpec
    maximize: bool
    n_threads: int
    variant: str
    writer: Writer
    sum_cols: list[str]
    partial_sum_cols: list[str]
    region_weights: dict[str, float]
    edge_weight_keys: list[str]
    output_file: str
    scores_output_file: str
    show_progress: bool
    write_improved_scores_only: bool
    burst_length: NotRequired[int]
    accept_rule: NotRequired[AcceptRule]
    accept_worse_prob: NotRequired[float]
    acceptance_beta: NotRequired[float]


RustRecomConfig = RecomChainConfig | OptimizerConfig


@dataclass
class RecomRunSpec(RunSpec):
    """Represents all of the settings that can be passed to the rustrecom Rust code.

    The settings are validated on construction, so mistakes (an unknown variant, region-aware
    variants without region weights, etc.) raise immediately rather than after the Docker container
    has started.

    Raises:
        ValueError: If a setting or constraint is invalid.
    """

    pop_col: str
    """The name of the column in the dual graph JSON file that contains the population data."""
    assignment_col: str
    """The name of the column in the dual graph JSON file that contains the assignment data."""
    variant: ChainVariant
    """The variant of the recom algorithm to be used. Options are A, B, C, D, R, AW, BW,
        or the equivalent rustrecom names (cut-edges-mst, district-pairs-mst, cut-edges-ust,
        district-pairs-ust, reversible, cut-edges-region-aware,
        district-pairs-region-aware)."""
    balance_ub: int = 0
    """The balance upper bound to be used in the rustrecom code. Only used in (R)eversible mode.
        The engine contract is an unsigned 32-bit integer (``pub balance_ub: u32``). Defaults to
        0."""
    n_steps: int = 10
    """The number of steps that the rustrecom code should run for. Defaults to 10."""
    pop_tol: float = 0.05
    """The population tolerance to be used in the recom code. Defaults to 0.05."""
    target_pop: int | None = None
    """The target district population. When None, rustrecom derives it as
        total population / number of districts."""
    n_threads: int = 1
    """The number of proposal threads. Defaults to 1."""
    batch_size: int = 1
    """The proposal batch size. Defaults to 1."""
    writer: Writer = "canonical"
    """The type of writer that should be used to write the output of the rustrecom code. Options
    are:

    - tsv
    - jsonl
    - pcompress
    - jsonl-full
    - assignments
    - canonicalized-assignments
    - canonical
    - ben
    - bendl (self-describing single file: graph + metadata + BEN stream)

    Defaults to ``"canonical"``.
    """
    sum_cols: list[str] = field(default_factory=list)
    """The columns that should be summed in the output of the rustrecom code. This will only
        be shown if the writer is set to jsonl or jsonl-full. Defaults to an empty list."""
    region_weights: dict[str, float] = field(default_factory=dict)
    """A dictionary of surcharges to be added to edges between regions. This is only used
        in the AW and BW variants of the rustrecom code. Defaults to an empty mapping."""
    edge_weight_keys: list[str] = field(default_factory=list)
    """Per-edge attribute columns added to edge weights in MST / region-aware
        spanning-tree sampling (A, B, AW, BW variants). Defaults to an empty list."""
    constraint: ConstraintsLike = None
    """A constraint for the run: either a Constraints builder (see
        gerrytools.mgrp.Constraints) or a raw spec dict passed to rustrecom as inline
        JSON. rustrecom currently accepts a single constraint per run, e.g.::

            Constraints().district_share_floor(
                numerator_col="BVAP", denominator_cols=["VAP"], threshold=0.4
            )

        Defaults to None.
    """
    cut_edges_count: bool = False
    """If true, rustrecom outputs the cut-edge count at each step. Defaults to False."""
    bendl_graph_order: str = "none"
    """Graph reordering applied before the chain runs, for better BENDL stream
        compression. Only valid with the bendl writer. Options are 'none', 'rcm',
        'mlc', or 'key:<attr>'. Defaults to ``"none"``."""
    show_progress: bool = False
    """If true, rustrecom renders a progress bar on stderr. Note that the runner captures
        stderr into the log file, so this mostly pollutes logs; useful with force_print. Defaults
        to False."""
    rng_seed: int = 42
    """The random number generator seed used by rustrecom. Defaults to 42."""
    force_print: bool = False
    """If true, the output of the rustrecom code will be printed to the console instead of
        being written to a file. Defaults to False."""
    updaters: dict[str, Callable] = field(default_factory=dict)
    """A dictionary of updaters that should be used when running the chain using the
        mcmc_run_with_updaters method. Defaults to an empty mapping."""

    @property
    def resolved_constraint(self) -> ConstraintSpec | None:
        """Return the normalized constraint spec derived from ``constraint``, if any.

        Raises:
            ValueError: If more than one constraint is supplied or a specification is invalid.
        """
        specs = constraint_specs(self.constraint, "recom")
        if len(specs) > 1:
            raise ValueError(
                f"rustrecom accepts a single constraint per run; got {len(specs)} constraint specs."
            )
        return specs[0] if specs else None

    def stem(self) -> str:
        """Return the human-readable stem derived from the run's headline settings.

        Returns:
            str: Stem containing the variant, assignment column, seed, and step count.
        """
        return f"Recom{self.variant}_{self.assignment_col}_{self.rng_seed}_{self.n_steps}"

    def output_name(self, stem: str | None = None) -> str | None:
        """The name of the run's output file, or None when it prints to stdout.

        Args:
            stem (str | None, optional): Stem to build the name from. Defaults to :meth:`stem`.

        Returns:
            str | None: Output file name, or None when output is printed to standard output.
        """
        if self.force_print:
            return None
        return _resolve_output_name(stem or self.stem(), self.writer)

    def scores_name(self, stem: str | None = None) -> str | None:
        """Return None because chain runs produce no scores CSV.

        Args:
            stem (str | None, optional): Ignored output stem. Defaults to None.

        Returns:
            None: Chain runs do not produce a scores file.
        """
        return None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate current settings before construction or config emission.

        Raises:
            TypeError: If ``balance_ub`` is not an integer.
            ValueError: If a setting is outside the range or combinations accepted by rustrecom.
        """
        # Normalize the rustrecom long names to their letter codes so downstream
        # naming (output files, logs) is consistent.
        self.variant = _checked_chain_variant(self.variant)
        self.writer = _checked_writer(self.writer)
        check_nonempty_string("pop_col", self.pop_col)
        check_nonempty_string("assignment_col", self.assignment_col)
        _validate_output_file_name(self.assignment_col, field="assignment_col")
        check_string_list("sum_cols", self.sum_cols)
        check_string_list("edge_weight_keys", self.edge_weight_keys)
        if isinstance(self.balance_ub, bool) or not isinstance(self.balance_ub, int):
            raise TypeError("balance_ub must be an integer.")
        if not 0 <= self.balance_ub <= 2**32 - 1:
            raise ValueError("balance_ub must be between 0 and 2**32 - 1.")
        check_positive_int("n_steps", self.n_steps)
        check_positive_int("n_threads", self.n_threads)
        check_positive_int("batch_size", self.batch_size)
        check_finite_nonnegative("pop_tol", self.pop_tol)
        check_rng_seed("rng_seed", self.rng_seed, minimum=0, maximum=2**64 - 1)
        check_boolean("cut_edges_count", self.cut_edges_count)
        check_boolean("show_progress", self.show_progress)
        check_boolean("force_print", self.force_print)
        if self.target_pop is not None:
            check_positive_int("target_pop", self.target_pop)
        if self.variant == "R" and self.balance_ub <= 0:
            raise ValueError("The reversible variant ('R') requires balance_ub > 0.")
        if self.variant in ("AW", "BW") and not self.region_weights:
            raise ValueError(
                f"variant={self.variant!r} requires 'region_weights' to be specified, "
                "e.g. region_weights={'COUNTY': 2.0}."
            )
        if not isinstance(self.region_weights, dict):
            raise ValueError(
                f"region_weights must be a dictionary, but found {self.region_weights!r}."
            )
        for region, weight in self.region_weights.items():
            check_nonempty_string("region_weights key", region)
            check_finite_number(f"region_weights[{region!r}]", weight)
        if self.edge_weight_keys and self.variant not in ("A", "B", "AW", "BW"):
            raise ValueError(
                "The 'edge_weight_keys' option is only valid with the MST "
                "and region-aware variants (A, B, AW, BW)."
            )
        order = self.bendl_graph_order
        known_order = order in ("none", "rcm", "mlc") or (
            isinstance(order, str) and order.startswith("key:") and len(order) > len("key:")
        )
        if not known_order:
            raise ValueError(
                f"Unknown bendl_graph_order {order!r}. Choose 'none', 'rcm', 'mlc', "
                "or 'key:<attr>' with a nonempty node-attribute name."
            )
        if self.bendl_graph_order != "none" and self.writer != "bendl":
            raise ValueError("The 'bendl_graph_order' option is only valid with the bendl writer.")
        self.validate_force_print(self.writer, self.force_print)
        # Eager validation; the property recomputes from the live ``constraint`` on access.
        _ = self.resolved_constraint


# The optimizer CLIs accept only the four base variants; region-aware sampling is reached by
# supplying region_weights on A or B, and reversible is not supported by the optimizers at all.
OPTIMIZER_VARIANTS = ("A", "B", "C", "D")


@dataclass(kw_only=True)
class OptimizerRunSpecBase(RunSpec, ABC):
    """Settings shared by the ``short-bursts`` and ``tilted`` optimizer runs.

    All fields are keyword-only; construct the concrete run specs with named arguments.

    Raises:
        ValueError: If a shared optimizer setting or objective is invalid.
    """

    pop_col: str
    """The name of the column in the dual graph JSON file that contains the population data."""
    assignment_col: str
    """The name of the column in the dual graph JSON file that contains the assignment data."""
    objective: ObjectiveSpec
    """The objective spec driving the optimizer. Build one with
        :class:`~gerrytools.mgrp.Objective`, e.g.
        ``Objective.gingles_partial(threshold=0.5, min_pop="BVAP", total_pop="VAP")``, or pass
        the equivalent raw dict."""
    n_steps: int = 10
    """The total number of proposals to generate. Defaults to 10."""
    pop_tol: float = 0.05
    """The population tolerance used when drawing districts. Defaults to 0.05."""
    maximize: bool = True
    """If True, maximize the objective; if False, minimize it. Defaults to True."""
    variant: OptimizerVariant = "B"
    """The proposal variant: A, B, C, or D, or an equivalent name. Defaults to ``"B"``."""
    n_threads: int = 1
    """The number of threads used to generate proposals. Defaults to 1."""
    writer: Writer = "canonical"
    """The chain-record writer. Same options as RecomRunSpec.writer. Defaults to ``"canonical"``."""
    sum_cols: list[str] = field(default_factory=list)
    """Additional columns to sum over districts. Defaults to an empty list."""
    partial_sum_cols: list[str] = field(default_factory=list)
    """Sum columns that may be missing on some nodes. Defaults to an empty list."""
    region_weights: dict[str, float] = field(default_factory=dict)
    """Edge surcharges between regions. Defaults to an empty mapping."""
    edge_weight_keys: list[str] = field(default_factory=list)
    """Per-edge attributes added to edge weights. Defaults to an empty list."""
    write_improved_scores_only: bool = False
    """Whether to record only globally improving scores. Defaults to False."""
    show_progress: bool = False
    """Whether to render a progress bar on stderr. Defaults to False."""
    rng_seed: int = 42
    """The random number generator seed. Defaults to 42."""
    output_file_name: str | None = None
    """Override for the output file name; derived from the run settings when None."""

    @property
    def resolved_objective(self) -> ObjectiveSpec:
        """The validated objective spec derived from the current ``objective``."""
        return validate_objective_spec(self.objective)

    @abstractmethod
    def stem(self) -> str:
        """Return the human-readable stem derived from the run's headline settings.

        Returns:
            str: File-name stem for the optimizer run.
        """

    def output_name(self, stem: str | None = None) -> str:
        """The name of the run's output file.

        Args:
            stem (str | None, optional): Stem to build the name from. Defaults to :meth:`stem`.
                Ignored when ``output_file_name`` overrides the name entirely.

        Returns:
            str: Optimizer output file name.
        """
        return _resolve_output_name(stem or self.stem(), self.writer, self.output_file_name)

    def scores_name(self, stem: str | None = None) -> str:
        """The name of the per-step/per-burst scores CSV the run produces.

        Args:
            stem (str | None, optional): Stem to build the name from. Defaults to :meth:`stem`.

        Returns:
            str: Scores CSV file name.
        """
        output_stem = Path(self.output_name(stem)).stem
        return f"{output_stem}_scores.csv"

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate current shared optimizer settings.

        Raises:
            ValueError: If a setting is outside the range or combinations accepted by rustrecom.
        """
        self.variant = _checked_optimizer_variant(self.variant)
        self.writer = _checked_writer(self.writer)
        check_nonempty_string("pop_col", self.pop_col)
        check_nonempty_string("assignment_col", self.assignment_col)
        _validate_output_file_name(self.assignment_col, field="assignment_col")
        check_string_list("sum_cols", self.sum_cols)
        check_string_list("partial_sum_cols", self.partial_sum_cols)
        check_string_list("edge_weight_keys", self.edge_weight_keys)
        if self.edge_weight_keys and self.variant not in ("A", "B"):
            raise ValueError(
                "The 'edge_weight_keys' option is only valid with the MST variants (A, B)."
            )
        if not isinstance(self.region_weights, dict):
            raise ValueError(
                f"region_weights must be a dictionary, but found {self.region_weights!r}."
            )
        for region, weight in self.region_weights.items():
            check_nonempty_string("region_weights key", region)
            check_finite_number(f"region_weights[{region!r}]", weight)
        if self.output_file_name is not None:
            _validate_output_file_name(self.output_file_name)
        check_positive_int("n_steps", self.n_steps)
        check_positive_int("n_threads", self.n_threads)
        check_finite_nonnegative("pop_tol", self.pop_tol)
        check_rng_seed("rng_seed", self.rng_seed, minimum=0, maximum=2**64 - 1)
        check_boolean("maximize", self.maximize)
        check_boolean("write_improved_scores_only", self.write_improved_scores_only)
        check_boolean("show_progress", self.show_progress)
        # Eager validation; the property recomputes from the live ``objective`` on access.
        _ = self.resolved_objective


@dataclass(kw_only=True)
class ShortBurstsRunSpec(OptimizerRunSpecBase):
    """
    Represents the settings for a ``rustrecom short-bursts`` optimizer run.

    Short bursts (Cannon et al. 2020, `arXiv:2011.02288 <https://arxiv.org/abs/2011.02288>`_) is
    an MCMC-with-restart heuristic: the chain accepts every valid proposal, and after every
    ``burst_length`` accepted steps it snaps back to the best-scoring plan seen during the burst
    (including the burst's starting plan) and begins a new burst from there. Short bursts therefore
    explore aggressively within a burst while ratcheting toward better scores across bursts. The
    score being chased is the ``objective`` spec, built with :class:`~gerrytools.mgrp.Objective`
    (see that class for the catalog of available objectives), and the direction is set by
    ``maximize``.

    The run writes two files: the chain records under the usual output-naming scheme
    (``SB<variant>_<assignment_col>_<seed>_<steps>_<burst>_<config hash>``), and a
    ``*_scores.csv`` next to it with one row per burst (step number, overall score, and
    per-district scores where the objective provides them). The settings are validated on
    construction, so mistakes raise before the Docker container starts. All settings are
    keyword-only; the shared optimizer settings (columns, objective, variant, writer, and output
    controls) are common to both optimizer run specs.

    Raises:
        ValueError: If ``burst_length`` or a shared optimizer setting is invalid.
    """

    burst_length: int
    """The number of accepted steps per short burst."""

    def stem(self) -> str:
        """Return the human-readable stem derived from the run's headline settings.

        Returns:
            str: Stem containing the variant, assignment column, seed, steps, and burst length.
        """
        return (
            f"SB{self.variant}_{self.assignment_col}"
            f"_{self.rng_seed}_{self.n_steps}_{self.burst_length}"
        )

    def validate(self) -> None:
        """Validate current short-bursts settings.

        Raises:
            ValueError: If ``burst_length`` or a shared optimizer setting is invalid.
        """
        check_positive_int("burst_length", self.burst_length)
        super().validate()


@dataclass(kw_only=True)
class TiltedRunSpec(OptimizerRunSpecBase):
    """
    Represents the settings for a ``rustrecom tilted`` optimizer run.

    A tilted run is a single continuous chain that always accepts proposals improving the
    objective score and accepts worsening proposals with a probability set by ``accept_rule``:

    - ``"fixed"``: accept with constant probability ``accept_worse_prob``, independent of how
      much worse the proposal is. ``0.0`` is pure hill-climbing, ``1.0`` a random walk. This
      matches GerryChain's ``SingleMetricOptimizer.tilted_run()``.
    - ``"linear"`` (default): accept with probability ``max(0, 1 - beta * score_loss)`` using
      ``acceptance_beta`` (default 1.0), so small losses are usually tolerated and large ones
      rejected.
    - ``"exponential"``: accept with probability ``exp(beta * delta)`` where ``delta <= 0`` is
      the signed score change, i.e. a Metropolis-style rule; larger ``acceptance_beta`` makes the
      chain pickier, ``0`` reduces to a random walk.

    Unlike short bursts, the chain never resets to a best-so-far plan, which makes it useful for
    exploring the neighborhood of good plans rather than converging hard. The score being chased
    is the ``objective`` spec, built with :class:`~gerrytools.mgrp.Objective` (see that class for
    the catalog of available objectives), and the direction is set by ``maximize``.

    The run writes two files: the chain records under the usual output-naming scheme
    (``Tilted<variant>_<rule>_<assignment_col>_<seed>_<steps>_<config hash>``), and a
    ``*_scores.csv`` next to it with one row per step. The settings are validated on construction,
    so mistakes raise before the Docker container starts. All settings are keyword-only; the shared
    optimizer settings (columns, objective, variant, writer, and output controls) are common to both
    optimizer run specs.

    Raises:
        ValueError: If an acceptance-rule option or shared optimizer setting is invalid.
    """

    accept_rule: AcceptRule = "linear"
    """Acceptance rule for worsening proposals. Defaults to ``"linear"``."""
    accept_worse_prob: float | None = None
    """Acceptance probability for worsening proposals; required by (and only valid
        with) the 'fixed' rule. Defaults to None."""
    acceptance_beta: float | None = None
    """Tilt strength for the 'linear' and 'exponential' rules; defaults to 1.0 engine-side."""

    def stem(self) -> str:
        """Return the human-readable stem derived from the run's headline settings.

        Returns:
            str: Stem containing the variant, acceptance rule, assignment column, seed, and steps.
        """
        return (
            f"Tilted{self.variant}_{self.accept_rule}_{self.assignment_col}"
            f"_{self.rng_seed}_{self.n_steps}"
        )

    def validate(self) -> None:
        """Validate current tilted-run settings.

        Raises:
            ValueError: If the acceptance-rule options or shared optimizer settings are invalid.
        """
        if self.accept_rule not in ("fixed", "linear", "exponential"):
            raise ValueError(
                f"Unknown accept_rule {self.accept_rule!r}. Choose one of "
                "'fixed', 'linear', or 'exponential'."
            )
        if self.accept_rule == "fixed":
            if self.accept_worse_prob is None:
                raise ValueError("accept_rule='fixed' requires accept_worse_prob.")
            if self.acceptance_beta is not None:
                raise ValueError(
                    "acceptance_beta is only valid with the 'linear' and 'exponential' rules."
                )
        elif self.accept_worse_prob is not None:
            raise ValueError("accept_worse_prob is only valid with the 'fixed' rule.")
        if self.accept_worse_prob is not None:
            check_unit_interval("accept_worse_prob", self.accept_worse_prob)
        if self.acceptance_beta is not None:
            check_finite_nonnegative("acceptance_beta", self.acceptance_beta)
        super().validate()


OptimizerRunSpec = ShortBurstsRunSpec | TiltedRunSpec
RustRecomRunSpec = RecomRunSpec | ShortBurstsRunSpec | TiltedRunSpec

# Canonical rustrecom CLI subcommand for each run-spec type. Keyed by exact type so instance
# data can never shadow the token interpolated into the sh -c command template.
RUN_SPEC_SUBCOMMANDS: dict[type[RunSpec], str] = {
    RecomRunSpec: "chain",
    ShortBurstsRunSpec: "short-bursts",
    TiltedRunSpec: "tilted",
}


def _subcommand(run_spec: RustRecomRunSpec) -> str:
    """The fixed CLI subcommand for ``run_spec``'s exact type."""
    try:
        return RUN_SPEC_SUBCOMMANDS[type(run_spec)]
    except KeyError:
        raise TypeError(
            f"No rustrecom subcommand for {type(run_spec).__name__}; expected exactly "
            "RecomRunSpec, ShortBurstsRunSpec, or TiltedRunSpec."
        ) from None


class RecomRunnerConfig(RunnerConfig[RustRecomRunSpec]):
    """
    Represents the configuration for a RunnerSession which is used to run the
    rustrecom code on a given dual graph within the docker container.
    """

    run_spec_type = (RecomRunSpec, ShortBurstsRunSpec, TiltedRunSpec)

    def __init__(
        self,
        json_file_path: str,
        output_folder: str = "./output",
        log_folder: str = "./logs",
    ):
        """
        Initializes the RecomRunnerConfig object.

        Args:
            json_file_path (str): The path to the dual graph JSON file that should be
                used in rustrecom.
            output_folder (str): The directory where the output files should be
                written to. Defaults to "./output".
            log_folder (str): The directory where the log files should be written
                    to. Defaults to "./logs".
        """
        super().__init__("recom", json_file_path, output_folder, log_folder)

    def run_command(self, run_spec: RustRecomRunSpec) -> list:
        """Return the command for a rustrecom run.

        The template is static per run-spec type: the subcommand token comes from
        ``RUN_SPEC_SUBCOMMANDS``, keyed by exact run-spec type, never from instance
        data (see ``RunnerConfig._shell_command`` for the config-as-``$1``
        transport). ``--overwrite-output`` is an output-lifecycle flag rather than
        a sampler value; the CLI accepts it alongside ``--config`` and it preserves
        the clobbering behavior of the shell redirects this template replaced.

        Args:
            run_spec (RustRecomRunSpec): Chain or optimizer run settings.

        Returns:
            list: Shell command arguments for the container.

        Raises:
            TypeError: If ``run_spec`` is not one of the supported concrete run-spec types.
        """
        self._check_run_spec(run_spec)
        template = (
            ". /root/.cargo/env; /usr/bin/time -v rustrecom "
            f'{_subcommand(run_spec)} --config "$1" --overwrite-output'
        )
        return self._shell_command(template, self.run_config(run_spec), argv0="rustrecom")

    @overload
    def run_config(self, run_spec: RecomRunSpec) -> RecomChainConfig: ...

    @overload
    def run_config(self, run_spec: OptimizerRunSpec) -> OptimizerConfig: ...

    def run_config(self, run_spec: RustRecomRunSpec) -> RustRecomConfig:
        """Return the complete effective configuration for a ReCom run.

        Unlike the Forest and SMC runners, which use gerrytools' own engine
        envelope, this emits rustrecom's native chain config: a flat document
        whose fields mirror the ``rustrecom chain`` CLI arguments, with a
        ``version``/``command`` envelope. Every field is written explicitly
        (rather than relying on the CLI's defaults) so the stored provenance
        is self-contained. The letter variant codes are gerrytools shorthand,
        so they translate to the CLI spellings here.

        Args:
            run_spec (RustRecomRunSpec): Chain or optimizer run settings.

        Returns:
            RustRecomConfig: Independent native rustrecom configuration.

        Raises:
            TypeError: If ``run_spec`` is not one of the supported concrete run-spec types.
            ValueError: If its settings are invalid.
        """
        self._check_run_spec(run_spec)
        return self._config_document(run_spec, self.file_stem(run_spec))

    def _base_config(self, run_spec: RustRecomRunSpec) -> RustRecomConfig:
        """The config document with hash-free names, hashed into :meth:`file_stem`."""
        return self._config_document(run_spec, self._stem(run_spec))

    def _config_document(self, run_spec: RustRecomRunSpec, stem: str) -> RustRecomConfig:
        """The independent effective config, with file names built from ``stem``."""
        if not isinstance(run_spec, RecomRunSpec):
            return copy.deepcopy(self._optimizer_config(run_spec, stem))
        output_name = run_spec.output_name(stem)
        output = None if output_name is None else f"{self.container_output_dir}/{output_name}"
        constraint = run_spec.resolved_constraint
        config = RecomChainConfig(
            version=1,
            command=_subcommand(run_spec),
            graph_json=self.container_graph_path,
            n_steps=run_spec.n_steps,
            tol=run_spec.pop_tol,
            pop_col=run_spec.pop_col,
            assignment_col=run_spec.assignment_col,
            rng_seed=run_spec.rng_seed,
            # Rechecked so post-construction assignments cannot bypass __post_init__.
            variant=VARIANT_TO_RUSTRECOM[_checked_chain_variant(run_spec.variant)],
            target_pop=run_spec.target_pop,
            balance_ub=run_spec.balance_ub,
            n_threads=run_spec.n_threads,
            batch_size=run_spec.batch_size,
            writer=_checked_writer(run_spec.writer),
            sum_cols=run_spec.sum_cols,
            region_weights=run_spec.region_weights,
            edge_weight_keys=run_spec.edge_weight_keys,
            cut_edges_count=run_spec.cut_edges_count,
            output_file=output,
            bendl_graph_order=run_spec.bendl_graph_order,
            show_progress=run_spec.show_progress,
        )
        if constraint:
            config["constraint"] = constraint
        return copy.deepcopy(config)

    def _optimizer_config(self, run_spec: OptimizerRunSpec, stem: str) -> OptimizerConfig:
        """The native config document for a short-bursts or tilted run."""
        config = OptimizerConfig(
            version=1,
            command=_subcommand(run_spec),
            graph_json=self.container_graph_path,
            n_steps=run_spec.n_steps,
            tol=run_spec.pop_tol,
            pop_col=run_spec.pop_col,
            assignment_col=run_spec.assignment_col,
            rng_seed=run_spec.rng_seed,
            objective=run_spec.resolved_objective,
            maximize=run_spec.maximize,
            n_threads=run_spec.n_threads,
            # Rechecked so post-construction assignments cannot bypass __post_init__.
            variant=VARIANT_TO_RUSTRECOM[_checked_optimizer_variant(run_spec.variant)],
            writer=_checked_writer(run_spec.writer),
            sum_cols=run_spec.sum_cols,
            partial_sum_cols=run_spec.partial_sum_cols,
            region_weights=run_spec.region_weights,
            edge_weight_keys=run_spec.edge_weight_keys,
            output_file=f"{self.container_output_dir}/{run_spec.output_name(stem)}",
            scores_output_file=f"{self.container_output_dir}/{run_spec.scores_name(stem)}",
            show_progress=run_spec.show_progress,
            write_improved_scores_only=run_spec.write_improved_scores_only,
        )
        if isinstance(run_spec, ShortBurstsRunSpec):
            config["burst_length"] = run_spec.burst_length
        else:
            config["accept_rule"] = run_spec.accept_rule
            if run_spec.accept_worse_prob is not None:
                config["accept_worse_prob"] = run_spec.accept_worse_prob
            if run_spec.acceptance_beta is not None:
                config["acceptance_beta"] = run_spec.acceptance_beta
        return config

    def _stem(self, run_spec: RustRecomRunSpec) -> str:
        """The human-readable stem derived from the run's headline settings."""
        return run_spec.stem()

    def _output_name(self, run_spec: RustRecomRunSpec) -> str | None:
        """
        The name of the file the run will produce, or None when the output is
        printed to stdout instead.
        """
        return run_spec.output_name(self.file_stem(run_spec))

    def scores_file(self, run_spec: RustRecomRunSpec) -> str | None:
        """Return the optimizer's scores CSV path, or None for a chain run.

        Args:
            run_spec (RustRecomRunSpec): Chain or optimizer run settings.

        Returns:
            str | None: Host scores CSV path, or None for a chain run.
        """
        scores_name = run_spec.scores_name(self.file_stem(run_spec))
        return None if scores_name is None else str(self.output_folder / scores_name)

    def expected_files(self, run_spec: RustRecomRunSpec) -> list[str]:
        """Return outputs, adding sidecar provenance when the primary output lacks it.

        Args:
            run_spec (RustRecomRunSpec): Chain or optimizer run settings.

        Returns:
            list[str]: Primary output plus applicable provenance and scores sidecars.
        """
        expected = super().expected_files(run_spec)
        inline_provenance = isinstance(run_spec, RecomRunSpec) and run_spec.writer in (
            "jsonl",
            "jsonl-full",
        )
        if expected and run_spec.writer != "bendl" and not inline_provenance:
            expected.append(self._sidecar_file(expected[0], "metadata.jsonl"))
        scores_file = self.scores_file(run_spec)
        if scores_file is not None:
            expected.append(scores_file)
        return expected

    def canonical_stdout_command(self, run_spec: RustRecomRunSpec) -> list:
        """Return the chain command with canonical assignment output forced to stdout.

        Args:
            run_spec (RustRecomRunSpec): Chain run settings.

        Returns:
            list: Shell command arguments for the container.

        Raises:
            TypeError: If ``run_spec`` describes an optimizer rather than a chain.
        """
        if not isinstance(run_spec, RecomRunSpec):
            raise TypeError("Canonical stdout runs require a RecomRunSpec, not an optimizer run.")
        return self.run_command(
            replace(
                run_spec,
                writer="canonical",
                force_print=True,
                bendl_graph_order="none",
            )
        )
