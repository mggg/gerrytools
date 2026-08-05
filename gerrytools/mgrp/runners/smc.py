from dataclasses import dataclass, field
from typing import ClassVar, Literal, get_args

from ..constraints import ConstraintsLike, ConstraintSpec, constraint_specs
from ..run_config import (
    NOT_ENGINE_CONFIG,
    EngineRunConfig,
    build_run_config,
    check_boolean,
    check_finite_nonnegative,
    check_finite_number,
    check_nonempty_string,
    check_positive_int,
    check_rng_seed,
    check_string_list,
    check_unit_interval,
    dataclass_config,
)
from ..run_container import RunnerConfig, RunSpec, _resolve_output_name

SMCWriter = Literal["jsonl", "ben", "csv"]
"""An SMC output writer: standard JSONL, BEN, or the redist plans CSV."""

# The redist_map() portion of the config; everything else rides in the run section.
_MAP_FIELDS = ("pop_col", "n_dists", "pop_tol", "pop_bounds")


@dataclass
class SMCRunSpec(RunSpec):
    """
    Represents all of the settings for one Sequential Monte Carlo (SMC) run.

    The map fields (``pop_col``, ``n_dists``, ``pop_tol``, ``pop_bounds``) control the
    `redist_map() <https://alarm-redist.org/redist/reference/redist_map.html>`_ call in the
    `redist` R package; the remaining fields control
    `redist_smc() <https://alarm-redist.org/redist/reference/redist_smc.html>`_ and the
    output handling.

    Raises:
        ValueError: If a setting or constraint is invalid.
    """

    pop_col: str = field(metadata=NOT_ENGINE_CONFIG)
    """The name of the column in the shapefile that contains the population data.
        This will be used to derive the `total_pop` parameter in the `redist_map()`
        method."""
    n_dists: int = field(metadata=NOT_ENGINE_CONFIG)
    """The number of districts that the shapefile should be partitioned into."""
    n_sims: int
    """The number of samples to draw"""
    pop_tol: float = field(default=0.01, metadata=NOT_ENGINE_CONFIG)
    """The population tolerance used by `redist_map()`. Defaults to 0.01."""
    pop_bounds: list[int] = field(default_factory=list, metadata=NOT_ENGINE_CONFIG)
    """The population bounds to be used in the `redist_map()` method. This
        needs to be a list of three ints: [lower_bound, target, upper_bound]. Defaults to an
        empty list, which lets redist derive the bounds."""
    rng_seed: int = 42
    """The random number generator seed used by the SMC algorithm. Defaults to 42."""
    compactness: float = 1.0
    """The compactness parameter used by the SMC algorithm. Defaults to 1.0."""
    constraints: ConstraintsLike = field(default=None, metadata=NOT_ENGINE_CONFIG)
    """Constraints for the run, as a Constraints builder (see
        gerrytools.mgrp.Constraints). The smc runner supports group_hinge,
        group_power, status_quo, and splits. Defaults to None."""
    resample: bool = False
    """Whether to perform a final resampling step so that the generated plans can
        be used immediately. Defaults to False."""
    adapt_k_thresh: float = 0.985
    """The threshold value used in the heuristic to select a value of :math:`k_i` for
        each splitting iteration. Must be in the range [0, 1]. Defaults to 0.985."""
    seq_alpha: float = 0.5
    """The resampling weight adjustment in ``[0, 1]``. Defaults to 0.5."""
    pop_temper: float = 0.0
    """The strength of the automatic population tempering. If the algorithm gets stuck, then it
        is recommended that you start with values between 0.01-0.05. Defaults to 0.0."""
    final_infl: float = 1.0
    """A multiplier for the population constraint on the final iteration. Used to loosen the
        constraint when the sampler is getting stuck on the final split. Defaults to 1.0."""
    verbose: bool = False
    """Whether to log intermediate SMC information. Defaults to False."""
    silent: bool = False
    """Whether to suppress all diagnostic output. Defaults to False."""
    tally_columns: list[str] = field(default_factory=list)
    """A list of shapefile columns whose per-district totals should be tallied. With the
        jsonl or ben writers the tallies are written to a ``<output-stem>_tallies.csv``
        sidecar next to the output file; with the csv writer they are included in the
        plans CSV itself. Defaults to an empty list."""
    output_file_name: str | None = field(default=None, metadata=NOT_ENGINE_CONFIG)
    """The desired name of the output file. If not set, then the file name will be determined
        according to a set of heuristics."""
    writer: SMCWriter = field(default="jsonl", metadata=NOT_ENGINE_CONFIG)
    """The output writer: standard ``"jsonl"``, ``"ben"``, or the redist plans ``"csv"``
        (which skips the parser stage). Defaults to ``"jsonl"``."""

    @property
    def resolved_constraints(self) -> list[ConstraintSpec]:
        """The normalized constraint specs derived from the current ``constraints``."""
        return constraint_specs(self.constraints, "smc")

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate current settings before construction or config emission.

        Raises:
            ValueError: If any SMC setting or constraint is invalid.
        """
        check_nonempty_string("pop_col", self.pop_col)
        if not isinstance(self.pop_bounds, list) or not all(
            not isinstance(value, bool) and isinstance(value, int) and value >= 0
            for value in self.pop_bounds
        ):
            raise ValueError(
                f"pop_bounds must contain nonnegative integers, but found {self.pop_bounds!r}."
            )
        if len(self.pop_bounds) not in (0, 3):
            raise ValueError(
                "pop_bounds must be a list of three values "
                "[lower_bound, target, upper_bound], or empty to use pop_tol; "
                f"got {self.pop_bounds!r}."
            )
        if self.pop_bounds and not (self.pop_bounds[0] <= self.pop_bounds[1] <= self.pop_bounds[2]):
            raise ValueError(
                "pop_bounds must be ordered [lower_bound, target, upper_bound]; "
                f"got {self.pop_bounds!r}."
            )
        check_positive_int("n_dists", self.n_dists)
        check_positive_int("n_sims", self.n_sims)
        check_finite_nonnegative("pop_tol", self.pop_tol)
        check_unit_interval("adapt_k_thresh", self.adapt_k_thresh)
        check_unit_interval("seq_alpha", self.seq_alpha)
        check_finite_number("compactness", self.compactness)
        check_finite_number("pop_temper", self.pop_temper)
        check_finite_number("final_infl", self.final_infl)
        check_rng_seed("rng_seed", self.rng_seed, minimum=-(2**31 - 1), maximum=2**31 - 1)
        check_boolean("resample", self.resample)
        check_boolean("verbose", self.verbose)
        check_boolean("silent", self.silent)
        check_string_list("tally_columns", self.tally_columns)
        if self.writer not in get_args(SMCWriter):
            raise ValueError(
                f"Unknown writer {self.writer!r}. Choose one of: {', '.join(get_args(SMCWriter))}."
            )
        # Eager validation; the property recomputes from the live ``constraints`` on access.
        _ = self.resolved_constraints


class SMCRunnerConfig(RunnerConfig[SMCRunSpec]):
    """
    Represents the configuration for a RunnerSession which is used to run the
    Sequential Monte Carlo (SMC) algorithm on a shapefile within the
    docker container.
    """

    parser_name: ClassVar[str | None] = "smc_parser"
    run_spec_type = SMCRunSpec

    def __init__(
        self,
        shapefile_path: str,
        output_folder: str = "./output",
        log_folder: str = "./logs",
    ):
        """
        Initializes the SMCRunnerConfig object.

        Args:
            shapefile_path (str): The path to the shapefile bundle that should be used in
                the SMC algorithm, e.g. ``"./shapefiles/testing"``.
            output_folder (str): The directory where the output files should be written to.
                Defaults to "./output".
            log_folder (str): The directory where the log files should be written to.
                Defaults to "./logs".
        """
        super().__init__("smc", shapefile_path, output_folder, log_folder)

    def run_command(self, run_spec: SMCRunSpec) -> list:
        """Return the command for an SMC run.

        The template is a code constant (see ``RunnerConfig._shell_command`` for the
        config-as-``$1`` transport); the jsonl and ben writers pipe the redist output
        through the parser stage.

        Args:
            run_spec (SMCRunSpec): SMC run settings.

        Returns:
            list: Shell command arguments for the container.

        Raises:
            TypeError: If ``run_spec`` is not exactly an SMCRunSpec.
        """
        self._check_run_spec(run_spec)
        template = '/usr/bin/time -v Rscript /home/smc/cli/smc_cli.R --config "$1"'
        return self._shell_command(
            template,
            self.run_config(run_spec),
            with_parser=run_spec.writer in ("jsonl", "ben"),
        )

    def run_config(self, run_spec: SMCRunSpec) -> EngineRunConfig:
        """Return the complete effective configuration for an SMC run.

        Args:
            run_spec (SMCRunSpec): SMC run settings.

        Returns:
            EngineRunConfig: Independent version-1 engine configuration.

        Raises:
            TypeError: If ``run_spec`` is not exactly an SMCRunSpec.
        """
        self._check_run_spec(run_spec)
        return self._config_document(run_spec, self._output_name(run_spec))

    def _base_config(self, run_spec: SMCRunSpec) -> EngineRunConfig:
        """The config document with hash-free names, hashed into ``file_stem``."""
        # SMC never prints to stdout, so the name resolves directly (no force_print hook).
        output_name = _resolve_output_name(
            self._stem(run_spec), run_spec.writer, run_spec.output_file_name
        )
        return self._config_document(run_spec, output_name)

    def _config_document(self, run_spec: SMCRunSpec, output_name: str) -> EngineRunConfig:
        """The effective config, naming the container-side output ``output_name``."""
        map_info = {name: getattr(run_spec, name) for name in _MAP_FIELDS}
        run = dataclass_config(run_spec)

        return build_run_config(
            "smc",
            io={
                "graph": self.container_graph_path,
                "output": f"{self.container_output_dir}/{output_name}",
                "writer": run_spec.writer,
            },
            map_info=map_info,
            run=run,
            constraints=run_spec.resolved_constraints,
        )

    def _stem(self, run_spec: SMCRunSpec) -> str:
        """The human-readable stem derived from the run's headline settings."""
        return f"SMC_{run_spec.rng_seed}_{run_spec.n_sims}"

    def _output_name(self, run_spec: SMCRunSpec) -> str:
        """The name of the file the run will produce."""
        return _resolve_output_name(
            self.file_stem(run_spec), run_spec.writer, run_spec.output_file_name
        )

    def expected_files(self, run_spec: SMCRunSpec) -> list[str]:
        """Return the primary output and its promised SMC sidecars.

        Every run writes a ``*_metadata.jsonl`` provenance sidecar. The jsonl and ben
        writers add a ``*_tallies.csv`` when tally columns are requested; the csv writer
        keeps tallies in the plans CSV itself and adds a ``*_assignments.csv`` instead.

        Args:
            run_spec (SMCRunSpec): SMC run settings.

        Returns:
            list[str]: Primary output and applicable metadata, tally, or assignment sidecars.
        """
        expected = super().expected_files(run_spec)
        expected.append(self._sidecar_file(expected[0], "metadata.jsonl"))
        if run_spec.writer == "csv":
            expected.append(self._sidecar_file(expected[0], "assignments.csv"))
        elif run_spec.tally_columns:
            expected.append(self._sidecar_file(expected[0], "tallies.csv"))
        return expected
