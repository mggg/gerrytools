from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
    dataclass_config,
)
from ..run_container import RunInfo, RunnerConfig

ForestWriter = Literal["jsonl", "ben", "raw"]
"""A forest output writer: standard JSONL, BEN, or the engine's raw atlas output."""


@dataclass
class ForestRunInfo(RunInfo):
    """Settings passed to the Multi-Scale Map Sampler (MSMS) Julia code.

    Raises:
        ValueError: If a setting or constraint is invalid.
    """

    levels: list[str]
    """The hierarchy level column names, coarsest first (e.g. ``["county", "precinct"]``).
        MSMS accepts any number of levels; a single-entry list runs a flat (one-level)
        chain."""
    pop_col: str
    """The name of the column in the dual graph JSON file that contains the population data."""
    num_dists: int = 2
    """The number of districts that the dual graph should be partitioned into. Defaults to 2."""
    pop_dev: float = 0.1
    """The maximum allowable population deviation between districts. Defaults to 0.1."""
    constraints: ConstraintsLike = field(default=None, metadata=NOT_ENGINE_CONFIG)
    """Constraints for the run, as a Constraints builder (see
        gerrytools.mgrp.Constraints). The forest runner supports pack_nodes,
        max_coarse_node_splits, allowed_excess_dists_in_coarse_nodes, and
        max_discontinuous_traversal_segments. Defaults to None."""
    gamma: float = 0.0
    """The gamma value to be used in the MSMS code. Defaults to 0.0."""
    n_steps: int = 10
    """The number of steps that the MSMS code should run for. Defaults to 10."""
    rng_seed: int = 42
    """The random seed to be used in the MSMS code. Defaults to 42."""
    output_file_name: str | None = field(default=None, metadata=NOT_ENGINE_CONFIG)
    """The name of the output file that the MSMS code should write to. If None, then the
        output file name will be determined according to a set of heuristics."""
    writer: ForestWriter = field(default="jsonl", metadata=NOT_ENGINE_CONFIG)
    """The output writer: standard ``"jsonl"``, ``"ben"``, or the engine's ``"raw"``
        atlas output (which skips the parser stage). Defaults to ``"jsonl"``."""
    force_print: bool = field(default=False, metadata=NOT_ENGINE_CONFIG)
    """Whether or not the output should be printed to the console. This will overwrite the
        output_file_name attribute. Defaults to False."""
    updaters: dict[str, Callable] = field(default_factory=dict, metadata=NOT_ENGINE_CONFIG)
    """A dictionary of updaters that should be used when running the chain using the
        mcmc_run_with_updaters method. Defaults to an empty mapping."""

    @property
    def resolved_constraints(self) -> list[ConstraintSpec]:
        """The normalized constraint specs derived from the current ``constraints``."""
        return constraint_specs(self.constraints, "forest")

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate current settings before construction or config emission.

        Raises:
            ValueError: If any Forest setting or constraint is invalid.
        """
        check_string_list("levels", self.levels)
        if not self.levels:
            raise ValueError(
                "levels must be a non-empty list of column names, coarsest first, "
                f"e.g. ['county', 'precinct']; got {self.levels!r}."
            )
        check_nonempty_string("pop_col", self.pop_col)
        if self.writer not in get_args(ForestWriter):
            raise ValueError(
                f"Unknown writer {self.writer!r}. Choose one of: "
                f"{', '.join(get_args(ForestWriter))}."
            )
        check_positive_int("num_dists", self.num_dists)
        check_positive_int("n_steps", self.n_steps)
        check_finite_nonnegative("pop_dev", self.pop_dev)
        check_finite_number("gamma", self.gamma)
        check_rng_seed("rng_seed", self.rng_seed, minimum=0, maximum=2**63 - 1)
        check_boolean("force_print", self.force_print)
        self.validate_force_print(self.writer, self.force_print)
        # Eager validation; the property recomputes from the live ``constraints`` on access.
        _ = self.resolved_constraints


class ForestRunnerConfig(RunnerConfig[ForestRunInfo]):
    """
    Represents the configuration for a RunContainer which is used to
    run the Multi-Scale Map Sampler (MSMS) algorithm on a dual graph
    within the docker container.
    """

    parser_name: ClassVar[str | None] = "msms_parser"
    run_info_type = ForestRunInfo

    def __init__(
        self,
        json_file_path: str,
        output_folder: str = "./output",
        log_folder: str = "./logs",
    ):
        """
        Initializes the ForestRunnerConfig object.

        Args:
            json_file_path (str): The path to the dual graph JSON file that should be
                used in the MSMS algorithm.
            output_folder (str): The directory where the output files should be
                written to. Defaults to "./output".
            log_folder (str): The directory where the log files should be written
                    to. Defaults to "./logs".
        """
        super().__init__("forest", json_file_path, output_folder, log_folder)

    def run_command(self, run_info: ForestRunInfo) -> list:
        """Return the command for a Forest run.

        The template is a code constant (see ``RunnerConfig._shell_command`` for the
        config-as-``$1`` transport); the jsonl and ben writers pipe the engine's raw
        atlas output through the parser stage.

        Args:
            run_info (ForestRunInfo): Forest run settings.

        Returns:
            list: Shell command arguments for the container.

        Raises:
            TypeError: If ``run_info`` is not exactly a ForestRunInfo.
        """
        self._check_run_info(run_info)
        template = (
            'export JULIA_PROJECT="/home/forest"; '
            '/usr/bin/time -v julia /home/forest/cli/multi_cli.jl --config "$1"'
        )
        return self._shell_command(
            template,
            self.run_config(run_info),
            with_parser=run_info.writer in ("jsonl", "ben"),
        )

    def run_config(self, run_info: ForestRunInfo) -> EngineRunConfig:
        """Return the complete effective configuration for a Forest run.

        Args:
            run_info (ForestRunInfo): Forest run settings.

        Returns:
            EngineRunConfig: Independent version-1 engine configuration.

        Raises:
            TypeError: If ``run_info`` is not exactly a ForestRunInfo.
        """
        self._check_run_info(run_info)
        return self._config_document(run_info, self._output_name(run_info))

    def _base_config(self, run_info: ForestRunInfo) -> EngineRunConfig:
        """The config document with hash-free names, hashed into ``file_stem``."""
        output_name = self._writer_output_name(
            self._stem(run_info),
            run_info.writer,
            run_info.output_file_name,
            force_print=run_info.force_print,
        )
        return self._config_document(run_info, output_name)

    def _config_document(self, run_info: ForestRunInfo, output_name: str | None) -> EngineRunConfig:
        """The effective config, naming the container-side output ``output_name``."""
        run = dataclass_config(run_info)
        run.update(edge_weights="connections", output_freq=1)

        output = None if output_name is None else f"{self.container_output_dir}/{output_name}"
        return build_run_config(
            "forest",
            io={
                "graph": self.container_graph_path,
                "output": output,
                "writer": run_info.writer,
            },
            run=run,
            constraints=run_info.resolved_constraints,
        )

    def canonical_stdout_command(self, run_info: ForestRunInfo) -> list:
        """Return the run command with standard JSONL assignment output forced to stdout.

        Args:
            run_info (ForestRunInfo): Forest run settings.

        Returns:
            list: Shell command arguments for the container.

        Raises:
            TypeError: If ``run_info`` is not exactly a ForestRunInfo.
        """
        self._check_run_info(run_info)
        return self.run_command(replace(run_info, writer="jsonl", force_print=True))

    def _stem(self, run_info: ForestRunInfo) -> str:
        """The human-readable stem derived from the run's headline settings."""
        return f"Forest_{run_info.rng_seed}_atlas_gamma{run_info.gamma}_{run_info.n_steps}"

    def _output_name(self, run_info: ForestRunInfo) -> str | None:
        """
        The name of the file the run will produce, or None when the output is
        printed to stdout instead.
        """
        return self._writer_output_name(
            self.file_stem(run_info),
            run_info.writer,
            run_info.output_file_name,
            force_print=run_info.force_print,
        )

    def expected_files(self, run_info: ForestRunInfo) -> list[str]:
        """Return every output path, including provenance alongside file output.

        Args:
            run_info (ForestRunInfo): Forest run settings.

        Returns:
            list[str]: Primary output and metadata sidecar paths, or an empty list for stdout.
        """
        expected = super().expected_files(run_info)
        if expected:
            expected.append(self._sidecar_file(expected[0], "metadata.jsonl"))
        return expected
