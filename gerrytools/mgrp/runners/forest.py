from dataclasses import dataclass, field
import os
import json
from pathlib import Path
from typing import Optional
from .. import RunnerConfig
from typing import Callable, Dict


@dataclass
class ForestRunInfo:
    """
    Represents all of the settings that can be passed to the Multi Scale
    Map Sampler (MSMS) Julia code.
    """

    region_name: str
    """The name of the greater region to be used in the MSMS algorithm"""
    subregion_name: str
    """The name of the subregion to be used in the MSMS algorithm."""
    pop_col: str
    """The name of the column in the dual graph JSON file that contains the population data."""
    num_dists: int = 2
    """The number of districts that the dual graph should be partitioned into."""
    pop_dev: float = 0.1
    """The maximum allowable population deviation between the districts."""
    gamma: float = 0.0
    """The gamma value to be used in the MSMS code."""
    n_steps: int = 10
    """The number of steps that the MSMS code should run for."""
    rng_seed: int = 42
    """The random seed to be used in the MSMS code."""
    output_file_name: Optional[str] = None
    """The name of the output file that the MSMS code should write to. If None, then the 
        output file name will be determined according to a set of heuristics."""
    standard_jsonl: bool = True
    """Whether or not the output should be in the standard JSONL format."""
    ben: bool = False
    """Whether or not the output should be in the BEN format. Overrides standard_jsonl."""
    force_print: bool = False
    """Whether or not the output should be printed to the console. This will overwrite the
        output_file_name attribute."""
    updaters: Dict[str, Callable] = field(default_factory=dict)
    """A dictionary of updaters that should be used when running the chain using the 
        mcmc_run_with_updaters method."""


class ForestRunnerConfig(RunnerConfig):
    """
    Represents the configuration for a ForestReplicator which is used to
    run the Multi-Scale Map Sampler (MSMS) algorithm on a dual graph
    within the docker contianer.
    """

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
        self.json_dir = Path(json_file_path).parent
        self.json_name = Path(json_file_path).name
        json_name_without_extension = os.path.splitext(self.json_name)[0]
        self.output_folder = os.path.join(output_folder, json_name_without_extension)
        self.log_folder = os.path.abspath(log_folder)

    def configure_vols_and_name(self):
        """
        Configures the volumes and name for the docker container that will be
        used to run the MSMS algorithm.
        """

        current_directory = os.getcwd()

        volumes = {
            current_directory: {"bind": "/home/forest/runner", "mode": "rw"},
            os.path.abspath(self.json_dir): {
                "bind": "/home/forest/shapefiles",
                "mode": "rw",
            },
            os.path.abspath(self.output_folder): {
                "bind": f"/home/forest/output/{self.json_name[:-5]}",
                "mode": "rw",
            },
        }

        return {
            "name": f"forest_runner",
            "volumes": volumes,
        }

    def run_command(self, run_info: ForestRunInfo):
        """
        Constructs the command that will be used to run the MSMS algorithm in the
        docker container.

        Args:
            run_info (ForestRunInfo): The settings that should be passed to the
                MSMS algorithm.

        Returns:
            List[str]: The command that will be passed to docker.exec_create to run the
                MSMS algorithm.
        """

        # Make sure that the julia project is set before running any of the julia stuff
        julia_cmd = 'export JULIA_PROJECT="/home/forest"; /usr/bin/time -v julia /home/forest/cli/multi_cli.jl'

        # Process the output and log data
        julia_cmd += " --input-file-name /home/forest/shapefiles/" + str(self.json_name)

        # Process the Run data
        julia_cmd += " --rng-seed " + str(run_info.rng_seed)
        julia_cmd += " --region-name " + str(run_info.region_name)
        julia_cmd += " --subregion-name " + str(run_info.subregion_name)
        julia_cmd += " --pop-name " + str(run_info.pop_col)
        julia_cmd += " --num-dists " + str(run_info.num_dists)
        julia_cmd += " --pop-dev " + str(run_info.pop_dev)
        julia_cmd += " --gamma " + str(run_info.gamma)
        julia_cmd += " --steps " + str(run_info.n_steps)

        if run_info.output_file_name is not None:
            output_file_name = run_info.output_file_name
        else:
            output_file_name = f"{run_info.rng_seed}_atlas_gamma{run_info.gamma}_{run_info.n_steps}.jsonl"

        if run_info.ben:
            julia_cmd += f" | msms_parser -g /home/forest/shapefiles/{self.json_name} -r {run_info.region_name} -s {run_info.subregion_name} --ben"
            if not run_info.force_print:
                julia_cmd += " -o /home/forest/output/{self.json_name[:-5]}/{output_file_name}.ben -w"
        elif run_info.standard_jsonl:
            julia_cmd += f" | msms_parser -g /home/forest/shapefiles/{self.json_name} -r {run_info.region_name} -s {run_info.subregion_name}"
            if not run_info.force_print:
                julia_cmd += f" -o /home/forest/output/{self.json_name[:-5]}/{output_file_name} -w"
        elif not run_info.force_print:
            julia_cmd += f" --output-file-name /home/forest/output/{output_file_name}"

        return ["sh", "-c", julia_cmd]

    def log_file(self, run_info: ForestRunInfo):
        """
        Constructs the name of the log file that will be written to when the MSMS
        algorithm is run. This is used to store the output of stderr in the event
        that the MSMS algorithm fails.
        """
        log_file_dir = self.log_folder + f"/{self.json_name[:-5]}"
        log_file_name = f"Forest_{run_info.rng_seed}_atlas_gamma{run_info.gamma}_{run_info.n_steps}.log"

        os.makedirs(log_file_dir, exist_ok=True)

        return f"{log_file_dir}/{log_file_name}"
