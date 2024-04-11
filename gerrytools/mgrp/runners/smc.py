from dataclasses import dataclass, field
import os
import subprocess
from typing import Optional, List
from .. import RunnerConfig


@dataclass
class SMCMapInfo:
    """
    Controls the information passed to the
    `redist_map() <https://alarm-redist.org/redist/reference/redist_map.html>`_
    method in the `redist` R package.
    """

    pop_col: str
    """The name of the column in the shapefile that contains the population data.
        This will be used to derive the `total_pop` parameter in the `redist_map()`
        method."""
    n_dists: int
    """The number of districts that the shapefile should be partitioned into."""
    pop_tol: float = 0.01
    """The population tolerance to be used in the `redist_map()` method."""
    pop_bounds: list = field(default_factory=list)
    """The population bounds to be used in the `redist_map()` method. This
        needs to be a list of three ints: [lower_bound, target, upper_bound]."""


@dataclass
class SMCRedistInfo:
    """
    Controls the information passed to the
    `redist_smc() <https://alarm-redist.org/redist/reference/redist_smc.html>`_
    method in the `redist` R package. The map is derived usind the map info and
    runner configuration.
    """

    n_sims: int
    """The number of samples to draw"""
    rng_seed: int = 42
    """The random number generator seed to be used the SMC algorithm."""
    compactness: float = 1.0
    """The compactness parameter to be used in the SMC algorithm."""
    resample: bool = False
    """Whether to perform a final resampling step so that the generated plans can 
        be used immediately."""
    adapt_k_thresh: float = 0.985
    """The threshold value used in the heuristic to select a value of :math:`k_i` for 
        each splitting iteration. Must be in the range [0, 1]."""
    seq_alpha: float = 0.5
    """The amount to adjust the weights by at each resampling step. Must be in the range [0, 1]."""
    pop_temper: float = 0.0
    """The strength of the automatic population tempering. If the algorithm gets stuck, then it
        is recommended that you start with values between 0.01-0.05."""
    final_infl: float = 1.0
    """A multiplier for the population constraint on the final iteration. Used to loosen the
        constraint when the sampler is getting suck on the final split."""
    est_label_mult: float = 1.0
    """A multiplier for the number of importance samples to use in estimating the number of 
        ways to sequentially label the districts"""
    verbose: bool = False
    """Whether or not to log the intermediate information during the running of SMC"""
    silent: bool = False
    """Whether or not to suppress all diagnostic output"""
    tally_columns: list = field(default_factory=list)
    """A list of columns to be tallied into the output file. This is only generated if the
        standard_jsonl and ben flags are set to False."""
    output_file_name: Optional[str] = None
    """The desired name of the output file. If not set, then the file name will be determied
        according to a set of heuristics."""
    standard_jsonl: bool = True
    """Whether or not to output the results in the standard JSONL format."""
    ben: bool = False
    """Whether or not to output the results in the BEN format. Overrides the standard_jsonl flag."""


class SMCRunnerConfig(RunnerConfig):
    """
    Represents the configuration for a SMCReplicator which is used to run the
    Sequential Monte Carlo (SMC) algorithm on a shapefile within the
    docker container.
    """

    def __init__(
        self,
        shapefile_dir: str,
        shapefile_name: str,
        output_folder: str = "./output",
        log_folder: str = "./logs",
    ):
        """
        Initializes the SMCRunnerConfig object.

        Args:
            shapefile_dir (str): The directory that contains the shapefile.
            shapefile_name (str): The name of the shapefile that should be used in the SMC algorithm.
            output_folder (str): The directory where the output files should be written to. Defaults to "./output".
            log_folder (str): The directory where the log files should be written to. Defaults to "./logs".
        """
        self.shapefile_dir = shapefile_dir
        self.shapefile_name = shapefile_name
        self.output_folder = os.path.abspath(f"{output_folder}/{shapefile_name}")
        self.log_folder = os.path.abspath(log_folder)

    def configure_vols_and_name(self):
        """
        Configures the volumes and name for the docker container that will be used to run the SMC algorithm.
        """
        current_directory = os.getcwd()

        volumes = {
            current_directory: {"bind": "/home/smc/runner", "mode": "rw"},
            os.path.abspath(self.shapefile_dir): {
                "bind": "/home/smc/shapefiles",
                "mode": "rw",
            },
            os.path.abspath(self.output_folder): {
                "bind": f"/home/smc/output/{self.shapefile_name}",
                "mode": "rw",
            },
        }

        return {
            "name": f"smc_runner",
            "volumes": volumes,
        }

    def run_command(
        self, map_info: SMCMapInfo, redist_info: SMCRedistInfo
    ) -> List[str]:
        """
        Construcs the command that will be used to run the SMC algorithm in the docker container

        Args:
            map_info (SMCMapInfo): The information that will be passed to the redist_map() method.
            redist_info (SMCRedistInfo): The information that will be passed to the redist_smc() method.

        Returns:
            List[str]: The command that will be used to run the SMC algorithm in the docker container.
        """
        r_cmd = f"/usr/bin/time -v Rscript /home/smc/cli/smc_cli.R"

        # Process the base IO information
        r_cmd += f" --shapefile /home/smc/shapefiles/{self.shapefile_name}"
        if len(redist_info.tally_columns) > 0:
            r_cmd += f" --tally-cols {' '.join(redist_info.tally_columns)}"

        # Process the SMC Map data
        r_cmd += f" --pop-col {map_info.pop_col}"
        r_cmd += f" --pop-tol {map_info.pop_tol}"
        r_cmd += f" --n-dists {map_info.n_dists}"
        if len(map_info.pop_bounds) == 3:
            r_cmd += f" --pop-bounds {' '.join(map_info.pop_bounds)}"

        # Process the SMC Redist data
        r_cmd += f" --n-sims {redist_info.n_sims}"
        r_cmd += f" --compactness {redist_info.compactness}"
        if redist_info.resample:
            r_cmd += " --resample"
        r_cmd += f" --adapt-k-thresh {redist_info.adapt_k_thresh}"
        r_cmd += f" --seq-alpha {redist_info.seq_alpha}"
        r_cmd += f" --pop-temper {redist_info.pop_temper}"
        r_cmd += f" --final-infl {redist_info.final_infl}"
        r_cmd += f" --est-label-mult {redist_info.est_label_mult}"
        if redist_info.verbose:
            r_cmd += " --verbose"
        if redist_info.silent:
            r_cmd += " --silent"

        if redist_info.output_file_name is not None:
            output_file_name = redist_info.output_file_name
        else:
            output_file_name = f"SMC_{redist_info.rng_seed}_{redist_info.n_sims}"

        if redist_info.ben:
            r_cmd += f" --print | smc_parser --ben -o /home/smc/output/{self.shapefile_name}/{output_file_name}.jsonl.ben -w"
        elif redist_info.standard_jsonl:
            r_cmd += f" --print | smc_parser --jsonl -o /home/smc/output/{self.shapefile_name}/{output_file_name}.jsonl -w"
        else:
            r_cmd += f" --output-file /home/smc/output/{self.shapefile_name}/{output_file_name}.csv"

        return ["sh", "-c", r_cmd]

    def log_file(self, map_info: SMCMapInfo, redist_info: SMCRedistInfo):
        """
        Constructs the name of teh log file that will be written to when the SMC
        algorithm is run. Thsi is used to store teh output of stderr in the event
        that the SMC algorithm fails.
        """
        log_file_dir = f"{self.log_folder}/{self.shapefile_name}"
        log_file_name = f"SMC_{redist_info.rng_seed}_{redist_info.n_sims}.log"

        os.makedirs(log_file_dir, exist_ok=True)

        return f"{log_file_dir}/{log_file_name}"
