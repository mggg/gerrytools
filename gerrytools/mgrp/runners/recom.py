from dataclasses import dataclass, field
import os
import json
from pathlib import Path
from .. import RunnerConfig
from typing import Callable, Dict


@dataclass
class RecomRunInfo:
    """
    Represents all of the settings that can be passed to the frcw
    Rust code.
    """

    pop_col: str
    """The name of the column in the dual graph JSON file that contains the population data."""
    assignment_col: str
    """The name of the column in the dual graph JSON file that contains the assignment data."""
    variant: str
    """The variant of the recom algorithm to be used. Options are A, B, C, D, R, AW, BW."""
    balance_ub: float = 0
    """The balance upper bound to be used in the frcw code. Only used in (R)eversible mode"""
    n_steps: int = 10
    """The number of steps that the frcw code should run for."""
    pop_tol: float = 0.05
    """The population tolerance to be used in the recom code."""
    n_threads: int = 1
    """The number of threads to be used to generate proposals for the frcw code."""
    batch_size: int = 1
    """The batch size to be used in the frcw code."""
    writer: str = "canonical"
    """The type of writer that should be used to write the output of the frcw code. Options 
    are:
        
    - tsv
    - jsonl
    - pcompress
    - jsonl-full
    - assignments
    - canonicalized-assignments
    - canonical
    - ben
    """
    sum_cols: list = field(default_factory=list)
    """The columns that should be summed in the output of the frcw code. This will only
        be shown if the writer is set to jsonl or jsonl-full."""
    region_weights: dict = field(default_factory=dict)
    """A dictionary of surcharges to be added to edges between regions. This is only used
        in the AW and BW variants of the frcw code."""
    # cut_edges_count: bool = False
    # spanning_tree_counts: bool = False
    rng_seed: int = 42
    """The random number generator seed to be used in the frcw code."""
    force_print: bool = False
    """If true, the output of the frcw code will be printed to the console instead of 
        being written to a file."""
    updaters: Dict[str, Callable] = field(default_factory=dict)
    """A dictionary of updaters that should be used when running the chain using the 
        mcmc_run_with_updaters method."""


# Should probably change the default log folder to /dev/null/
class RecomRunnerConfig(RunnerConfig):
    """
    Represents the configuration for a RunContainer which is used to run the
    frcw code on a given dual graph within the docker container.
    """

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
                used in frcw.
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

        self.variant_dict = {
            "A": "cut-edges-rmst",
            "B": "district-pairs-rmst",
            "C": "cut-edges-ust",
            "D": "district-pairs-ust",
            "R": "reversible",
            "AW": "cut-edges-region-aware",
            "BW": "district-pairs-region-aware",
        }

    def configure_vols_and_name(self):
        """
        Configures the volumes and name of the container that will be used to run the
        frcw code.
        """
        current_directory = os.getcwd()

        volumes = {
            current_directory: {"bind": "/home/recom/runner", "mode": "rw"},
            os.path.abspath(self.json_dir): {
                "bind": "/home/recom/shapefiles",
                "mode": "rw",
            },
            os.path.abspath(self.output_folder): {
                "bind": f"/home/recom/output/{self.json_name[:-5]}",
                "mode": "rw",
            },
        }

        return {
            "name": f"recom_runner",
            "volumes": volumes,
        }

    def run_command(self, run_info: RecomRunInfo):
        """
        Construcs the command that will be used to run the frcw code in the docker
        container

        Args:
            run_info (RecomRunInfo): The information that will be passed to the frcw code.

        Returns:
            List[str]: The command that will be passed to docker.exec_create to run frcw.
        """

        rust_cmd = ". /root/.cargo/env; /usr/bin/time -v frcw"

        # Process the Run data
        rust_cmd += " --graph-json /home/recom/shapefiles/" + self.json_name
        rust_cmd += " --rng-seed " + str(run_info.rng_seed)
        rust_cmd += " --pop-col " + str(run_info.pop_col)
        rust_cmd += " --assignment-col " + str(run_info.assignment_col)
        rust_cmd += " --n-steps " + str(run_info.n_steps)
        rust_cmd += " --tol " + str(run_info.pop_tol)
        rust_cmd += " --n-threads " + str(run_info.n_threads)
        rust_cmd += " --batch-size " + str(run_info.batch_size)
        rust_cmd += " --writer " + str(run_info.writer)
        if run_info.variant not in ["A", "B", "C", "D", "R", "AW", "VW"]:
            raise ValueError(
                "Please choose one of the acceptable recom methods: A, "
                "B, C, D, R (reversible), AW (cut edges region aware), "
                "or BW (district pairs region aware)"
            )
        else:
            rust_cmd += " --variant " + self.variant_dict[str(run_info.variant)]
            if str(run_info.variant) == "R":
                rust_cmd += " --balance-ub " + str(run_info.balance_ub)
        if run_info.sum_cols:
            rust_cmd += " --sum-cols " + " ".join(run_info.sum_cols)

        if run_info.variant in ["AW", "BW"]:
            if run_info.region_weights:
                rust_cmd += " --region-weights '"
                json_string = json.dumps(run_info.region_weights)
                json_string = json_string.replace(",", " ,")
                rust_cmd += json_string
                rust_cmd += "'"
            else:
                raise NameError(
                    f"Run info for option variant={run_info.variant} "
                    f"requires additional option 'region_wieghts'"
                    f"to be specified"
                )

        # TODO: Double check to frcw code to make sure that
        # these are being computed correctly
        # if run_info.cut_edges_count:
        #     rust_cmd += " --cut-edges-count"
        # if run_info.spanning_tree_counts:
        #     rust_cmd += " --st-counts"

        if run_info.force_print == True:
            pass
        elif run_info.writer == "pcompress":
            rust_cmd += f" > /home/recom/output/{self.json_name[:-5]}/Recom{run_info.variant}_{run_info.assignment_col}_{run_info.rng_seed}_{run_info.n_steps}_pcompress.chain"
        elif run_info.writer == "assignments":
            rust_cmd += f" > /home/recom/output/{self.json_name[:-5]}/Recom{run_info.variant}_{run_info.assignment_col}_{run_info.rng_seed}_{run_info.n_steps}.assignments"
        elif run_info.writer == "tsv":
            rust_cmd += f" > /home/recom/output/{self.json_name[:-5]}/Recom{run_info.variant}_{run_info.assignment_col}_{run_info.rng_seed}_{run_info.n_steps}.tsv"
        elif run_info.writer == "ben":
            rust_cmd += f" > /home/recom/output/{self.json_name[:-5]}/Recom{run_info.variant}_{run_info.assignment_col}_{run_info.rng_seed}_{run_info.n_steps}.jsonl.ben"
        else:
            rust_cmd += f" > /home/recom/output/{self.json_name[:-5]}/Recom{run_info.variant}_{run_info.assignment_col}_{run_info.rng_seed}_{run_info.n_steps}.jsonl"

        return ["sh", "-c", rust_cmd]

    def log_file(self, run_info: RecomRunInfo):
        """
        Constructs the name of the log file that will be written to when the
        frcw code is run. This is used to store teh output of stderr in the event
        that the code crashes.
        """
        log_file_dir = self.log_folder + f"/{self.json_name[:-5]}"
        log_file_name = f"Recom{run_info.variant}_{run_info.assignment_col}_{run_info.rng_seed}_{run_info.n_steps}.log"

        os.makedirs(log_file_dir, exist_ok=True)

        return f"{log_file_dir}/{log_file_name}"
