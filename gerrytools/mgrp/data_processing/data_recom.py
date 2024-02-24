from dataclasses import dataclass, field
import os
import json
import docker
from docker.errors import APIError
import re
import ast
import gerrychain
from typing import Callable, Dict
from geopandas import geodataframe
from gerrychain import (
    GeographicPartition,
    Partition,
    Graph,
    MarkovChain,
    proposals,
    updaters,
    constraints,
    accept,
    Election,
)
import json


@dataclass
class RecomDataRunInfo:
    pop_col: str
    assignment_col: str
    variant: str
    balance_ub: float = 0
    n_steps: int = 10
    pop_tol: float = 0.05
    n_threads: int = 1
    batch_size: int = 1
    sum_cols: list = field(default_factory=list)
    region_weights: dict = field(default_factory=dict)
    # cut_edges_count: bool = False
    # spanning_tree_counts: bool = False
    rng_seed: int = 42
    updaters: Dict[str, Callable] = field(default_factory=dict)


class RecomDataRunner:
    def __init__(
        self,
        json_dir: str,
        json_name: str,
        initial_partition: Partition,
        log_folder: str = "./logs",
    ):
        self.json_dir = json_dir
        self.json_name = json_name
        json_name_without_extension = os.path.splitext(json_name)[0]
        self.partition = initial_partition
        self.log_folder = os.path.abspath(log_folder)
        self.client = docker.from_env()
        self.variant_dict = {
            "A": "cut-edges-rmst",
            "B": "district-pairs-rmst",
            "C": "cut-edges-ust",
            "D": "district-pairs-ust",
            "R": "reversible",
            "DW": "district-pairs-region-aware",
            "CW": "cut-edges-region-aware",
        }
        # Change the names to AW and BW for the region aware
        self.prev_step = None
        self.sums = None
        self.pops = None

    def configure(self, image_name="mgggdev/replicate:latest", ports={3000: 3000}):
        current_directory = os.getcwd()

        volumes = {
            current_directory: {"bind": "/recom/runner", "mode": "rw"},
            os.path.abspath(self.json_dir): {"bind": "/recom/shapefiles", "mode": "rw"},
        }

        port_bindings = {}

        for container_port, host_port in ports.items():
            port_bindings[container_port] = host_port

        return {
            "image": image_name,
            "detach": True,
            "name": f"recom_data_runner",
            "volumes": volumes,
            "ports": port_bindings,
            "auto_remove": True,
            "tty": True,
        }

    def run(self, container, run_info: RecomDataRunInfo):
        rust_cmd = ". /root/.cargo/env; /usr/bin/time -v /frcw/target/release/frcw"

        # Process the Run data
        rust_cmd += " --graph-json /recom/shapefiles/" + self.json_name
        rust_cmd += " --rng-seed " + str(run_info.rng_seed)
        rust_cmd += " --pop-col " + str(run_info.pop_col)
        rust_cmd += " --assignment-col " + str(run_info.assignment_col)
        rust_cmd += " --n-steps " + str(run_info.n_steps)
        rust_cmd += " --tol " + str(run_info.pop_tol)
        rust_cmd += " --n-threads " + str(run_info.n_threads)
        rust_cmd += " --batch-size " + str(run_info.batch_size)
        rust_cmd += " --writer jsonl-full"
        if run_info.variant not in ["A", "B", "C", "D", "R", "DW", "CW"]:
            raise ValueError(
                "Please choose one of the acceptable recom methods: A, "
                "B, C, D, R (reversible), DW (district pairs region aware), "
                "or CW (cut edges region aware)"
            )
        else:
            rust_cmd += " --variant " + self.variant_dict[str(run_info.variant)]
            if str(run_info.variant) == "R":
                rust_cmd += " --balance-ub " + str(run_info.balance_ub)
        if run_info.sum_cols:
            rust_cmd += " --sum-cols " + " ".join(run_info.sum_cols)

        if run_info.variant in ["DW", "CW"]:
            if run_info.region_weights:
                rust_cmd += " --region-weights '"
                json_string = json.dumps(run_info.region_weights)
                json_string = json_string.replace(",", " ,")
                rust_cmd += json_string
                rust_cmd += "'"
            else:
                raise RuntimeError(
                    f"Run info for option variant={run_info.variant} "
                    f"requires additional option 'region_wieghts' "
                    f"to be specified"
                )

        # if run_info.cut_edges_count:
        #     rust_cmd += " --cut-edges-count"
        # rust_cmd += " --cut-edges-count"

        log_file_dir = self.log_folder + f"/{self.json_name[:-5]}"
        log_file_name = f"Recom{run_info.variant}_{run_info.assignment_col}_{run_info.rng_seed}_{run_info.n_steps}.log"
        os.makedirs(log_file_dir, exist_ok=True)

        exec_id = self.client.api.exec_create(
            container.id, cmd=["sh", "-c", rust_cmd], tty=False
        )

        self.prev_step = None
        updater_values = {}

        try:
            # Demux separates the streams for stdout and stderr so that
            # they can be processed separately
            output_stream = self.client.api.exec_start(exec_id, stream=True, demux=True)
            stdout_buffer = ""

            count = 0
            with open(log_file_dir + "/" + log_file_name, "w") as log_file:
                for stdout, stderr in output_stream:
                    if count > 3:
                        raise ValueError("MaxCount")
                    if stderr:
                        log_file.write(stderr.decode("utf-8"))

                    if stdout:
                        # Accumulate stdout in a buffer
                        stdout_buffer += stdout.decode("utf-8")

                        # When a complete line is reached, process the whole line
                        while "\n" in stdout_buffer:
                            line, stdout_buffer = stdout_buffer.split("\n", 1)
                            # print(line, end='\n')
                            yield from self._process_output(
                                line, run_info.updaters, updater_values
                            )

        except APIError as e:
            print("Docker API Error: ", e.explanation)

    def _process_output(self, b_string, updater_dict, updater_values):
        try:
            # Parse the line as JSON
            data = json.loads(b_string.strip())

        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            return

        if "init" in data:
            self.pops = data["init"].get("populations", [])
            self.sums = data["init"].get("sums", {})

        if "step" in data:
            nodes = data["step"].get("nodes", [])
            dists = data["step"].get("dists", [])
            sums = data["step"].get("sums", {})
            step = data["step"].get("step", None)
            pops = data["step"].get("populations", [])

            for key in self.sums.keys():
                self.sums[key][dists[0] - 1] = sums[key][0]
                self.sums[key][dists[1] - 1] = sums[key][1]

            self.pops[dists[0] - 1] = pops[0]
            self.pops[dists[1] - 1] = pops[1]

            # Fill in gaps in step numbers
            if self.prev_step is not None and step > self.prev_step + 1:
                for missing_step in range(self.prev_step + 1, step):
                    yield {
                        "Step": missing_step,
                        "Populations": self.pops[0],
                        "Sums": self.sums,
                        "Updaters": updater_values,
                    }

            self.prev_step = step  # Update the last step

            # Leave as list comprehension for speed
            new_assignments = {
                key: value for keys, value in zip(nodes, dists) for key in keys
            }
            assignments = self.partition.assignment.mapping
            assignments.update(new_assignments)

            # Update the assignment. We are throwing these away later anyway, so
            # it's best to use preallocated memory
            self.partition = Partition(self.partition.graph, assignments)

            # Update the existing updater_values dictionary
            for func_name, func in updater_dict.items():
                updater_values[func_name] = func(self.partition)

            yield {
                "Step": step,
                "Populations": self.pops[0],
                "Sums": self.sums,
                "Updaters": updater_values,
            }
