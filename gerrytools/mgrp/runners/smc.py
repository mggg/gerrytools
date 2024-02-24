from dataclasses import dataclass, field
import os
import docker
import subprocess


@dataclass
class SMCMapInfo:
    pop_col: str
    n_dists: int
    pop_tol: float = 0.01
    pop_bounds: list = field(default_factory=list)


@dataclass
class SMCRedistInfo:
    n_sims: int
    rng_seed: int = 42
    compactness: float = 1.0
    resample: bool = False
    adapt_k_thresh: float = 0.985
    seq_alpha: float = 0.5
    pop_temper: float = 0.0
    final_infl: float = 1.0
    est_label_mult: float = 1.0
    verbose: bool = False
    silent: bool = False
    tally_columns: list = field(default_factory=list)


class SMCRunner:
    def __init__(
        self,
        shapefile_dir: str,
        shapefile_name: str,
        output_folder: str = "./output",
        log_folder: str = "./logs",
    ):
        self.shapefile_dir = shapefile_dir
        self.shapefile_name = shapefile_name
        self.output_folder = os.path.abspath(output_folder)
        self.log_folder = os.path.abspath(log_folder)
        self.client = docker.from_env()

    def configure(self, image_name="mgggdev/replicate:latest", ports={3000: 3000}):
        current_directory = os.getcwd()

        volumes = {
            current_directory: {"bind": "/smc/runner", "mode": "rw"},
            os.path.abspath(self.shapefile_dir): {
                "bind": "/smc/shapefiles",
                "mode": "rw",
            },
            os.path.abspath(self.output_folder): {"bind": f"/smc/output", "mode": "rw"},
        }

        port_bindings = {}

        for container_port, host_port in ports.items():
            port_bindings[container_port] = host_port
        return {
            "image": image_name,
            "detach": True,
            "name": f"smc_runner",
            "volumes": volumes,
            "ports": port_bindings,
            "auto_remove": True,
            "tty": True,
        }

    def run(self, container, map_info: SMCMapInfo, redist_info: SMCRedistInfo):
        r_cmd = "/usr/bin/time -v Rscript /smc/cli/smc_cli.R"

        # Process the base IO information
        r_cmd += f" --shapefile /smc/shapefiles/{self.shapefile_name}"
        if len(redist_info.tally_columns) > 0:
            r_cmd += (
                f" --tally_cols /smc/shapefiles/{' '.join(redist_info.tally_columns)}"
            )
        r_cmd += f" --output_file /smc/output/{self.shapefile_name}/SMC_{redist_info.rng_seed}_{redist_info.n_sims}.csv"

        # Process the SMC Map data
        r_cmd += f" --pop_col {map_info.pop_col}"
        r_cmd += f" --pop_tol {map_info.pop_tol}"
        r_cmd += f" --n_dists {map_info.n_dists}"
        if len(map_info.pop_bounds) == 3:
            r_cmd += f" --pop_bounds {' '.join(map_info.pop_bounds)}"

        # Process the SMC Redist data
        r_cmd += f" --n_sims {redist_info.n_sims}"
        r_cmd += f" --compactness {redist_info.compactness}"
        if redist_info.resample:
            r_cmd += " --resample"
        r_cmd += f" --adapt_k_thresh {redist_info.adapt_k_thresh}"
        r_cmd += f" --seq_alpha {redist_info.seq_alpha}"
        r_cmd += f" --pop_temper {redist_info.pop_temper}"
        r_cmd += f" --final_infl {redist_info.final_infl}"
        r_cmd += f" --est_label_mult {redist_info.est_label_mult}"
        if redist_info.verbose:
            r_cmd += " --verbose"
        if redist_info.silent:
            r_cmd += " --silent"

        log_file_dir = f"{self.log_folder}/{self.shapefile_name}"
        log_file_name = f"SMC_{redist_info.rng_seed}_{redist_info.n_sims}.log"

        os.makedirs(f"./test_log/{self.shapefile_name}", exist_ok=True)

        # The R and C++ outputs are weird, so you have to handle them a little differently
        exec_id = self.client.api.exec_create(
            container.id,
            cmd=["sh", "-c", r_cmd],
            tty=False,
            stdout=True,
            stderr=True,
            stdin=False,
        )

        output_generator = self.client.api.exec_start(
            exec_id, stream=True, detach=False
        )

        buffer = ""
        for output in output_generator:
            output_decoded = output.decode("utf-8")
            buffer += output_decoded

            # Only print the buffer when a newline is encountered
            while "\n" in buffer or "\r" in buffer:
                line, _, buffer = buffer.partition("\n")

                if not line:
                    line, _, buffer = buffer.partition("\r")
                print(line)

        if buffer:
            print(buffer)
