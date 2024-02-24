from dataclasses import dataclass
import os
import json
import docker


@dataclass
class ForestRunInfo:
    precinct_name: str
    county_name: str
    pop_col: str
    num_dists: int = 2
    pop_dev: float = 0.1
    gamma: float = 0.0
    n_steps: int = 10
    rng_seed: int = 42


class ForestRunner:
    def __init__(
        self,
        json_dir: str,
        json_name: str,
        output_folder: str = "./output",
        log_folder: str = "./logs",
    ):
        self.json_dir = json_dir
        self.json_name = json_name
        json_name_without_extension = os.path.splitext(json_name)[0]
        self.output_folder = os.path.join(output_folder, json_name_without_extension)
        self.log_folder = os.path.abspath(log_folder)
        self.client = docker.from_env()

    def configure(self, image_name="mgggdev/replicate:latest", ports={3000: 3000}):
        current_directory = os.getcwd()

        volumes = {
            current_directory: {"bind": "/forest/runner", "mode": "rw"},
            os.path.abspath(self.json_dir): {
                "bind": "/forest/shapefiles",
                "mode": "rw",
            },
            os.path.abspath(self.output_folder): {
                "bind": f"/forest/output/{self.json_name[:-5]}",
                "mode": "rw",
            },
        }

        port_bindings = {}

        for container_port, host_port in ports.items():
            port_bindings[container_port] = host_port

        return {
            "image": image_name,
            "detach": True,
            "name": f"forest_runner",
            "volumes": volumes,
            "ports": port_bindings,
            "auto_remove": True,
            "tty": True,
        }

    def run(self, container, run_info: ForestRunInfo):
        # Make sure that the julia project is set before running any of the julia stuff
        julia_cmd = 'export JULIA_PROJECT="/forest"; /usr/bin/time -v julia /forest/cli/multi_cli.jl'

        # Process the output and log data
        julia_cmd += " --input_file_path /forest/shapefiles/" + str(self.json_name)
        julia_cmd += f" --output_folder_path /forest/output/{self.json_name[:-5]}"

        # Process the Run data
        julia_cmd += " --rng_seed " + str(run_info.rng_seed)
        julia_cmd += " --precinct_name " + str(run_info.precinct_name)
        julia_cmd += " --county_name " + str(run_info.county_name)
        julia_cmd += " --pop_name " + str(run_info.pop_col)
        julia_cmd += " --num_dists " + str(run_info.num_dists)
        julia_cmd += " --pop_dev " + str(run_info.pop_dev)
        julia_cmd += " --gamma " + str(run_info.gamma)
        julia_cmd += " --steps " + str(run_info.n_steps)

        # Execute the Docker command
        log_file_dir = self.log_folder + f"/{self.json_name[:-5]}"
        log_file_name = f"Forest_{run_info.rng_seed}_atlas_gamma{run_info.gamma}_{run_info.n_steps}.log"

        os.makedirs(log_file_dir, exist_ok=True)

        exec_id = self.client.api.exec_create(
            container.id,
            cmd=["sh", "-c", julia_cmd],
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
