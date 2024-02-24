from dataclasses import dataclass, field
import os
import json
import docker


@dataclass
class RecomRunInfo:
    pop_col: str
    assignment_col: str
    variant: str
    balance_ub: float = 0
    n_steps: int = 10
    pop_tol: float = 0.05
    n_threads: int = 1
    batch_size: int = 1
    writer: str = "jsonl-full"
    sum_cols: list = field(default_factory=list)
    region_weights: dict = field(default_factory=dict)
    cut_edges_count: bool = False
    spanning_tree_counts: bool = False
    rng_seed: int = 42


# Should probably change the default log folder to /dev/null/
class RecomRunner:
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
        self.variant_dict = {
            "A": "cut-edges-rmst",
            "B": "district-pairs-rmst",
            "C": "cut-edges-ust",
            "D": "district-pairs-ust",
            "R": "reversible",
            "DW": "district-pairs-region-aware",
            "CW": "cut-edges-region-aware",
        }

    def configure(self, image_name="mgggdev/replicate:latest", ports={3000: 3000}):
        current_directory = os.getcwd()

        volumes = {
            current_directory: {"bind": "/recom/runner", "mode": "rw"},
            os.path.abspath(self.json_dir): {"bind": "/recom/shapefiles", "mode": "rw"},
            os.path.abspath(self.output_folder): {
                "bind": f"/recom/output/{self.json_name[:-5]}",
                "mode": "rw",
            },
        }

        port_bindings = {}

        for container_port, host_port in ports.items():
            port_bindings[container_port] = host_port

        return {
            "image": image_name,
            "detach": True,
            "name": f"recom_runner",
            "volumes": volumes,
            "ports": port_bindings,
            "auto_remove": True,
            "tty": True,
        }

    def run(self, container, run_info: RecomRunInfo):
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
        rust_cmd += " --writer " + str(run_info.writer)
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
                # json_string = json_string.replace("\"", "\\\"")
                json_string = json_string.replace(",", " ,")
                rust_cmd += json_string
                rust_cmd += "'"
            else:
                raise NameError(
                    f"Run info for option variant={run_info.variant} "
                    f"requires additional option 'region_wieghts'"
                    f"to be specified"
                )

        if run_info.cut_edges_count:
            rust_cmd += " --cut-edges-count"
        if run_info.spanning_tree_counts:
            rust_cmd += " --st-counts"
        rust_cmd += f" > /recom/output/{self.json_name[:-5]}/Recom{run_info.variant}_{run_info.assignment_col}_{run_info.rng_seed}_{run_info.n_steps}.jsonl"

        log_file_dir = self.log_folder + f"/{self.json_name[:-5]}"
        log_file_name = f"Recom{run_info.variant}_{run_info.assignment_col}_{run_info.rng_seed}_{run_info.n_steps}.log"

        os.makedirs(log_file_dir, exist_ok=True)

        # Run the command in the Docker container
        exec_id = self.client.api.exec_create(
            container.id, cmd=["sh", "-c", rust_cmd], tty=False
        )
        output = self.client.api.exec_start(exec_id)
        with open(log_file_dir + "/" + log_file_name, "w") as log_file:
            log_file.write(output.decode("utf-8"))
