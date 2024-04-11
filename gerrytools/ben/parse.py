import docker
from pathlib import Path
from typing import Optional
import os
from .docker_manager import managed_docker_container
import logging

logger = logging.getLogger("ben")


def msms_parse(
    mode: str,
    region: str,
    subregion: str,
    dual_graph_path: str,
    input_file_path: str,
    output_file_path: str,
    verbose: bool = True,
    docker_image_name: str = "mgggdev/replicate:v0.2",
    docker_client_args: Optional[dict] = None,
):
    """
    Runs the ``msms_parser`` CLI tool from the
    `msms_parser <httpss://github.com/peterrrock2/msms_parser>`_
    git repo in a Docker container.

    Args:
        mode (str): The mode to run the program in. Must be one of 'ben', 'standard_jsonl'.
        region (str): The main region used in the MSMS algorithm (First position in
            the "levels in graph" tuple in the MSMS JSONL output).
        subregion (str): The subregion used in the MSMS algorithm. (Second position in
            the "levels in graph" tuple in the MSMS JSONL output).
        dual_graph_path (str): The path to the dual graph file JSON file used in
            the MSMS algorithm.
        input_file_path (str): The path to the input file to read from containing the
            MSMS output. This file will be an "Atlas" for the MSMS algorithm, and can
            be identified by the string "This is an Atlas for Redistricting Maps" at
            the top of the file.
        output_file_path (str): The path to the output file to write to. By convention,
            if this file already exists, then it will be overwritten. If using the
            "ben" mode, it is recommended that you use the ".jsonl.ben" extension, and
            when using the "standard_jsonl" mode, it is recommended that you use the
            ".jsonl" extension.
        verbose (bool, optional): Whether to run the program in verbose mode. Defaults to True.
        docker_image_name (str, optional): The name of the Docker image to run the program in.
            Defaults to "mgggdev/replicate:v0.2".
        docker_client_args (dict, optional): Additional arguments to pass to the Docker client.
            Used primarily if there are multiple docker contexts on the same machine.
            Defaults to None.
    """
    if docker_client_args is not None:
        client = docker.DockerClient(**docker_client_args)
    else:
        client = docker.from_env()

    try:
        print(f"Pulling Docker image {docker_image_name}")
        client.images.pull(docker_image_name)
    except Exception as e:
        print(
            f"Error comparing docker container {docker_image_name} against web version. "
            f"Attempting to run using local image"
        )

    # Set up the input and output paths
    input_parent_path = Path(input_file_path).parent
    input_file_name = Path(input_file_path).name

    dual_graph_parent_path = Path(dual_graph_path).parent
    dual_graph_name = Path(dual_graph_path).name

    if output_file_path is not None:
        output_parent_path = Path(output_file_path).parent
        output_file_name = Path(output_file_path).name
    else:
        output_parent_path = input_parent_path
        output_file_name = None

    os.makedirs(output_parent_path, exist_ok=True)

    config_args = {
        "name": "parse_msms_runner",
        "image": docker_image_name,
        "detach": True,
        "auto_remove": True,
        "tty": True,
        "stdin_open": True,
    }

    if (
        input_parent_path == output_parent_path
        and input_parent_path == dual_graph_parent_path
    ):
        config_args["volumes"] = {
            os.getcwd(): {"bind": "/home/ben/working", "mode": "rw"},
            input_parent_path.resolve(): {
                "bind": "/home/ben/io",
                "mode": "rw",
            },
        }
        docker_input_path = "/home/ben/io"
        docker_output_path = "/home/ben/io"
        docker_dual_graph_path = "/home/ben/io"

    # Now the dual graph path is different from the input path
    elif input_parent_path == output_parent_path:
        config_args["volumes"] = {
            os.getcwd(): {"bind": "/home/ben/working", "mode": "rw"},
            input_parent_path.resolve(): {
                "bind": "/home/ben/io",
                "mode": "rw",
            },
            dual_graph_parent_path.resolve(): {
                "bind": "/home/ben/dual_graph",
                "mode": "rw",
            },
        }
        docker_input_path = "/home/ben/io"
        docker_output_path = "/home/ben/io"
        docker_dual_graph_path = "/home/ben/dual_graph"

    # Now the dual graph path might be the same as one of the paths, but
    # the input and output paths are different
    elif dual_graph_parent_path == input_parent_path:
        config_args["volumes"] = {
            os.getcwd(): {"bind": "/home/ben/working", "mode": "rw"},
            input_parent_path.resolve(): {
                "bind": "/home/ben/input",
                "mode": "rw",
            },
            output_parent_path.resolve(): {
                "bind": "/home/ben/output",
                "mode": "rw",
            },
        }
        docker_input_path = "/home/ben/input"
        docker_output_path = "/home/ben/output"
        docker_dual_graph_path = "/home/ben/input"

    elif dual_graph_parent_path == output_parent_path:
        config_args["volumes"] = {
            os.getcwd(): {"bind": "/home/ben/working", "mode": "rw"},
            input_parent_path.resolve(): {
                "bind": "/home/ben/input",
                "mode": "rw",
            },
            output_parent_path.resolve(): {
                "bind": "/home/ben/output",
                "mode": "rw",
            },
        }
        docker_input_path = "/home/ben/input"
        docker_output_path = "/home/ben/output"
        docker_dual_graph_path = "/home/ben/output"

    # Now all the paths are different
    else:
        config_args["volumes"] = {
            os.getcwd(): {"bind": "/home/ben/working", "mode": "rw"},
            input_parent_path.resolve(): {
                "bind": "/home/ben/input",
                "mode": "rw",
            },
            output_parent_path.resolve(): {
                "bind": "/home/ben/output",
                "mode": "rw",
            },
            dual_graph_parent_path.resolve(): {
                "bind": "/home/ben/dual_graph",
                "mode": "rw",
            },
        }

        docker_input_path = "/home/ben/input"
        docker_output_path = "/home/ben/output"
        docker_dual_graph_path = "/home/ben/dual_graph"

    # Build the command to run in the container
    if mode not in [
        "ben",
        "standard_jsonl",
    ]:
        print(f"Invalid mode: {mode}. " "Mode must be one of 'ben', 'standard_jsonl'")
        return

    with managed_docker_container(client, config_args) as container:

        msms_cmd = (
            f"msms_parser -w -g {docker_dual_graph_path}/{dual_graph_name}"
            f" -i {docker_input_path}/{input_file_name}"
            f" -r {region} -s {subregion}"
        )
        if mode == "ben":
            msms_cmd += " -b"

        if verbose:
            msms_cmd += " -v"

        if output_file_name is not None:
            msms_cmd += f" -o {docker_output_path}/{output_file_name}"

        cmd = ["sh", "-c", f"{msms_cmd}"]

        logger.debug(f"Running command: {cmd}")
        exec_id = client.api.exec_create(
            container.id, cmd=cmd, tty=False, stdout=True, stderr=True, stdin=False
        )

        output_generator = client.api.exec_start(
            exec_id=exec_id, stream=True, detach=False
        )

        for output in output_generator:
            print(output.decode("utf-8"), end="")

    client.close()


def smc_parse(
    mode: str,
    input_file_path: str,
    output_file_path: str,
    verbose: bool = True,
    docker_image_name: str = "mgggdev/replicate:v0.2",
    docker_client_args: Optional[dict] = None,
):
    """
    Runs the ``smc_parser`` CLI tool from the
    `smc_parser <httpss://github.com/peterrrock2/smc_parser>`_
    git repo in a Docker container.

    Args:
        mode (str): The mode to run the program in. Must be one of 'ben', 'standard_jsonl'.
        input_file_path (str): The path to the input file to read from containing the
            SMC assignments output.
        output_file_path (str): The path to the output file to write to. By convention,
            if this file already exists, then it will be overwritten. If using the
            "ben" mode, it is recommended that you use the ".jsonl.ben" extension, and
            when using the "jsonl" mode, it is recommended that you use the ".jsonl" extension.
        verbose (bool, optional): Whether to run the program in verbose mode. Defaults to True.
        docker_image_name (str, optional): The name of the Docker image to run the program in.
            Defaults to "mgggdev/replicate:v0.2".
        docker_client_args (dict, optional): Additional arguments to pass to the Docker client.
            Used primarily if there are multiple docker contexts on the same machine.
            Defaults to None.
    """
    if docker_client_args is not None:
        client = docker.DockerClient(**docker_client_args)
    else:
        client = docker.from_env()

    try:
        print(f"Pulling Docker image {docker_image_name}")
        client.images.pull(docker_image_name)
    except Exception as e:
        print(
            f"Error comparing docker container {docker_image_name} against web version. "
            f"Attempting to run using local image"
        )

    # Set up the input and output paths
    input_parent_path = Path(input_file_path).parent
    input_file_name = Path(input_file_path).name

    if output_file_path is not None:
        output_parent_path = Path(output_file_path).parent
        output_file_name = Path(output_file_path).name
    else:
        output_parent_path = input_parent_path
        output_file_name = None

    os.makedirs(output_parent_path, exist_ok=True)

    config_args = {
        "name": "parse_msms_runner",
        "image": docker_image_name,
        "detach": True,
        "auto_remove": True,
        "tty": True,
        "stdin_open": True,
    }

    if input_parent_path == output_parent_path:
        config_args["volumes"] = {
            os.getcwd(): {"bind": "/home/ben/working", "mode": "rw"},
            input_parent_path.resolve(): {
                "bind": "/home/ben/io",
                "mode": "rw",
            },
        }
        docker_input_path = "/home/ben/io"
        docker_output_path = "/home/ben/io"
    else:
        config_args["volumes"] = {
            os.getcwd(): {"bind": "/home/ben/working", "mode": "rw"},
            input_parent_path.resolve(): {
                "bind": "/home/ben/input",
                "mode": "rw",
            },
            output_parent_path.resolve(): {
                "bind": "/home/ben/output",
                "mode": "rw",
            },
        }

        docker_input_path = "/home/ben/input"
        docker_output_path = "/home/ben/output"

    with managed_docker_container(client, config_args) as container:
        # Build the command to run in the container
        if mode not in [
            "ben",
            "standard_jsonl",
        ]:
            print(f"Invalid mode: {mode}. Mode must be one of 'ben', 'standard_jsonl'")
            return

        smc_cmd = f"smc_parser -w -i {docker_input_path}/{input_file_name}"

        if mode == "ben":
            smc_cmd += " -b"
        else:
            smc_cmd += " -j"

        if verbose:
            smc_cmd += " -v"

        if output_file_name is not None:
            smc_cmd += f" -o {docker_output_path}/{output_file_name}"

        cmd = ["sh", "-c", f"{smc_cmd}"]

        logger.debug(f"Running command: {cmd}")
        exec_id = client.api.exec_create(
            container.id, cmd=cmd, tty=False, stdout=True, stderr=True, stdin=False
        )

        output_generator = client.api.exec_start(
            exec_id=exec_id, stream=True, detach=False
        )

        for output in output_generator:
            print(output.decode("utf-8"), end="")

    client.close()
