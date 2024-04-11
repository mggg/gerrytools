import docker
from pathlib import Path
from typing import Optional
import os
from .docker_manager import managed_docker_container
import logging

logger = logging.getLogger("ben")


def canonicalize_ben_file(
    input_file_path: str,
    output_file_path: Optional[str] = None,
    verbose: bool = True,
    docker_image_name: str = "mgggdev/replicate:v0.2",
    docker_client_args: Optional[dict] = None,
):
    """
    Runs the REBEN CLI tool from the
    `binary-ensemble <https://crates.io/crates/binary-ensemble>`_
    crate in a Docker container. This function is specifically designed to just
    run the canonialization method of the REBEN tool. This canonicalizes the
    assignment vectors in the input file so that they are ordered starting at 1, so
    an assignment vector of the form

    [2,2,4,4,1,1,3,3]

    will be converted to

    [1,1,2,2,3,3,4,4]

    This can significantly improve the compression ratios when the new file is then run
    through the BEN tool with the 'x-encode' mode.

    Args:
        input_file_path (str): The path to the input file to read from. This is expected
            to be a BEN file.
        output_file_path (str, optional): The path of the output file to write to.
            When not given, the output name will be derived from the input name
            according to a set of heuristics (usually adding or deleting "ben"
            in the file extension). Defaults to None.
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
        "name": "reben_runner",
        "image": docker_image_name,
        "detach": True,
        "auto_remove": True,
        "tty": True,
        "stdin_open": True,
    }

    if input_parent_path == output_parent_path:
        config_args["volumes"] = {
            input_parent_path.resolve(): {
                "bind": "/home/ben/io",
                "mode": "rw",
            },
        }
        docker_input_path = "/home/ben/io"
        docker_output_path = "/home/ben/io"
    else:
        config_args["volumes"] = {
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
        reben_cmd = f"reben {docker_input_path}/{input_file_name} -m ben"

        if verbose:
            reben_cmd += " -v"

        if output_file_name is not None:
            reben_cmd += f" -o {docker_output_path}/{output_file_name}"

        cmd = ["sh", "-c", f"{reben_cmd}"]

        logging.debug(f"Running command: {cmd}")
        exec_id = client.api.exec_create(
            container.id, cmd=cmd, tty=False, stdout=True, stderr=True, stdin=False
        )

        output_generator = client.api.exec_start(
            exec_id=exec_id, stream=True, detach=False
        )

        for output in output_generator:
            print(output.decode("utf-8"), end="")

    client.close()


def relabel_json_file_by_key(
    key: str,
    dual_graph_path: str,
    output_file_path: Optional[str] = None,
    verbose: bool = True,
    docker_image_name: str = "mgggdev/replicate:v0.2",
    docker_client_args: Optional[dict] = None,
):
    """
    Runs the REBEN CLI tool from the
    `binary-ensemble <https://crates.io/crates/binary-ensemble>`_
    crate in a Docker container. This function is specifically designed to run the
    REBEN tool in its relabeling mode.

    This function will relabel the input dual-graph according to the key value
    provided and output a new dual-graph file together with a map file in the event that
    the user would like to recover the original file from the relabeled file. This is
    best done before any ensembles are run on the dual-graph file.

    Args:
        key (str): The key to relabel the dual-graph with. This should appear as an
            attribute of the nodes in the dual-graph file.
        dual_graph_path (str): The path to the dual-graph file to read from.
        input_file_path (str, optional): The path to the input file to read from. Only
            required if running in 'ben' mode. Defaults to None.
        output_file_path (str, optional): The path to the output file to write to. If not
            given, the output will be determined according to a set of heuristics. Defaults
            to None.
        verbose (bool, optional): Whether to run the program in verbose mode. Defaults to True.
        docker_image_name (str, optional): The name of the Docker image to run the program in.
            Defaults to "mgggdev/replicate:v0.2".
        docker_client_args (dict, optional): Additional arguments to pass to the Docker client.
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

    dual_graph_parent_path = Path(dual_graph_path).parent
    dual_graph_name = Path(dual_graph_path).name

    if output_file_path is not None:
        output_parent_path = Path(output_file_path).parent
        output_file_name = Path(output_file_path).name
    else:
        output_parent_path = dual_graph_parent_path
        output_file_name = None

    os.makedirs(output_parent_path, exist_ok=True)

    config_args = {
        "name": "reben_runner",
        "image": docker_image_name,
        "detach": True,
        "auto_remove": True,
        "tty": True,
        "stdin_open": True,
    }

    if dual_graph_parent_path == output_parent_path:
        config_args["volumes"] = {
            dual_graph_parent_path.resolve(): {
                "bind": "/home/ben/io",
                "mode": "rw",
            },
        }
        docker_output_path = "/home/ben/io"
        docker_dual_graph_path = "/home/ben/io"

    else:
        config_args["volumes"] = {
            output_parent_path.resolve(): {
                "bind": "/home/ben/output",
                "mode": "rw",
            },
            dual_graph_parent_path.resolve(): {
                "bind": "/home/ben/dual_graph",
                "mode": "rw",
            },
        }

        docker_output_path = "/home/ben/output"
        docker_dual_graph_path = "/home/ben/dual_graph"

    with managed_docker_container(client, config_args) as container:
        reben_cmd = f"reben {docker_dual_graph_path}/{dual_graph_name} -m json -k {key}"

        if verbose:
            reben_cmd += " -v"

        if output_file_name is not None:
            reben_cmd += f" -o {docker_output_path}/{output_file_name}"

        cmd = ["sh", "-c", f"{reben_cmd}"]

        logging.debug(f"Running command: {cmd}")
        exec_id = client.api.exec_create(
            container.id, cmd=cmd, tty=False, stdout=True, stderr=True, stdin=False
        )

        output_generator = client.api.exec_start(
            exec_id=exec_id, stream=True, detach=False
        )

        for output in output_generator:
            print(output.decode("utf-8"), end="")

    client.close()


def relabel_ben_file_by_key(
    key: str,
    dual_graph_path: str,
    input_file_path: str,
    output_file_path: Optional[str] = None,
    verbose: bool = True,
    docker_image_name: str = "mgggdev/replicate:v0.2",
    docker_client_args: Optional[dict] = None,
):
    """
    Runs the REBEN CLI tool from the
    `binary-ensemble <https://crates.io/crates/binary-ensemble>`_
    crate in a Docker container. This function is specifically designed to run the
    REBEN tool in its relabeling mode.

    This function expects the input file to be a BEN file will relabel the input dual-graph
    according to the key value provided and output a new dual-graph file together with a map
    file in the event that the user would like to recover the original file from the relabeled
    file. This function will then relabel teh assignment vectors in the BEN file according to
    the new ordering of the nodes in the remapped dual-graph file and write that to the
    `output_file_path` or a file derived from the input file name according to a set of
    heuristics.

    Args:
        key (str): The key to relabel the dual-graph with. This should appear as an
            attribute of the nodes in the dual-graph file.
        dual_graph_path (str): The path to the dual-graph file to read from.
        input_file_path (str, optional): The path to the input file to read from. Only
            required if running in 'ben' mode. Defaults to None.
        output_file_path (str, optional): The path to the output file to write to. If not
            given, the output will be determined according to a set of heuristics. Defaults
            to None.
        verbose (bool, optional): Whether to run the program in verbose mode. Defaults to True.
        docker_image_name (str, optional): The name of the Docker image to run the program in.
            Defaults to "mgggdev/replicate:v0.2".
        docker_client_args (dict, optional): Additional arguments to pass to the Docker client.
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
        "name": "reben_runner",
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

    with managed_docker_container(client, config_args) as container:
        reben_cmd = f"reben {docker_input_path}/{input_file_name} -m ben -k {key} -s {docker_dual_graph_path}/{dual_graph_name}"

        if verbose:
            reben_cmd += " -v"

        if output_file_name is not None:
            reben_cmd += f" -o {docker_output_path}/{output_file_name}"

        cmd = ["sh", "-c", f"{reben_cmd}"]

        logging.debug(f"Running command: {cmd}")
        exec_id = client.api.exec_create(
            container.id, cmd=cmd, tty=False, stdout=True, stderr=True, stdin=False
        )

        output_generator = client.api.exec_start(
            exec_id=exec_id, stream=True, detach=False
        )

        for output in output_generator:
            print(output.decode("utf-8"), end="")

    client.close()


def relabel_ben_file_with_map(
    map_file_path: str,
    input_file_path: str,
    output_file_path: Optional[str] = None,
    verbose: bool = True,
    docker_image_name: str = "mgggdev/replicate:v0.2",
    docker_client_args: Optional[dict] = None,
):
    """
    Runs the REBEN CLI tool from the
    `binary-ensemble <https://crates.io/crates/binary-ensemble>`_
    crate in a Docker container. This function is specifically designed to run the
    REBEN tool in its map-file mode.

    This function expects the input file to be a BEN file and will use the map file to
    relabel the assignment vectors in the BEN file according to the new ordering of the
    nodes in the map file.

    Args:
        map_file_path (str): The path to the map file to read from. This generally
            appears as the output of running REBEN in the 'json' mode, or as an output
            of the :func:`relabel_json_file_by_key` or :func:`relabel_ben_file_by_key`
            functions.
        input_file_path (str): The path to the input file to read from. This is expected
            to be a BEN file.
        output_file_path (str, optional): The path to the output file to write to. If not
            given, the output will be determined according to a set of heuristics. Defaults
            to None.
        verbose (bool, optional): Whether to run the program in verbose mode. Defaults to True.
        docker_image_name (str, optional): The name of the Docker image to run the program in.
            Defaults to "mgggdev/replicate:v0.2".
        docker_client_args (dict, optional): Additional arguments to pass to the Docker client.
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

    map_file_parent_path = Path(map_file_path).parent
    map_file_name = Path(map_file_path).name

    if output_file_path is not None:
        output_parent_path = Path(output_file_path).parent
        output_file_name = Path(output_file_path).name
    else:
        output_parent_path = input_parent_path
        output_file_name = None

    os.makedirs(output_parent_path, exist_ok=True)

    config_args = {
        "name": "reben_runner",
        "image": docker_image_name,
        "detach": True,
        "auto_remove": True,
        "tty": True,
        "stdin_open": True,
    }

    if (
        input_parent_path == output_parent_path
        and input_parent_path == map_file_parent_path
    ):
        config_args["volumes"] = {
            input_parent_path.resolve(): {
                "bind": "/home/ben/io",
                "mode": "rw",
            },
        }
        docker_input_path = "/home/ben/io"
        docker_output_path = "/home/ben/io"
        docker_map_file_path = "/home/ben/io"

    # Now the map file path is different from the input path
    elif input_parent_path == output_parent_path:
        config_args["volumes"] = {
            input_parent_path.resolve(): {
                "bind": "/home/ben/io",
                "mode": "rw",
            },
            map_file_parent_path.resolve(): {
                "bind": "/home/ben/map_file",
                "mode": "rw",
            },
        }
        docker_input_path = "/home/ben/io"
        docker_output_path = "/home/ben/io"
        docker_map_file_path = "/home/ben/map_file"

    # Now the map file path might be the same as one of the paths, but
    # the input and output paths are different
    elif map_file_parent_path == input_parent_path:
        config_args["volumes"] = {
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
        docker_map_file_path = "/home/ben/input"

    elif map_file_parent_path == output_parent_path:
        config_args["volumes"] = {
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
        docker_map_file_path = "/home/ben/output"

    # Now all the paths are different
    else:
        config_args["volumes"] = {
            input_parent_path.resolve(): {
                "bind": "/home/ben/input",
                "mode": "rw",
            },
            output_parent_path.resolve(): {
                "bind": "/home/ben/output",
                "mode": "rw",
            },
            map_file_parent_path.resolve(): {
                "bind": "/home/ben/map_file",
                "mode": "rw",
            },
        }

        docker_input_path = "/home/ben/input"
        docker_output_path = "/home/ben/output"
        docker_map_file_path = "/home/ben/map_file"

    with managed_docker_container(client, config_args) as container:
        reben_cmd = f"reben {docker_input_path}/{input_file_name} -m ben -p {docker_map_file_path}/{map_file_name}"

        if verbose:
            reben_cmd += " -v"

        if output_file_name is not None:
            reben_cmd += f" -o {docker_output_path}/{output_file_name}"

        cmd = ["sh", "-c", f"{reben_cmd}"]

        logging.debug(f"Running command: {cmd}")
        exec_id = client.api.exec_create(
            container.id, cmd=cmd, tty=False, stdout=True, stderr=True, stdin=False
        )

        output_generator = client.api.exec_start(
            exec_id=exec_id, stream=True, detach=False
        )

        for output in output_generator:
            print(output.decode("utf-8"), end="")

    client.close()
