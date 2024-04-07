import docker
import traceback
from abc import ABC, abstractmethod
from typing import Union, Optional, Type
from types import TracebackType
import json
from gerrychain import Graph, Partition
import os


class RunnerConfig(ABC):
    """
    An abstract class to define the methods that a runner
    variant must implement to be used with the RunContainer
    """

    @abstractmethod
    def configure_vols_and_name(self):
        """
        Configures the volumes and name for the Docker container to be run
        """
        pass

    @abstractmethod
    def run_command(self, *args, **kwargs):
        """
        Configures the command to be run in the Docker container
        """
        pass

    @abstractmethod
    def log_file(self, *args, **kwargs):
        """
        Configures the log file to be written to when the Docker container
        runs the command.
        """
        pass


class RunContainer:
    """
    A class to control a docker container.

    Example:

        with Replicator(variant) as rep:
            rep.run(run_information)

    The __enter__ and __exit__ magic methods help control
    the docker container and take care of instantiation and
    clean up for the end user.
    """

    def __init__(
        self,
        configuration: RunnerConfig,
        docker_image_name="mgggdev/replicate:latest",
        docker_client_args: dict = None,
    ):
        """
        Sets up the replicator class

        Args:
            variant (RunnerConfig): Type of runner with setup to use
            docker_image_name (str, optional): Override for the docker image to
                use when building the Docker container. Defaults to None.

        Raises:
            ValueError: When the type of runner is not RecomRunnerConfig, ForestRunner,
                or SMC Runner
        """
        if isinstance(
            configuration,
            RunnerConfig,
        ):
            self.config = configuration
        else:
            raise ValueError(
                f"Invalid variant passed. Expected "
                f"RecomRunnerConfig, ForestRunner, or SMCRunner ",
                f"and found {type(configuration)}",
            )

        if docker_client_args is not None:
            self.client = docker.DockerClient(**docker_client_args)
        else:
            self.client = docker.from_env()
        self.container = None
        self.image_name = docker_image_name

    def __enter__(self):
        """
        Magic method to control the Docker context

        Raises:
            KeyError: When the docker image is not defined properly in the runner
        """
        vols_and_name = self.config.configure_vols_and_name()

        config_args = vols_and_name | {
            "image": self.image_name,
            "detach": True,
            "auto_remove": False,  # DONT FORGET TO CHANGE THIS BACK TO TRUE
            "tty": True,
            "stdin_open": True,
        }

        try:
            print(f"Pulling Docker image {config_args['image']}")
            self.client.images.pull(config_args["image"])
        except Exception as e:
            print(
                f"Error comparing docker container {config_args['image']} against web version. "
                f"Attempting to run using local image"
            )

        try:
            self.container = self.client.containers.run(**config_args)
            print(f"Running Docker container {self.container.name}")
        except Exception as e:
            print(f"Error running Docker container: {e}")

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[Exception]],
        exc_value: str,
        traceback_obj: Optional[TracebackType],
    ):
        """
        Magic method to close the Docker container when run is finished.
        Helps keep Docker from accumulating undesired containers and bloating
        the user's computer in case they do not know to clean out their
        containers periodically.

        Args:
            exc_type: Type of exception raised
            exc_value: Value of exception raised
            traceback_obj: Traceback object

        Returns:
            False: If an exception is raised, the error message will be printed
                and the function will return False
        """
        self.client.close()
        if self.container:
            try:
                self.container.remove(force=True)
            except docker.errors.APIError as e:
                print(f"Error removing Docker container: {e}")
                if traceback_obj:
                    formatted_traceback = "".join(traceback.format_tb(traceback_obj))
                    print("Traceback for Docker container removal error:")
                    print(formatted_traceback)

        if exc_type is not None:
            print(f"An exception of type {exc_type} occurred: {exc_value}")
            if traceback_obj:
                formatted_traceback = "".join(traceback.format_tb(traceback_obj))
                print("Traceback for the with-block error:")
                print(formatted_traceback)

        return False

    def run(self, *args, **kwargs):
        """
        Calls the run method of the provided runner variant with
        the given arguments. Anything printed to the stderr in the container
        will be printed to the log file, and any output generated that is not
        sent to an output file will be printed to the console.

        Args:
            *args: Variable length argument list.
            **kwargs: Variable length keyword argument list.
        """
        if not hasattr(self.config, "run_command"):
            raise NotImplementedError(
                f"The runner of type {type(self.config)} does not have "
                f"an implemented run method."
            )

        cmd = self.config.run_command(*args, **kwargs)
        log_file = self.config.log_file(*args, **kwargs)
        exec_id = self.client.api.exec_create(
            self.container.id,
            cmd=cmd,
            tty=False,
            stdout=True,
            stderr=True,
            stdin=False,
        )

        output_generator = self.client.api.exec_start(
            exec_id,
            stream=True,
            detach=False,
            demux=True,
        )

        with open(log_file, "w") as f:
            for output in output_generator:
                if output[0] is not None:
                    print(output[0].decode("utf-8"), end="")
                if output[1] is not None:
                    f.write(output[1].decode("utf-8"))
                    f.flush()  # Ensure the output is written immediately

    def run_iter(self, *args, **kwargs):
        """
        Calls the run method of the provided runner variant with
        the given arguments

        Args:
            *args: Variable length argument list.
            **kwargs: Variable length keyword argument list.

        Yields:
            Tuple[Dict, str]: Dictionary of the sample number and updater values and the
            error message (if any)
        """
        if not hasattr(self.config, "run_command"):
            raise NotImplementedError(
                f"The runner of type {type(self.config)} does not have "
                f"an implemented run method."
            )

        cmd = self.config.run_command(*args, **kwargs)
        log_file = self.config.log_file(*args, **kwargs)
        exec_id = self.client.api.exec_create(
            self.container.id,
            cmd=cmd,
            tty=False,
            stdout=True,
            stderr=True,
            stdin=False,
        )

        output_generator = self.client.api.exec_start(
            exec_id,
            stream=True,
            detach=False,
            demux=True,
        )

        # Sometimes the output is split between multiple lines
        # So this will take care of that issue by accumulating the
        # output until a newline is reached
        stdout_buffer = ""
        for stdout, stderr in output_generator:
            if stdout:
                stdout_buffer += stdout.decode("utf-8")
                # Process any complete JSON objects in the buffer
                while "\n" in stdout_buffer and "}" in stdout_buffer:
                    newline_index = stdout_buffer.find("\n")
                    possible_json = stdout_buffer[:newline_index]

                    if "}" in possible_json:
                        try:
                            json_obj = json.loads(possible_json)
                            yield (json_obj, stderr.decode("utf-8") if stderr else None)
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON: {possible_json}")
                            exit(1)

                        stdout_buffer = stdout_buffer[newline_index + 1 :]
                    else:
                        # If no complete JSON object, wait for more data
                        break

            elif stderr:
                yield (None, stderr.decode("utf-8"))

