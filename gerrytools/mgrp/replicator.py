from . import RecomRunner, ForestRunner, SMCRunner, RecomDataRunner
import docker
import traceback


class Replicator:
    """
    Class to help control a docker container.
    Usually called with
        with Replicator(variant) as rep:
            <more code>
    The __enter__ and __exit__ magic methods help control
    the docker container and take care of instantiation and
    clean up for the end user.
    """

    def __init__(self, variant, docker_image_name=None):
        """
        Sets up the replicator class

        Args:
            variant: Type of runner with setup to use
            docker_image_name (str, optional): Override for the docker image to
                use when building the Docker container. Defaults to None.

        Raises:
            ValueError: When the type of runner is not RecomRunner, ForestRuner,
                or SMC Runner
        """
        if isinstance(variant, (RecomRunner, RecomDataRunner, ForestRunner, SMCRunner)):
            self.runner = variant
        else:
            raise ValueError(
                f"Invalid variant passed. Expected "
                f"RecomRunner, ForestRunner, or SMCRunner ",
                f"and found {type(variant)}",
            )

        self.client = docker.from_env()
        self.container = None
        self.image_name = docker_image_name

    def __enter__(self):
        """
        Magic method to control the Docker context

        Raises:
            KeyError: When the docker image is not defined properly in the runner
        """
        if self.image_name is not None:
            config_args = self.runner.configure(image_name=self.image_name)
        else:
            config_args = self.runner.configure()

        if "image" in config_args.keys():
            try:
                print(f"Pulling Docker image {config_args['image']}")
                self.client.images.pull(config_args["image"])
            except Exception as e:
                print(
                    f"Error comparing docker container {config_args['image']} against web version. "
                    f"Attempting to run using local image"
                )

        else:
            raise KeyError("Runner configuration does not contain a Docker image name.")

        try:
            self.container = self.client.containers.run(**config_args)
        except Exception as e:
            print(f"Error running Docker container: {e}")

        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        """
        Magic method to close the Docker container when run is finished.
        Helps keep Docker from accumulating undesired containers and bloating
        the user's computer in case they do not know to clean out their
        containers periodically.
        """
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
        the given arguments
        *args: Variable length argument list.
        **kwargs: Variable length keyword argument list.
        """
        if not hasattr(self.runner, "run"):
            raise NotImplementedError(
                f"The runner of type {type(self.runnner)} does not have "
                f"an implemented run method."
            )
        return self.runner.run(self.container, *args, **kwargs)
