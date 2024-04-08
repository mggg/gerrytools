from contextlib import contextmanager
import docker


@contextmanager
def managed_docker_container(docker_client, config_args):
    """
    Context manager for running a docker container. This helps to make
    sure that the container is properly removed after it is done running
    or if an error is encountered in the middle of running the container.

    Args:
        docker_client: The docker client that will be used to run the container.
        config_args: The arguments that will be passed to docker_client.containers.run.
    """
    container = None
    try:
        container = docker_client.containers.run(**config_args)
        print(f"Running container {container.name}")
        yield container
    except Exception as e:
        print(f"Error running container: {e}")
    finally:
        if container:
            try:
                container.remove(force=True)
            except docker.errors.APIError as e:
                print(f"Error removing container: {e}")
