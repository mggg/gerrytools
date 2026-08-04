"""Opt-in fixtures for live mgrp Docker tests."""

import os

import pytest
from docker.errors import DockerException

import docker


@pytest.fixture(scope="session")
def mgrp_image():
    image_name = os.environ.get("MGRP_TEST_IMAGE")
    if not image_name:
        pytest.skip("set MGRP_TEST_IMAGE to run live mgrp tests")

    client = None
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(image_name)
    except DockerException as error:
        pytest.skip(f"Docker image {image_name!r} is not available locally: {error}")
    finally:
        if client is not None:
            client.close()
    return image_name
