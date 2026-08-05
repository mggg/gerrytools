"""Shared fake-Docker wiring for the Docker-free mgrp tests."""

from typing import Any, cast

from gerrytools.mgrp import RunnerSession


def make_fake_run_container(config=None, client=None, container=None) -> RunnerSession:
    """A RunnerSession built without ``__init__`` (so no Docker connection), wired to fakes.

    Args:
        config: Runner configuration or a stand-in namespace; None for tests that never touch it.
        client: Fake Docker client (e.g. a SimpleNamespace with an ``api`` attribute).
        container: Fake started container (e.g. a SimpleNamespace with an ``id``).
    """
    run_container = RunnerSession.__new__(RunnerSession)
    run_container.config = cast(Any, config)
    run_container.client = cast(Any, client)
    run_container.container = cast(Any, container)
    run_container.image_name = "image:test"
    return run_container
