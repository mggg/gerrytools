import io
import logging

import pytest

from gerrytools.logging import TRACE, configure_logging, get_logger


@pytest.fixture
def restore_gerrytools_logger():
    logger = get_logger()
    handlers = list(logger.handlers)
    level = logger.level
    propagate = logger.propagate
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_reconfigure_changes_level_without_replacing_stream_or_format(
    restore_gerrytools_logger,
):
    del restore_gerrytools_logger
    # Regression: the handler used to pin its own level at creation, so a second
    # configure_logging call could not lower or raise the effective level.
    first_stream = io.StringIO()
    logger = configure_logging("INFO", stream=first_stream, force=True)
    (handler,) = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]

    logger.debug("hidden")
    assert first_stream.getvalue() == ""

    second_stream = io.StringIO()
    configure_logging("DEBUG", stream=second_stream, fmt="%(message)s")

    # Same handler, still writing to the first stream, but the new level applies.
    (handler_after,) = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert handler_after is handler
    assert handler_after.stream is first_stream
    assert handler_after.level == logging.NOTSET

    logger.debug("now visible")
    assert "now visible" in first_stream.getvalue()
    assert second_stream.getvalue() == ""


def test_force_removes_prior_handlers_and_recreates_console_handler(restore_gerrytools_logger):
    del restore_gerrytools_logger
    logger = get_logger()
    user_handler = logging.Handler()
    logger.addHandler(user_handler)
    first_stream = io.StringIO()
    configure_logging("INFO", stream=first_stream, force=True)
    assert user_handler not in logger.handlers

    second_stream = io.StringIO()
    configure_logging("INFO", stream=second_stream, force=True)

    logger.info("after force")
    assert "after force" in second_stream.getvalue()
    assert first_stream.getvalue() == ""


def test_configure_logging_rejects_unknown_level_string(restore_gerrytools_logger):
    del restore_gerrytools_logger
    # Regression: an unknown level string used to fall back silently to INFO.
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("VERBOSE")


def test_configure_logging_accepts_trace(restore_gerrytools_logger):
    del restore_gerrytools_logger
    assert configure_logging("TRACE", force=True).level == TRACE


def test_pre_attached_file_handler_does_not_suppress_console_handler(
    tmp_path, restore_gerrytools_logger
):
    del restore_gerrytools_logger
    # Regression: FileHandler subclasses StreamHandler, so an isinstance StreamHandler check
    # treated a user-attached file handler as the console handler and never created one.
    logger = get_logger()
    file_handler = logging.FileHandler(tmp_path / "gerrytools.log")
    logger.addHandler(file_handler)

    stream = io.StringIO()
    configure_logging("INFO", stream=stream)

    logger.info("to console")
    assert "to console" in stream.getvalue()


def test_env_var_supplies_level_when_level_is_none(monkeypatch, restore_gerrytools_logger):
    del restore_gerrytools_logger
    monkeypatch.setenv("GERRYTOOLS_LOG_LEVEL", "debug")
    stream = io.StringIO()

    logger = configure_logging(stream=stream, force=True)

    assert logger.level == logging.DEBUG
    logger.debug("env visible")
    assert "env visible" in stream.getvalue()


def test_get_logger_only_preserves_the_gerrytools_namespace():
    # Regression: the prefix check used startswith("gerrytools"), so a name like
    # "gerrytools_extra" escaped the package namespace.
    assert get_logger().name == "gerrytools"
    assert get_logger("gerrytools").name == "gerrytools"
    assert get_logger("gerrytools.sub").name == "gerrytools.sub"
    assert get_logger("other").name == "gerrytools.other"
    assert get_logger("gerrytools_extra").name == "gerrytools.gerrytools_extra"
