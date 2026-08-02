import logging
import os
import sys
from typing import TextIO

_GERRYTOOLS_LOGGER_NAME = "gerrytools"
TRACE = logging.DEBUG - 5
logging.addLevelName(TRACE, "TRACE")

_LEVELS_BY_NAME = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "TRACE": TRACE,
    "NOTSET": logging.NOTSET,
}

# Library-safe default: don't configure global logging on import.
logging.getLogger(_GERRYTOOLS_LOGGER_NAME).addHandler(logging.NullHandler())


class _GerrytoolsConsoleHandler(logging.StreamHandler):
    """Marker type for the console handler ``configure_logging`` creates.

    Detection by this type, rather than ``isinstance(..., StreamHandler)``, keeps a user-attached
    handler that happens to subclass StreamHandler (e.g. ``FileHandler``) from suppressing console
    handler creation.
    """


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger namespaced under ``gerrytools``.

    Args:
        name (str | None, optional): Logger name. An existing ``gerrytools`` namespace is
            preserved; another name is prefixed with ``"gerrytools."``. Defaults to None, which
            returns the package logger.

    Returns:
        logging.Logger: Requested package logger.
    """
    if name is None:
        return logging.getLogger(_GERRYTOOLS_LOGGER_NAME)
    if name == _GERRYTOOLS_LOGGER_NAME or name.startswith(f"{_GERRYTOOLS_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_GERRYTOOLS_LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str | None = None,
    *,
    stream: TextIO | None = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    force: bool = False,
) -> logging.Logger:
    """Configure package logging for an application, CLI, or notebook.

    This function is safe to call repeatedly.

    Args:
        level (int | str | None, optional): Numeric level or standard level name. Defaults to the
            ``GERRYTOOLS_LOG_LEVEL`` environment variable, or ``"INFO"`` when it is unset.
        stream (TextIO | None, optional): Destination for a newly created handler. Defaults to
            ``sys.stderr``.
        fmt (str, optional): Format for a newly created handler. Defaults to the package's
            timestamped console format.
        datefmt (str, optional): Date format for a newly created handler. Defaults to
            ``"%Y-%m-%d %H:%M:%S"``.
        force (bool, optional): Whether to remove existing package handlers first. Defaults to
            False.

    Returns:
        logging.Logger: Configured ``gerrytools`` package logger.

    Raises:
        ValueError: If a string ``level`` is not a recognized logging level.

    Notes:
        This helper installs a package-local console handler and disables propagation to the root
        logger to avoid duplicate records. Applications that centralize logging through root
        handlers should configure those handlers directly instead of calling this helper.
    """
    if level is None:
        level = os.getenv("GERRYTOOLS_LOG_LEVEL", "INFO")

    if isinstance(level, str):
        try:
            normalized_level = _LEVELS_BY_NAME[level.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown log level {level!r}; expected one of {sorted(_LEVELS_BY_NAME)}."
            ) from None
    else:
        normalized_level = level

    if stream is None:
        stream = sys.stderr

    logger = logging.getLogger(_GERRYTOOLS_LOGGER_NAME)
    logger.setLevel(normalized_level)

    if force:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    # The handler's level stays NOTSET, deferring to the logger level, so repeat calls can
    # change the effective level without touching the existing handler.
    if not any(isinstance(h, _GerrytoolsConsoleHandler) for h in logger.handlers):
        handler = _GerrytoolsConsoleHandler(stream)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)

    logger.propagate = False
    return logger
