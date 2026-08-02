from __future__ import annotations

import codecs
import hashlib
import json
import logging
import os
import sys
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from types import TracebackType
from typing import ClassVar, Generic, Iterator, Protocol, TypeVar, cast, runtime_checkable

import docker.errors
from gerrychain import Graph, Partition

logger = logging.getLogger(__name__)

DEFAULT_DOCKER_IMAGE = "mgggdev/replicate:v2.0.0@sha256:b9243d65bfce934dcb1318a509388f9b9f25eacf13ab4ec4eefddf8e566fc1f6"

BINARY_WRITERS = ("pcompress", "ben", "bendl")
"""Writers whose output cannot be decoded from a console stream."""


class RunInfo:
    """Base class for validated runner configuration dataclasses."""

    def validate(self) -> None:
        """Validate the current field values before they cross the runner boundary.

        Raises:
            NotImplementedError: Always; subclasses must implement validation.
        """
        raise NotImplementedError

    @staticmethod
    def validate_force_print(writer: str, force_print: bool) -> None:
        """Reject binary writers when output would be decoded as console text.

        Args:
            writer (str): Selected output writer.
            force_print (bool): Whether output is forced to the console stream.

        Raises:
            ValueError: If console output is requested for a binary writer.
        """
        if force_print and writer in BINARY_WRITERS:
            raise ValueError(
                f"force_print is not supported with the {writer} writer; its binary "
                "output cannot be decoded from the console stream."
            )


WRITER_SUFFIXES = {
    "pcompress": "_pcompress.chain",
    "assignments": ".assignments",
    "tsv": ".tsv",
    "ben": ".jsonl.ben",
    "bendl": ".bendl",
    "csv": ".csv",
    "raw": ".atlas",
}
"""Output suffixes that differ from the default JSONL suffix."""


RunInfoT = TypeVar("RunInfoT", bound=RunInfo)
"""The run-info type a runner configuration (and its container) accepts."""


def _partial_output_path(output: Path) -> Path:
    """Return an unused sibling name for output from a failed run."""
    candidate = output.with_name(f"{output.name}.partial")
    suffix = 1
    while candidate.exists() or candidate.is_symlink():
        candidate = output.with_name(f"{output.name}.partial.{suffix}")
        suffix += 1
    return candidate


@contextmanager
def _preserve_outputs_on_failure(expected_files: list[str]) -> Iterator[None]:
    """Hide old outputs during a rerun and preserve failed output under partial names."""
    outputs = [Path(output) for output in expected_files]
    backup_dirs: dict[Path, Path] = {}
    backups: list[tuple[Path, Path]] = []
    run_started = False

    try:
        for output in outputs:
            if not output.exists() and not output.is_symlink():
                continue
            if not output.is_file() and not output.is_symlink():
                raise IsADirectoryError(f"Expected output path is not a file: {output}")
            backup_dir = backup_dirs.get(output.parent)
            if backup_dir is None:
                backup_dir = Path(tempfile.mkdtemp(prefix=".gerrytools-backup-", dir=output.parent))
                backup_dirs[output.parent] = backup_dir
            backup = backup_dir / output.name
            output.replace(backup)
            backups.append((output, backup))

        run_started = True
        yield
    except BaseException as run_error:
        recovery_errors: list[BaseException] = []
        partials: list[Path] = []
        if run_started:
            for output in outputs:
                try:
                    if not output.exists() and not output.is_symlink():
                        continue
                    if not output.is_file() and not output.is_symlink():
                        raise IsADirectoryError(f"Expected output path is not a file: {output}")
                    partial = _partial_output_path(output)
                    output.replace(partial)
                    partials.append(partial)
                except BaseException as error:
                    recovery_errors.append(error)
        for output, backup in backups:
            try:
                backup.replace(output)
            except BaseException as error:
                recovery_errors.append(error)
        for backup_dir in backup_dirs.values():
            try:
                backup_dir.rmdir()
            except BaseException as error:
                recovery_errors.append(error)
        if partials:
            run_error.add_note(
                "Partial output from the failed run was preserved at: "
                + ", ".join(str(path) for path in partials)
            )
        if recovery_errors:
            raise BaseExceptionGroup(
                "The run failed and its previous outputs could not be fully restored",
                [run_error, *recovery_errors],
            )
        raise
    else:
        for _, backup in backups:
            backup.unlink()
        for backup_dir in backup_dirs.values():
            backup_dir.rmdir()


@contextmanager
def _preserve_log_on_failure(log_file: str) -> Iterator[None]:
    """Back up an existing log during a rerun and restore it if the rerun fails.

    Unlike outputs, a failed run's fresh log is kept when there was no previous log to
    restore: the captured stderr is the primary failure diagnostic.
    """
    log_path = Path(log_file)
    if not log_path.is_file():
        yield
        return
    backup_dir = Path(tempfile.mkdtemp(prefix=".gerrytools-backup-", dir=log_path.parent))
    backup = backup_dir / log_path.name
    log_path.replace(backup)
    try:
        yield
    except BaseException:
        try:
            backup.replace(log_path)
            backup_dir.rmdir()
        except Exception as error:
            logger.warning(f"Could not restore the previous log file {log_file}: {error}")
        raise
    else:
        backup.unlink()
        backup_dir.rmdir()


def _validate_output_file_name(name: str, *, field: str = "output_file_name") -> str:
    """Require a bare file name so outputs stay inside the run's output folder.

    Separator or traversal components would escape the mounted output directory, and an
    absolute path would make the host and container paths identify different locations.
    """
    if (
        name in ("", ".", "..")
        or "/" in name
        or "\\" in name
        or Path(name).is_absolute()
        or bool(PureWindowsPath(name).drive)
    ):
        raise ValueError(
            f"{field} must be a bare file name inside the output folder, but found {name!r}."
        )
    return name


def _resolve_output_name(stem: str, writer: str, output_file_name: str | None = None) -> str:
    """Resolve an output file name; an explicit override must be a bare file name.

    Every runner's output naming routes through here, so a post-construction
    ``output_file_name`` override is re-validated whenever the name is used.
    """
    if output_file_name is not None:
        return _validate_output_file_name(output_file_name)
    return stem + WRITER_SUFFIXES.get(writer, ".jsonl")


@runtime_checkable
class SupportsUpdaters(Protocol):
    """Run infos usable with :meth:`RunContainer.mcmc_run_with_updaters`."""

    @property
    def updaters(self) -> dict[str, Callable]:
        """Return updater names and callables for the recorded chain."""
        ...


class RunnerConfig(ABC, Generic[RunInfoT]):
    """Shared path, volume, and naming plumbing for the engine runner configurations.

    Concrete runners supply the engine name (which derives the ``/home/<engine>``
    container prefixes), the per-run file stem, output naming, and the run command.
    Each is generic over the run-info dataclass it accepts, so the naming and
    command hooks receive that concrete type directly.
    """

    parser_name: ClassVar[str | None] = None
    """Container-side parser binary piped after the engine's stdout, or None."""

    run_info_type: ClassVar[type[RunInfo] | tuple[type[RunInfo], ...]]
    """Exact run-info type or types accepted by this runner."""

    def __init__(
        self,
        engine: str,
        input_path: str,
        output_folder: str = "./output",
        log_folder: str = "./logs",
    ) -> None:
        """Resolve host paths and derive the container mount layout.

        Args:
            engine (str): The engine name; container paths live under ``/home/<engine>``.
            input_path (str): Host path of the run's input (a dual-graph JSON file, or a
                shapefile bundle directory for SMC).
            output_folder (str): Host directory for output files; a subfolder named after the
                input's stem is created inside it. Defaults to "./output".
            log_folder (str): Host directory for log files. Defaults to "./logs".
        """
        self.engine = engine
        resolved_input = Path(input_path).resolve()
        self.input_dir = resolved_input.parent
        self.input_name = resolved_input.name
        self.input_stem = resolved_input.stem
        self.output_folder = Path(output_folder).resolve() / self.input_stem
        self.log_folder = Path(log_folder).resolve()
        self.container_input_dir = f"/home/{engine}/shapefiles"
        self.container_graph_path = f"{self.container_input_dir}/{self.input_name}"
        self.container_output_dir = f"/home/{engine}/output/{self.input_stem}"

    def _check_run_info(self, run_info: RunInfo) -> None:
        """Reject run infos whose exact type is not supported by this runner."""
        accepted = (
            self.run_info_type if isinstance(self.run_info_type, tuple) else (self.run_info_type,)
        )
        if type(run_info) not in accepted:
            expected = " or ".join(run_info_type.__name__ for run_info_type in accepted)
            raise TypeError(
                f"{type(self).__name__} requires a {expected} run info, "
                f"but found {type(run_info).__name__}."
            )
        run_info.validate()

    @staticmethod
    def _writer_output_name(
        stem: str,
        writer: str,
        output_file_name: str | None = None,
        *,
        force_print: bool = False,
    ) -> str | None:
        """Resolve an output name, or None when the run prints to stdout instead."""
        if force_print:
            return None
        return _resolve_output_name(stem, writer, output_file_name)

    @property
    def host_graph_path(self) -> str:
        """Host path of the input file the container graph path mirrors."""
        return str(self.input_dir / self.input_name)

    def configure_volumes(self) -> dict:
        """The Docker volume configuration mounting the input and output folders.

        The container name is deliberately not set here: Docker generates one, so
        concurrent runs and reruns after a crash never collide on a fixed name.

        Returns:
            dict: Docker SDK volume configuration with read-only input and writable output mounts.
        """
        # Create the output folder before Docker mounts it; otherwise Docker
        # creates it owned by root and the user cannot delete their own results.
        os.makedirs(self.output_folder, exist_ok=True)

        return {
            "volumes": {
                str(self.input_dir): {
                    "bind": self.container_input_dir,
                    # Engine configs route every write to the separate output mount.
                    "mode": "ro",
                },
                str(self.output_folder): {
                    "bind": self.container_output_dir,
                    "mode": "rw",
                },
            },
        }

    @abstractmethod
    def _stem(self, run_info: RunInfoT) -> str:
        """The human-readable stem derived from the run's headline settings, without the hash."""

    @abstractmethod
    def _output_name(self, run_info: RunInfoT) -> str | None:
        """The name of the file the run will produce, or None when it prints to stdout."""

    @abstractmethod
    def _base_config(self, run_info: RunInfoT) -> Mapping[str, object]:
        """The engine config document built with hash-free file names, used for hashing."""

    def config_hash(self, run_info: RunInfoT) -> str:
        """Short deterministic digest of the run's full engine configuration and input file.

        Folded into every derived file stem, so runs whose configs differ outside the
        human-readable stem never collide on output or log paths, while an identical rerun
        reuses (and overwrites) its own. The hashed document uses hash-free file names, which
        are themselves pure functions of the configuration, so the digest is well defined.

        The resolved host path distinguishes input graphs that share a basename.

        Args:
            run_info (RunInfoT): Validated settings for one run.

        Returns:
            str: First ten hexadecimal characters of the configuration digest.

        Raises:
            TypeError: If ``run_info`` has the wrong concrete type for this runner.
            ValueError: If its settings cannot be validated or serialized as strict JSON.
        """
        self._check_run_info(run_info)
        payload = json.dumps(
            {"config": self._base_config(run_info), "host_graph_path": self.host_graph_path},
            sort_keys=True,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    def file_stem(self, run_info: RunInfoT) -> str:
        """The output/log file stem shared by every artifact of a run: the human-readable
        stem plus the config hash.

        Args:
            run_info (RunInfoT): Validated settings for one run.

        Returns:
            str: Human-readable run stem followed by its configuration digest.
        """
        self._check_run_info(run_info)
        return f"{self._stem(run_info)}_{self.config_hash(run_info)}"

    @abstractmethod
    def run_command(self, run_info: RunInfoT) -> list:
        """Return the command to execute in the Docker container.

        Args:
            run_info (RunInfoT): Validated settings for one run.

        Returns:
            list: Command arguments passed to the container.
        """

    def canonical_stdout_command(self, run_info: RunInfoT) -> list:
        """The argv for a run forced to canonical assignment output on stdout.

        Overridden by the MCMC runners; defensive here, since
        :meth:`RunContainer.mcmc_run_with_updaters` rejects run infos without updaters
        before this base implementation can be reached.

        Args:
            run_info (RunInfoT): Validated settings for one run.

        Returns:
            list: Command arguments that emit canonical assignments to standard output.

        Raises:
            NotImplementedError: If the runner does not support canonical standard output.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support canonical stdout runs.")

    def _shell_command(
        self,
        template: str,
        config: Mapping[str, object],
        *,
        argv0: str | None = None,
        with_parser: bool = False,
    ) -> list:
        """Frame an engine invocation as a static shell template plus a config argument.

        The template must be a code constant: every run-specific value rides in the JSON
        ``config``, which becomes the positional parameter ``"$1"`` that the shell passes
        through verbatim, so no user-supplied value is ever interpolated into the command
        string. ``with_parser`` pipes the engine's stdout through
        ``<parser_name> --config "$1"`` for engines with a parser stage, propagating an
        engine failure's exit status even when the parser exits cleanly. ``argv0`` names
        ``$0`` for the ``sh -c`` script; it defaults to the engine name.
        """
        if with_parser:
            if self.parser_name is None:
                raise ValueError(f"{type(self).__name__} does not define a parser_name.")
            # Plain sh reports only the last pipeline command's status (no pipefail), so a
            # clean parser exit would mask an engine crash mid-run. Capture the engine's
            # status through a temp file and fail with it before consulting the parser's.
            template = (
                "engine_status_file=$(mktemp) || exit 1; "
                f'{{ {template}; echo $? > "$engine_status_file"; }}'
                f' | {self.parser_name} --config "$1"; '
                "parser_status=$?; "
                'read engine_status < "$engine_status_file"; '
                'rm -f "$engine_status_file"; '
                '[ "${engine_status:-1}" -eq 0 ] || exit "${engine_status:-1}"; '
                'exit "$parser_status"'
            )
        return ["sh", "-c", template, argv0 or self.engine, json.dumps(config, allow_nan=False)]

    def output_file(self, run_info: RunInfoT) -> str | None:
        """The host path of the file the run will produce, or None when the output
        is printed to stdout instead.

        Args:
            run_info (RunInfoT): Validated settings for one run.

        Returns:
            str | None: Host output path, or None when output is printed to standard output.
        """
        self._check_run_info(run_info)
        output_name = self._output_name(run_info)
        return None if output_name is None else str(self.output_folder / output_name)

    def expected_files(self, run_info: RunInfoT) -> list[str]:
        """Host paths of every file the run promises to produce.

        Runners with sidecar artifacts (e.g. an optimizer scores CSV) extend this.

        Args:
            run_info (RunInfoT): Validated settings for one run.

        Returns:
            list[str]: Host paths the run promises to create.
        """
        output_file = self.output_file(run_info)
        return [] if output_file is None else [output_file]

    @staticmethod
    def _sidecar_file(output_file: str, suffix: str) -> str:
        """Return a sidecar beside ``output_file`` after replacing its final extension."""
        output = Path(output_file)
        return str(output.with_name(f"{output.stem}_{suffix}"))

    def log_file(self, run_info: RunInfoT) -> str:
        """Return the host path of the log file capturing the run's standard error.

        Args:
            run_info (RunInfoT): Validated settings for one run.

        Returns:
            str: Host path of the run log.
        """
        self._check_run_info(run_info)
        log_file_dir = self.log_folder / self.input_stem
        os.makedirs(log_file_dir, exist_ok=True)
        return f"{log_file_dir}/{self.file_stem(run_info)}.log"


class RunContainer(Generic[RunInfoT]):
    """
    A context manager that starts the Docker container for a runner
    configuration, runs commands in it, and cleans the container up afterwards.

    Example::

        config = RecomRunnerConfig("./graphs/my_state.json")
        run_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")

        with RunContainer(config) as container:
            container.run(run_info)
    """

    def __init__(
        self,
        configuration: RunnerConfig[RunInfoT],
        docker_image_name: str = DEFAULT_DOCKER_IMAGE,
        docker_client_args: dict | None = None,
    ):
        """
        Stores the Docker settings for use when the context is entered.

        Args:
            configuration (RunnerConfig): The runner configuration to use. One of
                RecomRunnerConfig, ForestRunnerConfig, or SMCRunnerConfig.
            docker_image_name (str, optional): Override for the Docker image to run.
                Defaults to the immutable digest currently published as
                ``mgggdev/replicate:v2.0.0``.
            docker_client_args (dict, optional): Extra keyword arguments for
                docker.DockerClient, for non-default Docker setups.

        Raises:
            TypeError: When configuration is not a RunnerConfig.
        """
        if not isinstance(configuration, RunnerConfig):
            raise TypeError(
                "Expected a RecomRunnerConfig, ForestRunnerConfig, or SMCRunnerConfig "
                f"for 'configuration', but found {type(configuration).__name__}."
            )
        self.config = configuration
        self.client: docker.DockerClient | None = None
        self._docker_client_args = docker_client_args
        self.container = None
        self.image_name = docker_image_name

    def __enter__(self):
        """
        Pulls the image (falling back to a local copy) and starts the container.

        The container gets a Docker-generated name (no fixed name, so concurrent
        runs never collide) and is removed explicitly in ``__exit__`` after output
        ownership is repaired.

        Raises:
            RuntimeError: When Docker cannot be reached or the container cannot be started.
        """
        if self.client is None:
            try:
                if self._docker_client_args is not None:
                    self.client = docker.DockerClient(**self._docker_client_args)
                else:
                    self.client = docker.from_env()
            except docker.errors.DockerException as e:
                raise RuntimeError(
                    "Could not connect to Docker. Make sure that Docker is installed "
                    "and that the Docker daemon (e.g. Docker Desktop) is running."
                ) from e
        client = self.client

        try:
            config_args = self.config.configure_volumes() | {
                "image": self.image_name,
                "detach": True,
                "tty": True,
                "stdin_open": True,
                "network_mode": "none",
            }
        except BaseException:
            self._close_client()
            raise

        try:
            logger.info(f"Pulling Docker image {self.image_name}")
            client.images.pull(self.image_name)
        except Exception:
            logger.warning(
                f"Could not pull Docker image {self.image_name} from the web. "
                f"Attempting to run using a local copy of the image."
            )
        except BaseException:
            # A KeyboardInterrupt mid-pull must not leak the freshly opened client.
            self._close_client()
            raise

        # Explicit create + start instead of containers.run(): run() does no cleanup when
        # start fails, so each retry would leak another container in the Created state.
        # BaseException here: an interrupt between create() and start() would otherwise
        # leak the created container, since __exit__ never runs when __enter__ raises.
        self.container = None
        try:
            self.container = client.containers.create(**config_args)
            self.container.start()
        except BaseException as e:
            created, self.container = self.container, None
            if created is not None:
                try:
                    created.remove(force=True)
                except Exception as remove_error:
                    logger.warning(f"Could not remove the unstarted container: {remove_error}")
            self._close_client()
            if not isinstance(e, Exception):
                raise
            raise RuntimeError(
                f"Could not start the Docker container for image {self.image_name}. "
                "Make sure that Docker is running and that the image exists locally "
                "or can be pulled."
            ) from e
        logger.info(f"Running Docker container {self.container.name}")

        return self

    def _running_container(self):
        """Returns the started container, or raises if used outside a `with` block."""
        if self.container is None:
            raise RuntimeError(
                "The container has not been started. Use RunContainer inside a "
                "`with` block, e.g. `with RunContainer(config) as container:`."
            )
        return self.container

    def _running_client(self):
        """Return the Docker client, or raise if the context has not been entered."""
        if self.client is None:
            raise RuntimeError("The Docker client is unavailable outside an active context.")
        return self.client

    def _close_client(self) -> None:
        """Close and release the Docker client if one is owned."""
        client = self.client
        if client is None:
            return
        try:
            # The SDK can leave sockets for GC, which surfaces as an "unclosed" warning.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="unclosed", category=ResourceWarning)
                client.close()
        except Exception as error:
            logger.warning(f"Error closing Docker client: {error}")
        finally:
            self.client = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback_obj: TracebackType | None,
    ):
        """
        Removes the Docker container when the `with` block is finished so that
        containers do not accumulate on the user's machine. Returns False so any
        exception from the block propagates normally.
        """
        try:
            if self.container:
                try:
                    self._chown_outputs()
                finally:
                    try:
                        self.container.remove(force=True)
                    except Exception as e:
                        # A dead daemon here must not mask the with-body's exception.
                        logger.warning(f"Error removing Docker container: {e}")
        finally:
            # Reset so a reused RunContainer never holds a stale container reference.
            self.container = None
            self._close_client()

        return False

    def _chown_outputs(self):
        """
        The container runs as root, so files it writes into the mounted output
        folder are root-owned on Linux hosts. Hand them back to the invoking
        user before the container goes away.
        """
        container = self.container
        if container is None or not hasattr(os, "getuid"):
            return
        try:
            exit_code, output = container.exec_run(
                ["chown", "-R", f"{os.getuid()}:{os.getgid()}", self.config.container_output_dir]
            )
            if exit_code != 0:
                logger.warning(
                    "Could not restore ownership of the output folder "
                    f"(exit code {exit_code}): {output!r}"
                )
        except Exception as e:
            logger.warning(f"Could not restore ownership of the output folder: {e}")

    def _exec_stream(self, cmd):
        """
        Starts a command in the container, returning the stream of (stdout, stderr)
        byte chunks and the exec id used to check the command's exit status once
        the stream is drained.
        """
        container = self._running_container()
        client = self._running_client()
        exec_id = client.api.exec_create(
            container.id,
            cmd=cmd,
            tty=False,
            stdout=True,
            stderr=True,
            stdin=False,
        )["Id"]
        stream = client.api.exec_start(
            exec_id,
            stream=True,
            detach=False,
            demux=True,
        )
        return stream, exec_id

    def _check_exit_status(self, exec_id) -> None:
        """Raises RuntimeError when the drained exec's command exited with nonzero status."""
        exit_code = self._running_client().api.exec_inspect(exec_id)["ExitCode"]
        if exit_code is None or exit_code != 0:
            raise RuntimeError(
                f"The {self.config.engine} engine command failed with exit code {exit_code}."
            )

    def _abort_streamed_exec(self) -> None:
        """Force-remove the dedicated container after a streamed command is abandoned."""
        container = self.container
        if container is None:
            return
        self._chown_outputs()
        try:
            container.remove(force=True)
        except Exception as error:
            raise RuntimeError("Could not stop the abandoned streamed engine command.") from error
        self.container = None

    def _cleanup_exec_stream(self, stream: object, *, completed: bool) -> None:
        """Close a Docker exec stream and stop its container if consumption failed."""
        primary_error = sys.exception()
        cleanup_errors: list[Exception] = []
        close = getattr(stream, "close", None)
        try:
            if close is not None:
                close()
        except Exception as error:
            cleanup_errors.append(error)
        if not completed:
            try:
                self._abort_streamed_exec()
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is not None and not isinstance(primary_error, GeneratorExit):
                for error in cleanup_errors:
                    primary_error.add_note(
                        f"Stream cleanup also failed with {type(error).__name__}: {error}"
                    )
            elif len(cleanup_errors) == 1:
                raise cleanup_errors[0] from primary_error
            else:
                raise ExceptionGroup("The streamed engine cleanup failed", cleanup_errors) from (
                    primary_error
                )

    @staticmethod
    def _parse_json_line(raw_line: bytes):
        """Decode and parse one stdout line as JSON, wrapping any failure uniformly."""
        try:
            return json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Could not parse container output as JSON: {raw_line!r}") from e

    def _iter_json_lines(self, cmd):
        """
        Runs a command in the container, yielding `(json_object, None)` tuples, one per
        complete JSON line on stdout, and `(None, stderr_text)` for stderr chunks.

        Raises:
            RuntimeError: When the command exits with nonzero status (raised on exhaustion,
                after the stream is drained), or a stdout line (including a non-blank final
                line without a trailing newline) is not valid JSON.
        """
        # stdout chunks can split a JSON line, or even a multi-byte character, so buffer bytes
        # and decode only complete lines.
        stdout_buffer = b""
        stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="backslashreplace")
        stream, exec_id = self._exec_stream(cmd)
        completed = False
        try:
            # Demuxed docker-py streams yield one-sided (stdout, None) / (None, stderr) frames.
            for stdout, stderr in stream:
                if stdout is not None:
                    stdout_buffer += stdout
                    complete_lines = stdout_buffer.split(b"\n")
                    stdout_buffer = complete_lines.pop()  # keep the incomplete tail
                    for raw_line in complete_lines:
                        if raw_line.strip():
                            yield (self._parse_json_line(raw_line), None)
                if stderr is not None:
                    yield (None, stderr_decoder.decode(stderr))

            # A nonzero exit makes any residual stream tail failure output, not data.
            completed = True
            self._check_exit_status(exec_id)
            final_stderr = stderr_decoder.decode(b"", final=True)
            if final_stderr:
                yield (None, final_stderr)

            # The final JSON line may arrive without a trailing newline; parse it too.
            if stdout_buffer.strip():
                yield (self._parse_json_line(stdout_buffer), None)
        finally:
            self._cleanup_exec_stream(stream, completed=completed)

    def run(self, run_info: RunInfoT) -> str | None:
        """
        Runs the configured engine once with the given run info. Anything printed
        to stderr in the container is written to the log file, and any output not
        sent to an output file is printed to the console.

        Args:
            run_info (RunInfoT): The run-info object for the configured runner.

        Returns:
            str | None: The host path of the output file the run produced, or
            None when the output was printed to stdout instead.

        Raises:
            RuntimeError: When the engine command exits with nonzero status, or exits
                cleanly without producing every expected nonempty output file.
        """
        cmd = self.config.run_command(run_info)
        log_file = self.config.log_file(run_info)
        expected_files = self.config.expected_files(run_info)

        with _preserve_outputs_on_failure(expected_files), _preserve_log_on_failure(log_file):
            # Docker frames can split multi-byte characters, so decode each stream incrementally.
            stdout_decoder = codecs.getincrementaldecoder("utf-8")()
            stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="backslashreplace")
            stream, exec_id = self._exec_stream(cmd)
            completed = False
            try:
                with open(log_file, "w", encoding="utf-8", newline="") as f:
                    for stdout, stderr in stream:
                        if stdout is not None:
                            print(stdout_decoder.decode(stdout), end="")
                        if stderr is not None:
                            f.write(stderr_decoder.decode(stderr))
                            f.flush()  # Ensure the output is written immediately
                    self._check_exit_status(exec_id)
                    print(stdout_decoder.decode(b"", final=True), end="")
                    f.write(stderr_decoder.decode(b"", final=True))
                completed = True
            finally:
                self._cleanup_exec_stream(stream, completed=completed)

            invalid = [
                expected
                for expected in expected_files
                if not os.path.isfile(expected) or os.path.getsize(expected) == 0
            ]
            if invalid:
                raise RuntimeError(
                    f"The {self.config.engine} engine exited successfully but did not "
                    f"produce the expected nonempty output file(s): {', '.join(invalid)}."
                )
        output_file = self.config.output_file(run_info)
        if output_file is not None:
            logger.info(f"Output written to {output_file}")
        return output_file

    def run_iter(self, run_info: RunInfoT):
        """
        Runs the configured engine once and yields its stdout as parsed JSON lines.

        Unlike :meth:`run`, this intentionally applies none of the output-file protections
        (stale-output hiding, expected-file verification, log preservation): the results
        stream to the caller, and any files the engine writes are left untouched.

        Exhaust the iterator normally. Closing it early force-removes the dedicated container so
        the abandoned engine cannot continue running; the current ``RunContainer`` cannot then be
        reused.

        Args:
            run_info (RunInfoT): The run-info object for the configured runner.

        Yields:
            tuple[dict, str]: JSON object parsed from the container's stdout and
            the error message (if any)

        Raises:
            RuntimeError: When the engine command exits with nonzero status, raised
                once the stream is exhausted.
        """
        cmd = self.config.run_command(run_info)
        yield from self._iter_json_lines(cmd)

    def mcmc_run_with_updaters(self, run_info: RunInfoT):
        """
        Runs the configured MCMC engine with canonical assignment output forced to
        stdout, applies the updater functions from ``run_info.updaters`` to every
        sampled plan, and yields the results.

        Args:
            run_info (RunInfoT): A RecomRunInfo or ForestRunInfo carrying updaters.

        Yields:
            tuple[dict, str]: Dictionary of the sample number and updater values and the
            error message (if any)

        Raises:
            TypeError: If ``run_info`` does not carry an ``updaters`` mapping.
            RuntimeError: When the engine command exits with nonzero status, raised
                once the stream is exhausted.
        """
        if not isinstance(run_info, SupportsUpdaters):
            raise TypeError(
                f"{type(run_info).__name__} does not carry updaters; mcmc_run_with_updaters "
                "requires a run info with an 'updaters' mapping (RecomRunInfo or ForestRunInfo)."
            )
        cmd = self.config.canonical_stdout_command(run_info)
        graph = Graph.from_json(self.config.host_graph_path)

        # Output assignments are positional; a set comparison would miss permuted labels.
        node_labels = list(graph.nodes)
        if node_labels != list(range(len(node_labels))):
            raise RuntimeError(
                "mcmc_run_with_updaters requires the dual graph's node labels to be exactly "
                f"0..{len(node_labels) - 1} in ascending order: the engines emit positional "
                "assignment lists, and reordered or non-integer labels would be matched to "
                "the wrong nodes."
            )

        stream = self._iter_json_lines(cmd)
        try:
            for json_obj, stderr_text in stream:
                if json_obj is None:
                    yield (None, stderr_text)
                else:
                    yield from self._process_output(graph, json_obj, run_info.updaters, stderr_text)
        finally:
            primary_error = sys.exception()
            try:
                stream.close()
            except Exception as error:
                if primary_error is not None and not isinstance(primary_error, GeneratorExit):
                    primary_error.add_note(
                        f"Stream cleanup also failed with {type(error).__name__}: {error}"
                    )
                else:
                    raise

    def _process_output(
        self,
        graph: Graph,
        canon_json_line: object,
        updater_dict: dict[str, Callable],
        error=None,
    ):
        """
        Processes the output of the run and applies the updater functions

        Args:
            graph (Graph): The dual graph the assignments index into.
            canon_json_line (object): JSON value from the output of the run. This is expected
                to be in the standard `{'assignment': List[int], 'sample': int}` format
            updater_dict (Dict): Dictionary of updater functions to apply
            error (str, optional): Error message if there is one. Defaults to None.

        Yields:
            Tuple[Dict, str]: Dictionary of the sample number and updater values and the
            error message (if any)
        """
        if not isinstance(canon_json_line, dict):
            raise RuntimeError(
                f"Container output line must be a JSON object, but found {canon_json_line!r}."
            )
        line = cast(dict[str, object], canon_json_line)
        try:
            assignment = line["assignment"]
            sample = line["sample"]
        except KeyError as e:
            # Metadata or other schema-mismatched lines get the module's uniform error surface.
            raise RuntimeError(
                f"Container output line is missing the {e.args[0]!r} key: {canon_json_line!r}"
            ) from e
        if not isinstance(assignment, list):
            raise RuntimeError(
                f"Container output 'assignment' must be a list, but found {assignment!r}."
            )
        if len(assignment) != len(graph):
            raise RuntimeError(
                f"Container output 'assignment' must contain {len(graph)} entries, "
                f"but found {len(assignment)}."
            )
        if not all(not isinstance(label, bool) and isinstance(label, int) for label in assignment):
            raise RuntimeError(
                f"Container output 'assignment' entries must be integers, but found {assignment!r}."
            )
        if isinstance(sample, bool) or not isinstance(sample, int):
            raise RuntimeError(
                f"Container output 'sample' must be an integer, but found {sample!r}."
            )
        partition = Partition(graph, dict(enumerate(assignment)), updaters=updater_dict)

        # Build a fresh dict per line so yielded results never alias each other.
        updater_values = {name: partition[name] for name in updater_dict}

        yield (
            {
                "sample": sample,
                "updaters": updater_values,
            },
            error,
        )
