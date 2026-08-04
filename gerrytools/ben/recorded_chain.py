"""GerryChain integration for recording Markov chains as BENDL bundles.

``RecordedChain`` subclasses ``gerrychain.MarkovChain``: iterating it runs the chain as usual,
yielding the live partitions, while streaming each step's assignment vector into a BENDL file that
also embeds the dual graph and optional metadata. The file is finalized and atomically published
only when the run completes cleanly, so a crashed run never replaces the destination.
"""

from __future__ import annotations

import copy
import errno
import json
import os
import stat
import tempfile
import threading
import warnings
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol, Self, cast, get_args

import networkx as nx
from binary_ensemble import BendlDecoder, BendlEncoder
from gerrychain import Graph, MarkovChain, Partition
from gerrychain.graph import FrozenGraph

from ._graph_prep import (
    GraphOrder,
    GraphOrderName,
    _assert_graph_equal,
    _assert_original_edge_attributes_unchanged,
    _execution_graph,
    _normalize_numpy,
    _prepare_graph,
    _source_graph,
)

Variant = Literal["standard", "mkv_chain", "twodelta"]

# Filesystems without hard-link support (exFAT, some network mounts) surface one of these.
_HARD_LINK_UNSUPPORTED_ERRNOS = frozenset(
    {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV}
)


class RunIterator(Iterator[Partition], Protocol):
    """Run iterator that, like a generator, can also be closed early."""

    def close(self) -> None:
        """Release resources without consuming the rest of the run."""
        ...


def _warn(message: str) -> None:
    """Warn without allowing a warnings-as-errors policy to break cleanup."""
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    except RuntimeWarning:
        pass


class _BendlTransaction:
    """One atomic BENDL publication: a temporary bundle streamed next to the destination and
    moved over it only on clean completion.

    A context manager owning every filesystem concern of a run. Construction only creates the
    temporary directory, so a partially constructed transaction never leaves an unreachable
    file behind. Entering builds the encoder, embeds and verifies the graph, and opens the
    assignment stream; the body then calls :meth:`write` per sample and, on the success path,
    :meth:`finalize`, :meth:`publish`, and :meth:`cleanup_published`. Exiting on an exception
    closes a still-open stream and attaches the recovery notes (except for ``GeneratorExit``,
    which is an early close, not a failure, and gets no notes). Chain state (locking, run
    validation, partition verification) stays with ``RecordedChain._record``.
    """

    def __init__(
        self,
        *,
        output_path: Path,
        source_graph: nx.Graph,
        expected_graph: nx.Graph,
        graph_order: GraphOrder,
        graph_order_key: str | None,
        metadata: dict[str, Any] | list[Any] | None,
        variant: Variant,
    ) -> None:
        self._output_path = output_path
        self._source_graph = source_graph
        self._expected_graph = expected_graph
        self._graph_order: GraphOrder = graph_order
        self._graph_order_key = graph_order_key
        self._metadata = metadata
        self._variant: Variant = variant
        self.temporary_directory = Path(
            tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
        )
        self.temporary_path = self.temporary_directory / "recording.bendl"
        self.finalized = False
        self.committed = False
        self._stream: Any = None
        self._stream_open = False
        self._linked = False

    def __enter__(self) -> _BendlTransaction:
        """Build the encoder, embed and verify the graph, add metadata, and open the stream.

        An exception here bypasses ``__exit__``, so this method attaches its own recovery
        notes before re-raising.
        """
        try:
            encoder = BendlEncoder(self.temporary_path, overwrite=False)
            try:
                stored_graph = encoder.add_graph(
                    self._source_graph, sort=self._graph_order, key=self._graph_order_key
                )
            except BaseException as exc:
                exc.add_note("RecordedChain failed during run-time graph serialization.")
                raise
            _assert_graph_equal(stored_graph, self._expected_graph, "encoder graph")
            if self._metadata is not None:
                encoder.add_metadata(copy.deepcopy(self._metadata))
            stream = encoder.ben_stream(variant=self._variant)
            stream.__enter__()
            self._stream = stream
            self._stream_open = True
        except BaseException as exc:
            self._note_failure(exc)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Close a still-open stream and attach recovery notes on failure.

        ``GeneratorExit`` (the consumer closed the run iterator early) still closes the stream
        but gets no notes; :meth:`RecordedChain._record`'s cleanup warns about the preserved
        recording instead. Exceptions are never suppressed. Closing is idempotent with
        :meth:`finalize`, which the success path is expected to have called already.
        """
        if exc is not None:
            if self._stream_open:
                self._abort_stream(exc)
            if not isinstance(exc, GeneratorExit):
                self._note_failure(exc)
        elif self._stream_open:
            self._stream_open = False
            self._stream.__exit__(None, None, None)
        return False

    def write(self, partition: Partition, sample_index: int) -> None:
        """Write one assignment vector, naming the failing sample in any raised error."""
        try:
            self._stream.write(partition.assignment_vector)
        except BaseException as exc:
            exc.add_note(f"RecordedChain failed while writing sample {sample_index}")
            raise

    def _abort_stream(self, exc: BaseException) -> None:
        """Close the stream during exception unwinding, noting any secondary close failure."""
        self._stream_open = False
        try:
            self._stream.__exit__(type(exc), exc, exc.__traceback__)
        except BaseException as close_error:
            exc.add_note(f"Closing the BENDL stream also failed: {close_error!r}")

    def finalize(self) -> None:
        """Close the stream, then re-read and verify the finalized bundle's embedded graph."""
        self._stream_open = False
        self._stream.__exit__(None, None, None)
        self.finalized = True
        embedded_graph = BendlDecoder(self.temporary_path).read_graph()
        if embedded_graph is None:
            raise RuntimeError("finalized BENDL has no embedded graph")
        _assert_graph_equal(embedded_graph, self._expected_graph, "embedded graph")

    def publish(self, overwrite: bool) -> None:
        """Publish the finalized recording to the destination.

        A fresh publication hard-links the temporary file into place, failing if a destination
        appeared meanwhile; an authorized overwrite copies the destination's permissions onto
        the new file and atomically replaces it.
        """
        if not overwrite:
            try:
                os.link(self.temporary_path, self._output_path)
            except OSError as error:
                if error.errno not in _HARD_LINK_UNSUPPORTED_ERRNOS:
                    raise
                raise OSError(
                    errno.ENOTSUP,
                    "atomic no-clobber publication requires hard-link support",
                    self._output_path,
                ) from error
            else:
                self._linked = True
        else:
            try:
                destination = os.lstat(self._output_path)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(destination.st_mode):
                    raise OSError(
                        errno.EINVAL,
                        "overwrite destination must be a regular file",
                        self._output_path,
                    )
                os.chmod(self.temporary_path, stat.S_IMODE(destination.st_mode))
            os.replace(self.temporary_path, self._output_path)
        self.committed = True

    def cleanup_published(self) -> None:
        """Remove the writable alias and temporary directory, downgrading failures to warnings."""
        if self._linked:
            try:
                self.temporary_path.unlink()
            except OSError:
                _warn(
                    f"RecordedChain published successfully but retained writable alias "
                    f"{self.temporary_path}; chain-owned-path guarantees are degraded until it "
                    "is removed"
                )
        try:
            self.temporary_directory.rmdir()
        except OSError:
            _warn(
                f"RecordedChain published successfully but could not remove temporary directory "
                f"{self.temporary_directory}"
            )

    def has_uncommitted_recording(self) -> bool:
        """True when a partial or unpublished recording file survives on disk."""
        return self.temporary_path.exists() and not self.committed

    def note_preserved(self, exc: BaseException) -> None:
        """Record on ``exc`` where the surviving recording lives and whether it finalized."""
        state = "finalized" if self.finalized else "unfinalized"
        exc.add_note(f"Preserved {state} BENDL recording at {self.temporary_path}")

    def _note_failure(self, exc: BaseException) -> None:
        """Attach recovery notes for a failed run, removing the empty temporary directory."""
        if self.has_uncommitted_recording():
            self.note_preserved(exc)
        elif self.temporary_directory.exists():
            try:
                self.temporary_directory.rmdir()
            except OSError as cleanup_error:
                exc.add_note(
                    f"Temporary directory {self.temporary_directory} could not be removed: "
                    f"{cleanup_error}"
                )


class RecordedRun:
    """Read-back API for one successfully published recording.

    Produced by a clean :class:`RecordedChain` run and exposed as
    :attr:`RecordedChain.recording`. It holds the published file's resolved path plus the
    recorded partition class and updaters, so assignment vectors and full partitions can be
    read back without the chain object.

    Handles follow their file: every read opens (or reuses a decoder for) the file at
    :attr:`path`. Assignment lookup, iteration, sample counts, metadata, integrity verification,
    and partition reconstruction are available directly on the handle; :attr:`decoder` exposes
    lower-level BENDL operations. When a later authorized rerun publishes over that same file, the
    handle is invalidated at the moment of replacement and every read method afterwards raises
    ``RuntimeError`` naming the path; only :attr:`path` stays readable. A rerun whose destination
    resolves to a different file leaves the handle fully usable.
    """

    __slots__ = (
        "_decoder_cache",
        "_invalidated",
        "_lock",
        "_lookup_graph",
        "_partition_type",
        "_path",
        "_updaters",
    )

    def __init__(
        self,
        path: Path,
        partition_type: type[Partition],
        updaters: dict[str, Any],
    ) -> None:
        """Initialize a reader for the published recording at ``path``.

        Args:
            path (Path): Path of the published ``.bendl`` file. Resolved on construction so
                the handle keeps reading the file it was created for even if symlinks on the
                original path are later retargeted.
            partition_type (type[Partition]): Partition class recorded by the run.
            updaters (dict[str, Any]): Updaters recorded partitions are rebuilt with.
        """
        self._path = Path(path).resolve()
        self._partition_type = partition_type
        self._updaters = dict(updaters)
        self._lock = threading.RLock()
        self._invalidated = False
        self._decoder_cache: BendlDecoder | None = None
        self._lookup_graph: FrozenGraph | None = None

    @classmethod
    def from_bendl(
        cls,
        path: str | os.PathLike[str],
        *,
        partition_type: type[Partition] = Partition,
        updaters: dict[str, Any] | None = None,
    ) -> Self:
        """Create a reader for an existing BENDL recording.

        Opening is lazy: the file is read when a lookup, decoder, or partition method is first
        used. Assignment vectors can be read from a graphless BENDL file, but reconstructing
        partitions requires an embedded graph.

        Args:
            path (str | os.PathLike[str]): Path to the recorded ``.bendl`` file.
            partition_type (type[Partition], optional): Partition class to reconstruct. Its root
                constructor must accept ``graph``, ``assignment``, ``updaters``, and
                ``use_default_updaters``. Defaults to ``Partition``.
            updaters (dict[str, Any] | None, optional): Updaters attached to reconstructed
                partitions. Defaults to an empty dictionary.

        Returns:
            RecordedRun: A lazy reader for the recording.
        """
        return cls(Path(path), partition_type, {} if updaters is None else updaters)

    @property
    def path(self) -> Path:
        """Resolved absolute path of the published recording."""
        return self._path

    def _invalidate(self) -> None:
        """Mark the handle stale; called by the chain when a rerun replaces its file."""
        with self._lock:
            self._invalidated = True

    def _revalidate(self) -> None:
        """Undo :meth:`_invalidate` after a failed publish left this handle's file untouched."""
        with self._lock:
            self._invalidated = False

    def _guard(self) -> None:
        """Raise if a later rerun has overwritten this handle's file."""
        with self._lock:
            if self._invalidated:
                raise RuntimeError(
                    f"RecordedRun for {self._path} is no longer readable: "
                    "a later rerun overwrote that file"
                )

    @property
    def decoder(self) -> BendlDecoder:
        """The shared decoder for the published file, created on first use."""
        with self._lock:
            self._guard()
            if self._decoder_cache is None:
                self._decoder_cache = self._open_decoder()
            return self._decoder_cache

    def count_samples(self) -> int:
        """Return the number of recorded assignment samples.

        Returns:
            int: Number of samples in the assignment stream.
        """
        with self._lock:
            self._guard()
            return self.decoder.count_samples()

    def __len__(self) -> int:
        """Return the number of recorded assignment samples."""
        return self.count_samples()

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over all recorded assignment vectors using an independent decoder."""
        return self._guarded_assignments(self._new_decoder())

    @property
    def metadata(self) -> Any:
        """Deep copy of the recording's parsed JSON metadata, or None when absent."""
        with self._lock:
            self._guard()
            return copy.deepcopy(self.decoder.read_metadata())

    def verify(self) -> None:
        """Verify the recording's asset, stream, and sample-count integrity."""
        with self._lock:
            self._guard()
            self.decoder.verify()

    def assignment_at(self, index: int) -> list[int]:
        """Return the zero-based recorded assignment vector at ``index``.

        Args:
            index (int): Sample index; sample 0 is the initial partition.

        Returns:
            list[int]: Assignment vector in recorded graph node order.
        """
        with self._lock:
            self._guard()
            return self.decoder.lookup(index)

    def lookup(self, index: int) -> list[int]:
        """Return the assignment at ``index``; alias for :meth:`assignment_at`.

        Args:
            index (int): Sample index; sample 0 is the initial partition.

        Returns:
            list[int]: Assignment vector in recorded graph node order.
        """
        return self.assignment_at(index)

    def partition_at(self, index: int) -> Partition:
        """Reconstruct the recorded partition at ``index``.

        The partition is rebuilt on the embedded graph, with the same class and updaters as the
        recorded run.

        Args:
            index (int): Sample index; sample 0 is the initial partition.

        Returns:
            Partition: The reconstructed partition.

        Raises:
            RuntimeError: If the file has no embedded graph.
            TypeError: If the recorded partition class cannot be reconstructed.
        """
        with self._lock:
            self._guard()
            return self._reconstruct_partition(self.decoder.lookup(index))

    def partitions(self, assignments: Iterable[list[int]]) -> Iterator[Partition]:
        """Lazily reconstruct partitions from assignment vectors.

        Composes with any vector source: this handle's iterator, the ``subsample_*`` iterators,
        :meth:`assignment_at` results, or vectors from another decoder for the same graph.

        Args:
            assignments (Iterable[list[int]]): Assignment vectors in recorded graph node order.

        Returns:
            Iterator[Partition]: Reconstructed partitions, one per input vector.

        Raises:
            RuntimeError: If the file has no embedded graph (raised lazily, on iteration).
            TypeError: If the recorded partition class cannot be reconstructed.
        """
        return (
            self._reconstruct_partition(assignment)
            for assignment in self._guarded_assignments(assignments)
        )

    def subsample_every(self, step: int, offset: int = 0) -> Iterator[list[int]]:
        """Iterate over every ``step``-th recorded assignment vector, starting at ``offset``.

        Args:
            step (int): Stride between samples.
            offset (int, optional): Zero-based index of the first sample. Defaults to 0.

        Returns:
            Iterator[list[int]]: Lazily decoded assignment vectors.
        """
        return self._guarded_assignments(self._new_decoder().subsample_every(step, offset))

    def subsample_indices(self, indices: Sequence[int]) -> Iterator[list[int]]:
        """Iterate over the recorded assignment vectors at the given zero-based indices.

        Args:
            indices (Sequence[int]): Sorted, unique sample indices to decode.

        Returns:
            Iterator[list[int]]: Lazily decoded assignment vectors.

        Raises:
            ValueError: If ``indices`` is not sorted and unique.
        """
        requested = list(indices)
        if requested != sorted(set(requested)):
            raise ValueError("indices must be sorted and unique")
        return self._guarded_assignments(self._new_decoder().subsample_indices(requested))

    def subsample_range(self, start: int, end: int) -> Iterator[list[int]]:
        """Iterate over the recorded assignment vectors in the half-open range ``[start, end)``.

        Args:
            start (int): First zero-based sample index.
            end (int): One past the last sample index.

        Returns:
            Iterator[list[int]]: Lazily decoded assignment vectors.
        """
        return self._guarded_assignments(self._new_decoder().subsample_range(start, end))

    def _new_decoder(self) -> BendlDecoder:
        """Return a fresh decoder so concurrent subsample iterators do not share cursor state."""
        self._guard()
        return self._open_decoder()

    def _open_decoder(self) -> BendlDecoder:
        """Open the recording with a stable public error surface."""
        try:
            return BendlDecoder(self._path)
        except Exception as exc:
            raise RuntimeError(f"Could not open recorded BENDL {self._path}: {exc}") from exc

    def _guarded_assignments(self, assignments: Iterable[list[int]]) -> Iterator[list[int]]:
        """Yield assignments only while this handle remains readable."""
        iterator = iter(assignments)
        while True:
            with self._lock:
                self._guard()
                try:
                    assignment = next(iterator)
                except StopIteration:
                    return
            yield assignment

    def _reconstruct_partition(self, assignment_vector: list[int]) -> Partition:
        """Build a partition of the recorded class from an assignment vector.

        The first reconstruction reads the embedded graph from the file and keeps the resulting
        frozen graph; later reconstructions reuse it, assigning by internal node index directly.

        Raises:
            RuntimeError: If the file has no embedded graph.
            ValueError: If the assignment vector's length does not match the graph's node count.
            TypeError: If the recorded partition class cannot be constructed from graph,
                assignment, updaters, and use_default_updaters.
        """
        with self._lock:
            self._guard()
            if self._lookup_graph is None:
                decoded_graph = self.decoder.read_graph()
                if decoded_graph is None:
                    raise RuntimeError("recorded BENDL has no embedded graph")
                node_count = decoded_graph.number_of_nodes()
            else:
                decoded_graph = None
                node_count = len(self._lookup_graph.nodes)
            # Checked explicitly so both graph-cache states surface the same clear error.
            if len(assignment_vector) != node_count:
                raise ValueError(
                    f"assignment vector has {len(assignment_vector)} entries but the recorded "
                    f"graph has {node_count} nodes"
                )
            try:
                if decoded_graph is not None:
                    assignment = dict(zip(decoded_graph.nodes, assignment_vector, strict=True))
                    partition = self._partition_type(
                        decoded_graph,
                        assignment,
                        updaters=self._updaters,
                        use_default_updaters=False,
                    )
                    self._lookup_graph = partition.graph
                else:
                    partition = self._partition_type(
                        self._lookup_graph,
                        dict(enumerate(assignment_vector)),
                        updaters=self._updaters,
                        use_default_updaters=False,
                    )
            except TypeError as exc:
                raise TypeError(
                    f"Cannot reconstruct recorded partition class "
                    f"{self._partition_type.__name__}; its root constructor must accept "
                    "graph, assignment, updaters, and use_default_updaters"
                ) from exc
        return partition


class RecordedChain(MarkovChain):
    """A GerryChain Markov chain that atomically publishes a recorded BENDL run.

    Iterating the chain runs it as usual, yielding the live partitions, while each step's
    assignment vector streams to a temporary BENDL file that embeds the reordered dual graph and
    optional metadata. The file is verified and atomically published to ``output_path`` only when
    the run finishes cleanly; a failed or interrupted run never replaces the destination.

    Construction canonicalizes and reorders the graph, so configure graph-dependent chain
    components (initial partition, proposal, constraints) only after construction, from
    ``chain.graph``; partitions built on any other graph fail the run's verification. The
    initial and final partitions are verified structurally; every step in between gets an
    identity-based check (same graph object and partition class as the initial partition),
    which partitions produced by ``flip`` satisfy automatically. After a
    successful run, :attr:`recording` exposes a :class:`RecordedRun` whose iterator,
    :meth:`~RecordedRun.assignment_at`, and ``subsample_*`` methods read the recorded assignment
    vectors back, and whose :meth:`~RecordedRun.partition_at` and
    :meth:`~RecordedRun.partitions` reconstruct full partitions.
    """

    __slots__ = (
        "_active",
        "_graph",
        "_graph_order",
        "_graph_order_key",
        "_expected_graph",
        "_metadata",
        "_output_path",
        "_recording",
        "_source_graph",
        "_state_lock",
        "_variant",
    )
    _graph_order: GraphOrder
    _variant: Variant

    def __init__(
        self,
        graph: nx.Graph | Graph,
        *,
        output_path: str | os.PathLike[str],
        total_steps: int | None = None,
        rng: Any = None,
        graph_order: GraphOrder = "mlc",
        graph_order_key: str | None = None,
        metadata: dict[str, Any] | list[Any] | None = None,
        variant: Variant = "twodelta",
    ) -> None:
        """Initialize a RecordedChain.

        Args:
            graph (nx.Graph | Graph): Dual graph to record: a NetworkX graph or a NetworkX-backed
                GerryChain ``Graph``. Deep-copied, so later caller mutations do not reach the
                chain; build run components from ``chain.graph``, not from this argument.
            output_path (str | os.PathLike[str]): Destination for the published ``.bendl`` file.
                The parent directory must exist, and the file itself must not (see
                ``allow_overwrite``).
            total_steps (int | None, optional): Number of steps to run, forwarded to
                ``MarkovChain``. Defaults to None.
            rng (Any, optional): Seed or random generator, forwarded to ``MarkovChain``. Defaults
                to None.
            graph_order (GraphOrder): Node reordering applied before encoding: ``"mlc"`` (the
                default), ``"rcm"``, ``"key"`` (sort by the ``graph_order_key`` node attribute),
                or ``None`` to keep the input order. Reordering improves compression; the
                permutation back to the source order is stored in the file.
            graph_order_key (str | None): Node attribute to sort by; required exactly when
                ``graph_order="key"``.
            metadata (dict[str, Any] | list[Any] | None, optional): JSON-serializable metadata
                embedded in the bundle. Defaults to None.
            variant (Variant): Assignment-stream encoding: ``"twodelta"`` (the default),
                ``"mkv_chain"``, or ``"standard"``.

        Raises:
            ValueError: If ``graph_order`` or ``variant`` is invalid, or ``graph_order`` is
                inconsistent with ``graph_order_key``.
            TypeError: If ``metadata`` is not a JSON-serializable dict, list, or None, or
                ``graph`` is unsupported.
        """
        if graph_order is not None and graph_order not in get_args(GraphOrderName):
            raise ValueError("graph_order must be 'mlc', 'rcm', 'key', or None")
        if (graph_order == "key") != (graph_order_key is not None):
            raise ValueError("graph_order_key is required only when graph_order='key'")
        if variant not in get_args(Variant):
            raise ValueError("variant must be 'standard', 'mkv_chain', or 'twodelta'")
        if metadata is not None and not isinstance(metadata, (dict, list)):
            raise TypeError("metadata must be a dict, list, or None")

        source = _source_graph(graph)
        prepared = _prepare_graph(source, graph_order, graph_order_key)
        self._source_graph = source
        self._expected_graph = copy.deepcopy(prepared)
        self._graph = nx.freeze(prepared)
        self._graph_order = graph_order
        self._graph_order_key = graph_order_key
        self._output_path = Path(output_path).expanduser().absolute()
        # Mirror the graph path's normalization: absorb NumPy scalars and apply JSON's value
        # coercions now, so genuinely unserializable metadata fails here instead of mid-run.
        if metadata is None:
            self._metadata = None
        else:
            try:
                self._metadata = json.loads(json.dumps(_normalize_numpy(metadata)))
            except (TypeError, ValueError) as error:
                raise TypeError(f"metadata is not JSON-serializable: {error}") from error
        self._variant = variant
        self._state_lock = threading.RLock()
        self._active = False
        self._recording: RecordedRun | None = None
        super().__init__(initial_partition=None, total_steps=total_steps, rng=rng)

    @property
    def graph(self) -> nx.Graph:
        """The frozen, canonicalized graph every partition in a run must be built on."""
        return self._graph

    @property
    def graph_order(self) -> GraphOrder:
        """Node reordering mode applied before encoding."""
        return self._graph_order

    @property
    def graph_order_key(self) -> str | None:
        """Node attribute sorted by when ``graph_order="key"``, else None."""
        return self._graph_order_key

    @property
    def output_path(self) -> Path:
        """Absolute destination path of the published recording."""
        return self._output_path

    @property
    def metadata(self) -> dict[str, Any] | list[Any] | None:
        """Deep copy of the metadata embedded in the bundle, or None."""
        return copy.deepcopy(self._metadata)

    @property
    def variant(self) -> Variant:
        """Assignment-stream encoding variant written by runs."""
        return self._variant

    @property
    def recording(self) -> RecordedRun:
        """The reader for the latest successfully published recording.

        A new successful run replaces this object. A handle kept from before an authorized
        rerun stays usable while the rerun is active or if it fails at any point, including
        a failed publish (the destination is only replaced by a successful publish); once a
        rerun publishes over the same file, the old handle is invalidated and its read
        methods raise.

        Raises:
            RuntimeError: If a run is active or no successful recording exists yet.
        """
        with self._state_lock:
            if self._active:
                raise RuntimeError("recording is unavailable while RecordedChain is running")
            if self._recording is None:
                raise RuntimeError("recording is unavailable before a successful recording")
            return self._recording

    def __iter__(self) -> RunIterator:
        """Return the single-use run iterator; the destination file must not already exist."""
        return cast(RunIterator, self._record(overwrite=False))

    def allow_overwrite(self) -> RunIterator:
        """Return one run iterator authorized to replace the destination file.

        The replacement is atomic: the new recording is finalized in a temporary location and
        moved over ``output_path`` only on clean completion.

        Returns:
            RunIterator: Single-use iterator that records the run and may replace the output file.
        """
        return cast(RunIterator, self._record(overwrite=True))

    def _publish_run(
        self,
        transaction: _BendlTransaction,
        *,
        overwrite: bool,
        partition_type: type[Partition],
        updaters: dict[str, Any],
    ) -> None:
        """Publish a finalized run and replace the read handle without leaving it stale."""
        with self._state_lock:
            previous_recording = self._recording
        if (
            previous_recording is not None
            and previous_recording.path != self._output_path.resolve()
        ):
            previous_recording = None
        if previous_recording is not None:
            previous_recording._invalidate()
        try:
            transaction.publish(overwrite)
        except BaseException:
            # A failed atomic publish leaves the old file intact, so its handle remains valid.
            if previous_recording is not None:
                previous_recording._revalidate()
            raise
        with self._state_lock:
            self._recording = RecordedRun(self._output_path, partition_type, updaters)
        transaction.cleanup_published()

    def _finish_run(
        self,
        parent_iterator: Iterator[Partition] | None,
        transaction: _BendlTransaction | None,
        pending: BaseException | None,
    ) -> None:
        """Close the parent iterator, release the chain, and report preserved recordings."""
        early_close = (
            isinstance(pending, GeneratorExit)
            and transaction is not None
            and transaction.has_uncommitted_recording()
        )
        parent_close_error: BaseException | None = None
        if parent_iterator is not None:
            try:
                close = getattr(parent_iterator, "close", None)
                if close is not None:
                    close()
            except BaseException as close_error:
                if pending is not None:
                    pending.add_note(
                        f"Closing the parent chain iterator also failed: {close_error!r}"
                    )
                else:
                    parent_close_error = close_error
        with self._state_lock:
            self._active = False
        if early_close and transaction is not None:
            _warn(
                f"Incomplete RecordedChain run preserved at {transaction.temporary_path}; "
                "recover its stream with extract_stream(..., allow_unfinalized=True)"
            )
        if parent_close_error is None:
            return
        if transaction is not None and transaction.committed:
            _warn(
                "RecordedChain published successfully but closing the parent iterator "
                f"failed: {parent_close_error!r}"
            )
            return
        if transaction is not None and transaction.temporary_path.exists():
            transaction.note_preserved(parent_close_error)
        raise parent_close_error

    def _record(self, *, overwrite: bool) -> Iterator[Partition]:
        """Run the chain once, streaming samples through an atomic BENDL transaction.

        This generator owns chain concerns only: the active-run flag, run validation, partition
        verification, and streaming/yielding partitions. The transaction owns the temporary
        bundle, finalization checks, atomic publication, and failure annotation. On any failure
        the destination is untouched and a partial recording is preserved next to it for recovery.
        """
        parent_iterator: Iterator[Partition] | None = None
        transaction: _BendlTransaction | None = None
        pending: BaseException | None = None

        with self._state_lock:
            if self._active:
                raise RuntimeError("another RecordedChain run is already active")
            self._active = True

        try:
            self._validate_run(overwrite)
            parent_iterator = super().__iter__()
            first_partition = next(parent_iterator)
            self._verify_partition(first_partition, "initial partition")

            transaction = _BendlTransaction(
                output_path=self._output_path,
                source_graph=self._source_graph,
                expected_graph=self._expected_graph,
                graph_order=self._graph_order,
                graph_order_key=self._graph_order_key,
                metadata=self._metadata,
                variant=self._variant,
            )
            with transaction:
                final_partition = first_partition
                transaction.write(first_partition, 0)
                yield first_partition
                for sample_index, partition in enumerate(parent_iterator, start=1):
                    self._verify_step(partition, first_partition, sample_index)
                    final_partition = partition
                    transaction.write(partition, sample_index)
                    yield partition

                transaction.finalize()
                self._verify_partition(final_partition, "final partition")
                self._publish_run(
                    transaction,
                    overwrite=overwrite,
                    partition_type=type(first_partition),
                    updaters=dict(first_partition.updaters),
                )
        except BaseException as exc:
            pending = exc
            raise
        finally:
            self._finish_run(parent_iterator, transaction, pending)

    def _validate_run(self, overwrite: bool) -> None:
        """Check the destination directory and overwrite authorization.

        Chain configuration is checked by ``check_valid`` inside ``super().__iter__()``, which
        runs immediately after this and before any side effect.
        """
        parent = self._output_path.parent
        if not parent.exists():
            raise FileNotFoundError(parent)
        if not parent.is_dir():
            raise NotADirectoryError(parent)
        if not overwrite:
            if self._recording is not None:
                raise RuntimeError(
                    "RecordedChain has already run; use allow_overwrite() for one authorized rerun"
                )
            if self._output_path.exists() or self._output_path.is_symlink():
                raise FileExistsError(
                    errno.EEXIST,
                    "destination exists; use allow_overwrite() to replace it",
                    self._output_path,
                )

    def _verify_step(
        self, partition: Partition, first_partition: Partition, sample_index: int
    ) -> None:
        """O(1) identity check that a mid-run partition still runs on the initial graph.

        A transient foreign-graph partition (same nodes, different order) would record a
        permuted assignment vector, so every step's partition must share the verified initial
        partition's graph object and exact class; partitions produced by ``flip`` do. The full
        structural verification still runs on the first and final partitions.
        """
        if partition.graph is not first_partition.graph:
            raise RuntimeError(
                f"sample {sample_index} partition does not share the initial partition's graph"
            )
        if type(partition) is not type(first_partition):
            raise RuntimeError(
                f"sample {sample_index} partition class {type(partition).__name__} does not "
                f"match the initial partition class {type(first_partition).__name__}"
            )

    def _verify_partition(self, partition: Partition, context: str) -> None:
        """Verify ``partition`` was built from ``self.graph`` with matching node order.

        Assignment vectors are positional, so a partition whose internal node ids do not
        enumerate ``self.graph``'s nodes in order would silently record permuted assignments.
        """
        expected_nodes = list(self._graph.nodes)
        internal_nodes = partition.graph.graph.node_indices
        if internal_nodes != set(range(len(expected_nodes))):
            raise RuntimeError(f"{context} does not use contiguous internal node ids")
        actual_nodes = [
            partition.graph.original_nx_node_id_for_internal_node_id(index)
            for index in range(len(expected_nodes))
        ]
        if actual_nodes != expected_nodes:
            raise RuntimeError(f"{context} node order does not match RecordedChain.graph")
        if any(
            partition.graph.graph.node_data(index) is not self._graph.nodes[node]
            for index, node in enumerate(expected_nodes)
        ):
            raise RuntimeError(f"{context} was not built from RecordedChain.graph")
        execution_graph = _execution_graph(partition)
        execution_context = f"{context} execution graph"
        _assert_graph_equal(
            execution_graph,
            self._expected_graph,
            execution_context,
            graph_attributes=False,
            edge_attributes=False,
        )
        _assert_original_edge_attributes_unchanged(
            execution_graph, self._expected_graph, execution_context
        )
