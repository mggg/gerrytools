
from sortedcontainers import SortedDict, SortedList
import zlib

class AssignmentCompressor:
    """
    A class for compressing and decompressing lots of assignments very, very
    quickly. Intended for use with ``jsonlines``-like libraries (where assignments
    are read in line-by-line) or for network requests (where assignments are 
    retrieved one-by-one). When decompressing, yields `SortedDict` objects, where
    keys are in sorted order.

    The compression schema considers the set of unique identifiers, imposes an
    ordering (lexicographic order) on the identifiers, and matches the assignment
    to that ordering. We assign all unassigned units to ``"-1"`` and, once the
    default cache size is hit (or assignments are no longer being read in),
    compress all assignments in the cache. Assignments are read in and out in
    the same order, and the keys for each assignment are in the same order.

    Example:
        To compress assignments, we need a set of unique identifiers such that
        each identifier maps one geometric unit to one district.

            ...

            geoids = blocks["GEOID20"]
            ac = AssignmentCompressor(geoids, location="compressed-assignments.ac")

            with ac as compressor:
                for assignment in assignments:
                    compressor.compress(assignment)

            ...

        To decompress assignments, we again must have a set of unique geometric
        identifiers which match the assignments. We can then iterate over the
        decompressed assignments as they're read out of the file.

            ...

            geoids = blocks["GEOID20"]
            ac = AssignmentCompressor(geoids, location="compressed-assignments.ac")

            for assignment in ac.decompress():
                <do whatever!>

            ...

    Attributes:
        DISTRICT_SEPARATOR: A bytestring which separates district identifiers in
            an assignment.
        ASSIGNMENT_SEPARATOR: A bytestring which separates assignments from
            each other.
        CHUNK_SEPARATOR: A bytestring which separates assignment chunks from each
            other.
        CHUNK_SIZE: Default number of bytes read in from the IO stream at each
            step.
        ENCODING: Default string encoding style.
        identifiers: A sortable, iterable collection of unique items corresponding
            to geographic identifiers.
        compressed: A pandas `Index` containing the identifiers; this is used
            to quickly perform vectorized identifier matchings, rather than using
            traditional iterative methods.
        cache: Collection of assignments to be compressed. Assignments are loaded
            into the cache every time the ``.compress()`` method is called, and
            is cleared whenever the length of the cache exceeds the window width.
        window: Maximum cache length before the cache is compressed, written to
            file, and emptied.
        default: The default assignment which is updated each time an assignment
            is passed to the compressor.
    """

    def __init__(self, identifiers, window=10, location="compressed.ac"):
        """
        Creates `AssignmentCompressor` instance.

        Args:
            identifiers: An iterable collection of string identifiers; any
                assignment's keys must be a subset of `identifiers`.
            window: A positive integer representing the cache window size. Defaults
                to cache.
            location: The path to the compressed resource (read or write). Defaults
                to `compressed.ac`.
        """
        self.identifiers = SortedList(identifiers)
        self.default = frozenset(zip(self.identifiers, ["-1"]*len(self.identifiers)))
        self.cache = []
        self.location = location
        self.window = window

        self.DISTRICT_SEPARATOR = b","
        self.ASSIGNMENT_SEPARATOR = b"*"
        self.CHUNK_SEPARATOR = b"()"
        self.CHUNK_SIZE = 16384
        self.ENCODING = "raw_unicode_escape"

    def match(self, assignment):
        """
        Matches an assignment to an index (the set of geometric identifiers)
        and returns a `SortedDict`.

        Args:
            assignment: Dictionary which matches geometric identifiers to
                districts. All keys and values in this dictionary must be strings.

        """
        # Create a dictionary which maps identifiers to `-1`, and update our
        # dictionary with assignment values.
        indexer = SortedDict(self.default)
        indexer.update(assignment)
        
        return indexer

    def __enter__(self):
        """
        A simple context-management method. Allows the user to use `with`
        statements when compressing stuff so we don't have to worry about the
        user specifying the last item they'll be compressing.
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Teardown. Once we've exited the `with` statement (i.e. the user's all done
        feeding items to the compressor) we can force the remaining items to be
        compressed and written to file.
        """
        self._compress(force=True)


    def compress(self, assignment):
        """
        Compresses the assignment `assignment` using ``zlib``.

        Args:
            assignment: Dictionary which matches geometric identifiers to districts.
                All keys and values in this dictionary must be strings.
        """
        # If the user provides an empty assignment or an incorrectly 
        if not assignment:
            UserWarning("`assignment` is empty; skipping")
            return

        if not set(assignment.keys()).issubset(self.identifiers):
            UserWarning(
                "`assignment`'s keys are not a subset of `identifiers`; skipping. " + \
                "Please ensure that all keys and values in `assignment` are strings."
            )
            return

        # Join the things on the district separator, encode the whole thing, and
        # encode according to the default encoding.
        indexed = self.match(assignment)
        sep = self.DISTRICT_SEPARATOR.decode()
        encoded = bytes(sep.join(indexed.values()).encode(self.ENCODING))

        # Compress.
        self.cache += [encoded]
        self._compress()

    def _compress(self, force=False):
        """
        Private method which actually does the compression.

        Args:
            force: If truthy, then the length of the cache is ignored, the data
                in the cache are compressed, and the compressed data is written
                to file. ``force`` is truthy when teardown logic is entered
                (i.e. after ``__exit()__`` is called).
        """
        # Check to see if the cache is full. If so, compress the data, write to
        # file, and reset the cache.
        if len(self.cache) == self.window or force:
            with open(self.location, "ab") as writer:
                # Write compressed data to file.
                compressed = zlib.compress(
                    bytes(self.ASSIGNMENT_SEPARATOR.join(self.cache))
                )
                writer.write(compressed)

                # If this is the last chunk, don't write the delimiter.
                if not force: writer.write(self.CHUNK_SEPARATOR)

            # Reset the cache.
            self.cache = []

    def decompress(self):
        """
        Decompresses the data at ``location``. A generator which ``yield``s
        assignments.

        Yields:
            Yields decompressed assignment dictionaries.
        """
        # Open the compressed file. Then we read it in chunks, loading until
        # we hit our separator or until the end of the file.
        with open(self.location, "rb") as _compressed_fin:
            for chunk in self._chunk(_compressed_fin):
                if not chunk: break
                for assignment in self._decompress(chunk): yield assignment

    def _chunk(self, stream):
        """
        Private method for reading chunks from file without holding the entire
        file in memory. A generator for decompressed assignments.

        Args:
            stream: A ``BytesIO`` instance from which data is read.
        """
        # Create a buffer.
        _buffer = []

        # Read until we hit the end of the file `yield`ing each chunk as we go.
        while True:
            # Read in a chunk of data.
            chunk = stream.read(self.CHUNK_SIZE)
            _buffer.append(chunk)

            # Check if the chunk has our delimiter in it. If it contains our
            # delimiter, then the buffer *up to the delimiter* contains compressed
            # assignments; this should be `yield`ed and decompressed. We only
            # want to get the part before the delimiter for decompression, but
            # retain the rest.
            if self.CHUNK_SEPARATOR in chunk:
                _buffered_bytes = b"".join(_buffer)
                part, _buffer_first = _buffered_bytes.split(self.CHUNK_SEPARATOR, 1)
                _buffer = [_buffer_first]
                yield part

            # If the chunk's empty, `yield` the remaining buffer and return.
            if not chunk: yield b"".join(_buffer); break

    def _decompress(self, chunk):
        """
        Private method which decompresses assignments.

        Args:
            chunk: Compressed, byte-encoded data representing ``window``
                assignments.

        Returns:
            List of decompressed assignment objects.
        """
        # Decompress the chunk and split it on our delimiter.
        decompressed = zlib.decompress(chunk)
        decompressed_parts = decompressed.split(self.ASSIGNMENT_SEPARATOR)
        
        # For each of the parts, decode the bytes, make them into lists, and
        # match them to GEOIDs.
        decoded_parts = [part.decode() for part in decompressed_parts]
        split_parts = [part.split(self.DISTRICT_SEPARATOR.decode()) for part in decoded_parts]
        indexed_parts = [dict(zip(self.identifiers, part)) for part in split_parts]

        return indexed_parts
