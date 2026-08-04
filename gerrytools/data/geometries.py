import os
import stat
import tempfile

import httpx
import us
from frozendict import frozendict

from gerrytools.data._geometry_etags import GEOMETRY_ETAGS
from gerrytools.logging import get_logger

logger = get_logger(__name__)

# The path-style S3 REST endpoint provides TLS for the public data.mggg.org bucket.
DATA_MGGG_BASE_URL = "https://s3.us-east-2.amazonaws.com/data.mggg.org"

_REQUEST_TIMEOUT = httpx.Timeout(120)

# Dual-graph geometry levels mapped to their data.mggg.org filename identifiers. "block group"
# and "blockgroup" are accepted spellings of "bg".
_DUALGRAPH_GEOMETRY_IDS = frozendict(
    {
        "bg": "bg",
        "block group": "bg",
        "blockgroup": "bg",
        "vtd": "vtd",
    }
)

# Census-2020 geometry levels mapped to their data.mggg.org identifiers. Accepts the same
# block-group spellings as the dual-graph ids.
_CENSUS_GEOMETRY_IDS = frozendict(
    {
        "bg": "bg",
        "block group": "bg",
        "blockgroup": "bg",
        "block": "block",
        "congress": "cd116",
        "county": "county",
        "cousub": "cousub",
        "place": "place",
        "senate": "sldu",
        "house": "sldl",
        "tract": "tract",
        "vtd": "vtd",
    }
)


def _download_to_file(
    object_key: str,
    filepath: str | os.PathLike[str],
) -> None:
    """Stream a pinned S3 object to ``filepath`` atomically, raising on HTTP error.

    Streaming keeps memory use flat for large shapefile and dual-graph archives, and
    ``raise_for_status`` ensures an S3 error page is never written to disk in place of the
    requested file. ``If-Match`` makes S3 reject an object whose ETag has changed since this
    package was released. The body streams to a temporary file beside the destination that is
    renamed onto ``filepath`` only after the full body arrives, so a mid-download failure never
    leaves a truncated destination behind.

    Args:
        object_key (str): Object path within the public data.mggg.org bucket.
        filepath (str | os.PathLike): Destination path for the response body.

    Raises:
        ValueError: If this package has no pinned ETag for ``object_key``.
        httpx.HTTPStatusError: If the server returns an error status.
    """

    try:
        etag = GEOMETRY_ETAGS[object_key]
    except KeyError:
        raise ValueError(f"No published checksum is available for {object_key!r}.") from None

    url = f"{DATA_MGGG_BASE_URL}/{object_key}"
    destination = os.fspath(filepath)
    # Same directory as the destination so os.replace stays a same-filesystem atomic rename.
    with tempfile.TemporaryDirectory(
        dir=os.path.dirname(destination) or ".",
        prefix=os.path.basename(destination) + ".",
    ) as staging_dir:
        temp_path = os.path.join(staging_dir, "download.part")
        with open(temp_path, "xb") as output_file:
            with (
                httpx.Client(timeout=_REQUEST_TIMEOUT, follow_redirects=True) as client,
                client.stream("GET", url, headers={"If-Match": f'"{etag}"'}) as response,
            ):
                if response.status_code == 412:
                    raise RuntimeError(
                        f"The published geometry {object_key!r} no longer matches this "
                        "gerrytools release's pinned checksum."
                    )
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    output_file.write(chunk)
        try:
            destination_mode = stat.S_IMODE(os.stat(destination).st_mode)
        except FileNotFoundError:
            pass
        else:
            os.chmod(temp_path, destination_mode)
        os.replace(temp_path, destination)


def dualgraphs20(
    state: us.states.State,
    filepath: str | os.PathLike[str],
    geometry: str = "block group",
) -> None:
    """Download Lab-processed dual graph data for a state and write it to disk.

    Args:
        state (us.states.State): State for which to retrieve data.
        filepath (str | os.PathLike): Destination path for the downloaded dual graph JSON.
        geometry (str): Geometry level. One of ``"bg"`` / ``"block group"`` / ``"blockgroup"`` or
            ``"vtd"``. Defaults to ``"block group"``.

    Raises:
        ValueError: If ``geometry`` is not an available dual-graph level or the requested object
            has no package-pinned checksum.
        httpx.HTTPStatusError: If the download request fails.

    Warning:
        Writes the downloaded file to ``filepath``.
    """

    geometry_key = geometry.lower()
    if geometry_key not in _DUALGRAPH_GEOMETRY_IDS:
        raise ValueError(
            f"Requested geometry {geometry!r} is not available as a dual graph; "
            f"choose one of {sorted(_DUALGRAPH_GEOMETRY_IDS)}."
        )

    geometry_id = _DUALGRAPH_GEOMETRY_IDS[geometry_key]
    object_key = f"dual-graphs/{state.abbr.lower()}-{geometry_id}-connected.json"

    logger.info("Downloading %s %s dual graph to %s.", state.abbr, geometry_id, filepath)
    _download_to_file(object_key, filepath)


def vtds20(state: us.states.State, filepath: str | os.PathLike[str]) -> None:
    """Download Lab-processed 2020 VTD shapefile data and write it to disk.

    Args:
        state (us.states.State): State for which to retrieve data.
        filepath (str | os.PathLike): Destination path for the downloaded zipped shapefile.

    Raises:
        ValueError: If the requested object has no package-pinned checksum.
        httpx.HTTPStatusError: If the download request fails.

    Warning:
        Writes the downloaded file to ``filepath``.
    """

    object_key = f"vtd-shapefiles/{state.abbr.upper()}_vtd20.zip"

    logger.info("Downloading %s VTD shapefile to %s.", state.abbr, filepath)
    _download_to_file(object_key, filepath)


def geometries20(
    state: us.states.State,
    filepath: str | os.PathLike[str],
    geometry: str = "tract",
) -> None:
    """Download Lab-processed 2020 geometric data and write it to disk.

    Args:
        state (us.states.State): State for which to retrieve data.
        filepath (str | os.PathLike): Destination path for the downloaded zipped shapefile.
        geometry (str): Geometry level at which to retrieve data. One of ``"bg"`` / ``"block
            group"`` / ``"blockgroup"``, ``"block"``, ``"congress"``, ``"county"``, ``"cousub"``,
            ``"place"``, ``"senate"``, ``"house"``, ``"tract"``, or ``"vtd"``. Defaults to
            ``"tract"``.

    Raises:
        ValueError: If ``geometry`` is not a recognized level or the requested object has no
            package-pinned checksum.
        httpx.HTTPStatusError: If the download request fails.

    Warning:
        Writes the downloaded file to ``filepath``.
    """

    geometry_key = geometry.lower()
    if geometry_key not in _CENSUS_GEOMETRY_IDS:
        raise ValueError(
            f"Requested geometry {geometry!r} is not allowed; "
            f"choose one of {sorted(_CENSUS_GEOMETRY_IDS)}."
        )

    geometry_id = _CENSUS_GEOMETRY_IDS[geometry_key]
    state_abbr = state.abbr.lower()
    object_key = f"census-2020/{state_abbr}/{state_abbr}_{geometry_id}.zip"

    logger.info("Downloading %s %s geometries to %s.", state.abbr, geometry_id, filepath)
    _download_to_file(object_key, filepath)
