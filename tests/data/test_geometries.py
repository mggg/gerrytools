import logging
import stat
from pathlib import Path

import httpx
import pytest
import us

from gerrytools.data.geometries import (
    DATA_MGGG_BASE_URL,
    dualgraphs20,
    geometries20,
    vtds20,
)
from tests.data._helpers import MockHTTP

# ==================================
# == DUAL GRAPHS (dualgraphs20)   ==
# ==================================


class TestDualGraphs:
    @pytest.mark.parametrize(
        "geometry,expected_id",
        [
            ("bg", "bg"),
            ("block group", "bg"),
            ("blockgroup", "bg"),
            ("vtd", "vtd"),
            # Geometry is matched case-insensitively.
            ("BG", "bg"),
            ("Block Group", "bg"),
            ("VTD", "vtd"),
        ],
    )
    def test_builds_expected_url(self, mock_http: MockHTTP, tmp_path: Path, geometry, expected_id):
        mock_http.route(content=b"{}")
        out = tmp_path / "dg.json"

        dualgraphs20(us.states.WI, out, geometry=geometry)

        expected = f"{DATA_MGGG_BASE_URL}/dual-graphs/wi-{expected_id}-connected.json"
        assert mock_http.urls == [expected]

    def test_block_group_does_not_leak_a_space_into_the_url(
        self, mock_http: MockHTTP, tmp_path: Path
    ):
        # Regression: the previous implementation interpolated the raw geometry
        # string, producing "wi-block group-connected.json" (a broken URL).
        mock_http.route(content=b"{}")

        dualgraphs20(us.states.WI, tmp_path / "dg.json", geometry="block group")

        assert " " not in mock_http.urls[0]
        assert mock_http.urls[0].endswith("wi-bg-connected.json")

    def test_writes_response_body_to_disk(self, mock_http: MockHTTP, tmp_path: Path):
        payload = b'{"nodes": [1, 2, 3]}'
        mock_http.route(content=payload)
        out = tmp_path / "dg.json"

        dualgraphs20(us.states.WI, out, geometry="bg")

        assert out.read_bytes() == payload
        assert mock_http.requests[0].headers["if-match"] == '"bc206fc60ddc9b831529da730ce43313"'

    def test_invalid_geometry_raises_without_any_request(self, mock_http: MockHTTP, tmp_path: Path):
        with pytest.raises(ValueError, match="not available as a dual graph"):
            dualgraphs20(us.states.WI, tmp_path / "dg.json", geometry="tract")

        assert mock_http.requests == []

    def test_http_error_raises_and_leaves_no_file(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(status_code=404, text="not found")
        out = tmp_path / "dg.json"

        with pytest.raises(httpx.HTTPStatusError):
            dualgraphs20(us.states.WI, out, geometry="bg")

        # The body streams to a temp file that is only renamed onto the destination on success,
        # so a failed download never leaves a truncated or error-page file behind.
        assert not out.exists()
        assert list(tmp_path.iterdir()) == []

    def test_checksum_mismatch_has_actionable_error(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(status_code=412, text="precondition failed")
        out = tmp_path / "dg.json"

        with pytest.raises(RuntimeError, match="pinned checksum"):
            dualgraphs20(us.states.WI, out, geometry="bg")

        assert not out.exists()
        assert list(tmp_path.iterdir()) == []


# =============================
# == VTD SHAPEFILES (vtds20) ==
# =============================


class TestVTDs:
    def test_builds_expected_url_with_uppercase_abbr(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(content=b"zipbytes")

        vtds20(us.states.WI, tmp_path / "vtd.zip")

        assert mock_http.urls == [f"{DATA_MGGG_BASE_URL}/vtd-shapefiles/WI_vtd20.zip"]

    def test_writes_response_body_to_disk(self, mock_http: MockHTTP, tmp_path: Path):
        payload = b"PK\x03\x04 zip contents"
        mock_http.route(content=payload)
        out = tmp_path / "vtd.zip"

        vtds20(us.states.WI, out)

        assert out.read_bytes() == payload

    def test_http_error_raises_and_leaves_no_file(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(status_code=500, text="server error")
        out = tmp_path / "vtd.zip"

        with pytest.raises(httpx.HTTPStatusError):
            vtds20(us.states.WI, out)

        assert not out.exists()


# ======================================
# == CENSUS GEOMETRIES (geometries20) ==
# ======================================


class TestGeometries:
    def test_defaults_to_tract(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(content=b"zip")

        geometries20(us.states.WI, tmp_path / "geo.zip")

        assert mock_http.urls == [f"{DATA_MGGG_BASE_URL}/census-2020/wi/wi_tract.zip"]

    @pytest.mark.parametrize(
        "geometry,expected_id",
        [
            # Block groups accept the same spellings as the dual-graph ids.
            ("bg", "bg"),
            ("block group", "bg"),
            ("blockgroup", "bg"),
            ("block", "block"),
            ("congress", "cd116"),
            ("county", "county"),
            ("cousub", "cousub"),
            ("place", "place"),
            ("senate", "sldu"),
            ("house", "sldl"),
            ("tract", "tract"),
            ("vtd", "vtd"),
        ],
    )
    def test_maps_geometry_to_identifier(
        self, mock_http: MockHTTP, tmp_path: Path, geometry, expected_id
    ):
        mock_http.route(content=b"zip")

        geometries20(us.states.WI, tmp_path / "geo.zip", geometry=geometry)

        assert mock_http.urls == [f"{DATA_MGGG_BASE_URL}/census-2020/wi/wi_{expected_id}.zip"]

    def test_invalid_geometry_raises_without_any_request(self, mock_http: MockHTTP, tmp_path: Path):
        with pytest.raises(ValueError, match="not allowed"):
            geometries20(us.states.WI, tmp_path / "geo.zip", geometry="nonsense")

        assert mock_http.requests == []

    def test_http_error_raises_and_leaves_no_file(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(status_code=403, text="forbidden")
        out = tmp_path / "geo.zip"

        with pytest.raises(httpx.HTTPStatusError):
            geometries20(us.states.WI, out)

        assert not out.exists()

    def test_unpublished_object_raises_without_any_request(
        self, mock_http: MockHTTP, tmp_path: Path
    ):
        with pytest.raises(ValueError, match="No published checksum"):
            geometries20(us.states.AS, tmp_path / "geo.zip", geometry="block")

        assert mock_http.requests == []


# ============================
# == CROSS-CUTTING BEHAVIOR ==
# ============================


class _ExplodingStream(httpx.SyncByteStream):
    """Response body that fails partway through, simulating a dropped connection."""

    def __iter__(self):
        yield b"partial bytes"
        raise RuntimeError("mid-body failure")


class TestAtomicDownload:
    def test_mid_body_failure_leaves_no_file(self, mock_http: MockHTTP, tmp_path: Path):
        # Regression: the destination used to be opened before the body was consumed, so a
        # mid-stream failure left a truncated file behind.
        mock_http.route(responder=lambda request: httpx.Response(200, stream=_ExplodingStream()))
        out = tmp_path / "vtd.zip"

        with pytest.raises(RuntimeError, match="mid-body"):
            vtds20(us.states.WI, out)

        assert not out.exists()
        # The temp file is cleaned up too.
        assert list(tmp_path.iterdir()) == []

    def test_mid_body_failure_preserves_existing_destination(
        self, mock_http: MockHTTP, tmp_path: Path
    ):
        mock_http.route(responder=lambda request: httpx.Response(200, stream=_ExplodingStream()))
        out = tmp_path / "vtd.zip"
        out.write_bytes(b"previous good download")

        with pytest.raises(RuntimeError, match="mid-body"):
            vtds20(us.states.WI, out)

        assert out.read_bytes() == b"previous good download"
        assert list(tmp_path.iterdir()) == [out]

    def test_successful_download_leaves_only_the_destination(
        self, mock_http: MockHTTP, tmp_path: Path
    ):
        payload = b"zip contents"
        mock_http.route(content=payload)
        out = tmp_path / "vtd.zip"

        vtds20(us.states.WI, out)

        assert out.read_bytes() == payload
        assert list(tmp_path.iterdir()) == [out]

    def test_new_download_uses_normal_file_permissions(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(content=b"zip contents")
        reference = tmp_path / "reference"
        reference.write_bytes(b"")
        out = tmp_path / "vtd.zip"

        vtds20(us.states.WI, out)

        assert stat.S_IMODE(out.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)

    def test_replacing_download_preserves_permissions(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(content=b"new contents")
        out = tmp_path / "vtd.zip"
        out.write_bytes(b"old contents")
        out.chmod(0o640)

        vtds20(us.states.WI, out)

        assert out.read_bytes() == b"new contents"
        assert stat.S_IMODE(out.stat().st_mode) == 0o640


class TestCommonBehavior:
    def test_accepts_str_path(self, mock_http: MockHTTP, tmp_path: Path):
        payload = b"zip"
        mock_http.route(content=payload)
        out = tmp_path / "geo.zip"

        # Pass the destination as a plain string rather than a Path.
        geometries20(us.states.WI, str(out), geometry="county")

        assert out.read_bytes() == payload

    def test_multichunk_body_is_written_intact(self, mock_http: MockHTTP, tmp_path: Path):
        # A body larger than httpx's internal chunk size exercises the streamed
        # iter_bytes loop across multiple chunks.
        payload = b"abcd" * 100_000
        mock_http.route(content=payload)
        out = tmp_path / "vtd.zip"

        vtds20(us.states.WI, out)

        assert out.read_bytes() == payload

    def test_logs_download_at_info(
        self, mock_http: MockHTTP, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        mock_http.route(content=b"{}")

        with caplog.at_level(logging.INFO, logger="gerrytools.data.geometries"):
            dualgraphs20(us.states.WI, tmp_path / "dg.json", geometry="bg")

        assert any("dual graph" in record.message for record in caplog.records)
