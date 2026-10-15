"""Release candidate allocation and promotion checks."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "release_metadata", Path(__file__).resolve().parents[1] / ".github/scripts/release_metadata.py"
)
assert spec is not None and spec.loader is not None
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_candidate_number_uses_only_matching_release_candidates() -> None:
    assert release.next_candidate("2.0.2", {}) == "2.0.2rc1"
    assert (
        release.next_candidate(
            "2.0.2", {version: [] for version in ["2.0.2", "2.0.2rc2", "2.0.2rc10", "2.0.3rc99"]}
        )
        == "2.0.2rc11"
    )


def candidate_and_run() -> tuple[dict, dict]:
    return (
        {"base_version": "2.0.2", "version": "2.0.2rc1", "sha": "abc123"},
        {
            "conclusion": "success",
            "event": "workflow_dispatch",
            "path": ".github/workflows/release.yml",
            "head_branch": "main",
        },
    )


def test_final_release_accepts_tested_source() -> None:
    candidate, run = candidate_and_run()
    release.verify_candidate(candidate, run, "2.0.2", "abc123")


@pytest.mark.parametrize(
    ("key", "value"),
    [("sha", "other"), ("base_version", "2.0.3"), ("version", "2.0.2")],
)
def test_final_release_rejects_different_candidate(key: str, value: str) -> None:
    candidate, run = candidate_and_run()
    candidate[key] = value
    with pytest.raises(ValueError, match="exact source commit"):
        release.verify_candidate(candidate, run, "2.0.2", "abc123")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("conclusion", "failure"),
        ("conclusion", None),
        ("event", "pull_request"),
        ("path", ".github/workflows/tests.yml"),
        ("head_branch", "dev"),
    ],
)
def test_final_release_rejects_unverified_run(key: str, value: str | None) -> None:
    candidate, run = candidate_and_run()
    run[key] = value
    with pytest.raises(ValueError, match="successful release workflow"):
        release.verify_candidate(candidate, run, "2.0.2", "abc123")


@pytest.mark.parametrize("destination", ["testpypi", "pypi"])
def test_release_selection_records_version_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    import json

    monkeypatch.chdir(tmp_path)
    (tmp_path / "source").mkdir()
    (tmp_path / "source/pyproject.toml").write_text('[project]\nversion = "2.0.2"\n')
    candidate, run = candidate_and_run()
    (tmp_path / "candidate").mkdir()
    (tmp_path / "candidate/release-metadata.json").write_text(json.dumps(candidate))
    for name, value in {
        "RELEASE_TAG": "v2.0.2",
        "UPLOAD_TO": destination,
        "CANDIDATE_RUN": "123",
        "GITHUB_REPOSITORY": "mggg/gerrytools",
        "GH_TOKEN": "test-token",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(release.subprocess, "check_output", lambda *args, **kwargs: "abc123\n")
    monkeypatch.setattr(
        release,
        "read_json",
        lambda url, token=None: {"releases": {"2.0.2rc1": []}} if token is None else run,
    )
    release.main()
    metadata = json.loads((tmp_path / "release-metadata.json").read_text())
    expected = "2.0.2rc2" if destination == "testpypi" else "2.0.2"
    assert metadata == {"version": expected, "base_version": "2.0.2", "sha": "abc123"}
    assert f"version={expected}\n" in (tmp_path / "output").read_text()
    assert expected in (tmp_path / "summary").read_text()
