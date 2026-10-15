"""Choose candidate versions and bind final releases to a verified candidate run."""

import json
import os
import re
import subprocess
import tomllib
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def next_candidate(base: str, releases: dict) -> str:
    pattern = re.compile(rf"{re.escape(base)}rc(\d+)")
    numbers = [int(match[1]) for version in releases if (match := pattern.fullmatch(version))]
    return f"{base}rc{max(numbers, default=0) + 1}"


def read_json(url: str, token: str | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def verify_candidate(candidate: dict, run: dict, base: str, sha: str) -> None:
    if (
        run["conclusion"] != "success"
        or run["event"] != "workflow_dispatch"
        or run["path"] != ".github/workflows/release.yml"
        or run["head_branch"] != "main"
    ):
        raise ValueError(
            "Candidate must come from a successful release workflow dispatched on main"
        )
    if (
        candidate["base_version"] != base
        or candidate["sha"] != sha
        or not re.fullmatch(rf"{re.escape(base)}rc[1-9]\d*", candidate["version"])
    ):
        raise ValueError(
            "Release tag must match the candidate's exact source commit and base version"
        )


def main() -> None:
    source = Path("source")
    base = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", base) or os.environ["RELEASE_TAG"] != f"v{base}":
        raise ValueError("Release tag must be vX.Y.Z and match the source project's stable version")
    sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if os.environ["UPLOAD_TO"] == "testpypi":
        try:
            releases = read_json("https://test.pypi.org/pypi/gerrytools/json")["releases"]
        except HTTPError as error:
            if error.code != 404:
                raise
            releases = {}
        version = next_candidate(base, releases)
    else:
        run_id = os.environ["CANDIDATE_RUN"]
        if not run_id.isdecimal():
            raise ValueError("A numeric successful candidate run ID is required for PyPI")
        run = read_json(
            f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}",
            os.environ["GH_TOKEN"],
        )
        candidate = json.loads(Path("candidate/release-metadata.json").read_text())
        verify_candidate(candidate, run, base, sha)
        version = base
    metadata = {"sha": sha, "base_version": base, "version": version}
    Path("release-metadata.json").write_text(json.dumps(metadata) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key, value in metadata.items():
            print(f"{key}={value}", file=output)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write(f"Building **gerrytools {version}** from `{sha}`.\n")


if __name__ == "__main__":
    main()
