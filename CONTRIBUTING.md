# Contributing to GerryTools

Thanks for your interest in contributing to GerryTools! Contributions of all sizes are welcome,
including bug reports, documentation improvements, tests, examples, and new features.

If you are planning anything larger than a small bug fix or documentation change, please contact
`code@mggg.org` before you start coding so the maintainers can help you line up with the
current roadmap and target branch.

## Ways to contribute

- Report bugs or confusing behavior.
- Improve or expand documentation.
- Add tests for uncovered behavior or regressions.
- Fix bugs or edge cases.
- Propose or implement new election, cleaning, metrics, or visualization features.

## Development setup

GerryTools uses:

- [uv](https://astral.sh/uv/) for environment and dependency management
- [go-task](https://taskfile.dev/) for common development commands
- [Ruff](https://docs.astral.sh/ruff/) for formatting, import sorting, and linting
- [ty](https://github.com/astral-sh/ty) for type checking
- [Pyright](https://github.com/microsoft/pyright) for type checking
- [pytest](https://docs.pytest.org/) for tests
- [pre-commit](https://pre-commit.com/) for local quality checks
- [maturin](https://www.maturin.rs/) and a stable Rust toolchain to build the scoring engine

Recommended setup:

1. Install `go-task`.
2. Fork and clone the repository.
3. From the repository root, run `task setup`.

`task setup` installs Astral's official standalone `uv` if you don't have it, installs a managed
Python 3.11 environment, syncs the pinned dependencies, and installs the pre-commit hooks.

If you already have `uv` installed and prefer to run the steps directly, the equivalent setup is:

```bash
uv python install 3.11
uv --managed-python sync --locked --all-groups --all-extras --python 3.11
uv run pre-commit install
```

### The Rust scoring engine

`gerrytools.scoring` is backed by a compiled extension module (`gerrytools._scoring_engine`) whose
source lives in `rust/`. The package builds with the maturin backend, so `uv sync` compiles the
crate for you and a stable Rust toolchain must be on your `PATH`.

`task setup` does **not** install Rust. If you do not have it, install it from
[rustup.rs](https://rustup.rs/) and add the components CI uses:

```bash
rustup component add clippy rustfmt
```

You only need this section if you touch `rust/`. Contributors working purely in Python can rely on
`uv sync` and skip ahead.

## Contributor workflow

1. Fork the repository and clone your fork locally.
2. Create a descriptive branch from the current target branch.
3. Keep your change focused. Avoid bundling unrelated refactors into the same pull request.
4. Add or update tests for any behavior change.
5. Run the relevant checks locally before you open a PR.
6. Open a pull request with a clear summary of the problem, your approach, and how you tested it.

If you are unsure which branch to target, ask the maintainers before opening the PR.

### Branch naming

Use descriptive branch names such as:

- `fix/score-profile-csv-validation`
- `feat/add-schulze-example`
- `docs/update-contributing-guide`

## Running checks locally

Preferred `task` commands:

```bash
task all-checks           # format-check, lint, typecheck, and the Rust checks
task format
task lint
task lint -- --fix
task typecheck
task test
task test -- <pytest cli args>
task test:tests/path
task coverage
task docs
```

`task all-checks` is the closest single command to what CI gates on. `task check` runs only
formatting and linting, so it will not catch type or Rust failures.

If you already have `uv` on your `PATH`, the equivalent direct commands are:

```bash
uv run ruff check --select I --fix gerrytools tests
uv run ruff format gerrytools tests
uv run ruff check gerrytools tests
uv run ty check gerrytools tests
uv run pyright gerrytools tests
uv run pytest tests
uv run pytest tests --cov=gerrytools --cov-branch --cov-report=term-missing
uv run pre-commit run --all-files
```

### Rust checks

If you change anything under `rust/`, run:

```bash
task rust-check           # cargo fmt --check and clippy with warnings denied
task test-rust            # cargo test --locked --all-features
task coverage-rust        # cargo llvm-cov against the 85% line threshold
```

`task coverage-rust` needs `cargo-llvm-cov`, which is not part of `task setup`:

```bash
cargo install cargo-llvm-cov --version 0.8.4 --locked
```

That is the version CI pins, so matching it avoids threshold surprises.

**Run cargo through `uv run` when you pass `--all-features`.** That flag enables the `python`
feature, which links the test binary against `libpython`, and a bare invocation will fail to start:

```console
$ cargo test --manifest-path rust/Cargo.toml --all-features
error while loading shared libraries: libpython3.13.so.1.0: cannot open shared object file
```

`uv run` puts the managed interpreter's shared library on the loader path, which is why
`task test-rust` works. The other cargo commands do not enable that feature and run fine on their
own.

Notes:

- To scope a Task-based test run, use `task test -- tests/<path>` or `task test:tests/<path>`.
- `task coverage` runs the default test suite with a terminal coverage summary for `gerrytools`.
- If you change public documentation or tutorial content, run `task docs`.
- If you touch plotting behavior, run `task snapshots-verify`.
- If you touch LaTeX rendering, run `task snapshots-latex-verify`.

## Pull request expectations

Before opening a pull request, make sure that:

- the change is scoped to a single topic
- code, tests, and docs are updated together when needed
- new behavior is covered by tests
- linting, formatting, and type checks pass locally (`task all-checks`)
- Rust changes pass `task rust-check` and `task test-rust`
- user-facing changes are recorded under `## [Unreleased]` in `CHANGELOG.md`, in the
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) section that fits (`Added`, `Changed`,
  `Deprecated`, `Removed`, `Fixed`, or `Security`)
- the PR description explains the user-facing impact and any notable tradeoffs

Small pull requests are much easier to review and merge than large mixed changes.

## Code style guidelines

GerryTools is a Python 3.11+ codebase. When contributing, prefer the current conventions below and
avoid style-only churn in unrelated files.

- Follow the repo tooling first: Ruff defines the baseline formatting, import, and lint style.
- Keep lines at roughly 100 characters to match the configured formatter and linter settings.
- Add type annotations for function parameters and return values. Run `task typecheck` on changes
  that add or reshape APIs.
- Prefer modern type syntax in new or substantially updated code, such as `str | None` instead of
  `Optional[str]`. Older files still contain pre-3.10 style hints, and you do not need to rewrite
  them unless you are already editing that area for a substantive reason.
- Use descriptive `snake_case` names for variables and functions, `PascalCase` for classes, and
  `UPPER_SNAKE_CASE` for module-level constants.
- Keep functions focused. Small helpers are preferred over long functions with several distinct
  responsibilities.
- Put validation and obvious guard clauses near the top of a function. This pattern is common
  across the election, profile, and utility modules.
- Prefer straightforward control flow over extra abstraction. This codebase generally favors clear
  data flow and targeted helpers over deep inheritance or unnecessary indirection.
- Match the surrounding file when touching older modules. Consistency within a file is more
  important than forcing a full-file style migration.
- Use comments sparingly. Prefer names and small helper functions to explain intent, and reserve
  comments for non-obvious logic or domain-specific reasoning.

### Docstrings

Public classes, functions, and methods should have docstrings that follow the project’s existing
Google-style variant:

```python
def foo(arg1: str | None, arg2: int = 3) -> str:
    """Brief description.

    More details if needed.

    Args:
        arg1 (str | None): Description.
        arg2 (int, optional): Description. Defaults to 3.

    Returns:
        str: Description of the returned value.

    Raises:
        ValueError: Description of the failure mode.
    """
```

Docstring conventions used throughout the repository:

- Docstrings should be no more than 100 characters per line (including indents) to match the
  configured formatter settings.
- Include `Args`, `Returns`, and `Raises` when they apply.
- Document optional parameters and default behavior explicitly.
- Add examples only when they help clarify non-obvious usage.

## Testing guidelines

Tests are required for behavior changes.

- Add tests in `tests/` near the existing area that covers the same module or feature.
- Mirror the module structure where practical. For example, scoring code belongs under
  `tests/scoring/...`.
- Cover both successful behavior and expected failures.
- When raising exceptions, prefer tests that check the error message with `pytest.raises(...,
match=...)`.
- Include edge cases that are natural for the change: empty inputs, invalid candidate data,
  malformed rankings, tie handling, or zero-weight behavior.
- Mark image regression tests with `@pytest.mark.snapshot` and LaTeX-dependent tests with
  `@pytest.mark.latex`. Both are opt-in: a plain `pytest` run skips them, and they need
  `--with-snapshot` or `--with-latex` (or `task snapshots-verify` / `task snapshots-latex-verify`).
  Selecting such a test by path rather than by `-m` currently skips it and still exits 0, so check
  for `s` in the output before trusting a green run.
- Rust tests live beside the crate in `rust/src/tests/`. Add coverage there for engine changes;
  the Python suite does not count toward the Rust coverage threshold.

## Documentation guidelines

If your change affects public behavior, update the relevant documentation alongside the code.
Depending on the change, that may include:

- docstrings in `gerrytools/`
- narrative docs under `user_guide/`
- tutorial notebooks or generated tutorial pages
- examples or README references

## Cutting a release

The `release.yml` workflow does the heavy lifting: it validates the tag, builds wheels and the
sdist, smoke tests them, and publishes through TestPyPI to PyPI. The manual steps around it, in
order:

1. Bump the version (`uv version <X.Y.Z>`) and roll the `## [Unreleased]` entries in
   `CHANGELOG.md` into a `## [X.Y.Z]` section.
2. Publish the `mgggdev/replicate` Docker image under the release tag. Build it locally and run
   the live mgrp tests against it first:

   ```console
   task docker-build -- vX.Y.Z; MGRP_TEST_IMAGE=mgggdev/replicate:vX.Y.Z uv run pytest tests/mgrp/live -q
   ```

   Then publish, picking the path that matches whether `docker/` changed since the last release:

   ```console
   task docker-push -- vX.Y.Z     # docker/ changed: full multi-arch rebuild and push
   task docker-retag -- vX.Y.Z    # docker/ unchanged: retag the published digest, no rebuild
   ```

   Both tasks finish by rewriting `DEFAULT_DOCKER_IMAGE` in `gerrytools/mgrp/run_container.py` to
   the freshly published `vX.Y.Z@sha256:...` pin; commit that change with the release. If the tag
   is already on Docker Hub and only the constant needs re-pointing, `task docker-pin -- vX.Y.Z`
   rewrites the pin from the registry digest without touching the image. The release workflow
   refuses to build if the pinned tag does not match the release version or does not resolve to
   the pinned digest on Docker Hub.

3. Run `task all-checks` and `task check-release` on the release commit and merge it to `main`.
   `check-release` verifies the version is not already tagged, the changelog has a section for
   it, and the Docker pin matches it and resolves on Docker Hub.
4. Tag the merged commit `vX.Y.Z` and push the tag.
5. Dispatch `release.yml` from `main` with the tag and `upload_to: testpypi`; once the TestPyPI
   verification job passes, rerun it with `upload_to: pypi`.
6. Write the GitHub release notes from the new changelog section.

## Community guidelines

This project follows the Contributor Covenant Code of Conduct. By participating, you agree to
abide by the expectations in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Questions

If anything in the contribution process is unclear, please feel free to reach out to
`code@mggg.org` with questions. Thanks!
