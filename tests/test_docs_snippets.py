"""Execute every Python code block in the documentation.

The user guide is the single source of truth: this runner discovers ``user_guide/**/*.md`` and
``user_guide/**/*.rst``, extracts every Python code block,
and executes each page's blocks in order in a shared namespace inside a temp working directory.
Tutorial notebooks execute through ``task docs-cache-notebooks``.

Pages can annotate a block with a marker on the line before it (blank lines allowed between).
In RST the marker is a comment, in Markdown an HTML comment:

    .. docs-test: skip -- <reason>
    .. docs-test: setup

    <!-- docs-test: skip -- <reason> -->
    <!-- docs-test: setup -->

``skip`` excludes the block from execution and requires a short reason. ``setup`` blocks are run
once, before any page, in the shared working directory (e.g. to unzip downloaded data).
"""

import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import re
import sys
import textwrap
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "user_guide"

MARKERS = ("skip", "setup")

RST_BLOCK_RE = re.compile(r"^(?P<indent>[ \t]*)\.\.\s+code(?:-block)?::\s+python[ \t]*$")
RST_MARKER_RE = re.compile(
    r"^[ \t]*\.\.\s+docs-test:\s*(?P<name>[\w-]+)[ \t]*(?:--[ \t]*(?P<reason>.*\S))?[ \t]*$"
)
RST_OPTION_RE = re.compile(r"^[ \t]*:[\w-]+:")
MD_FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)```python[ \t]*$")
MD_MARKER_RE = re.compile(
    r"^[ \t]*<!--\s*docs-test:\s*(?P<name>[\w-]+)[ \t]*(?:--[ \t]*(?P<reason>.*?))?\s*-->[ \t]*$"
)

# Loose pattern used by the completeness check: anything that looks like a Python code directive
# must have been extracted by the strict scanner above (or the scanner needs fixing).
RST_LOOSE_RE = re.compile(r"code(?:-block)?::\s*python", re.IGNORECASE)
MD_LOOSE_RE = re.compile(r"^[ \t]*```python", re.MULTILINE)


@dataclass
class Block:
    page: Path
    lineno: int  # 1-based line of the block's first code line, for tracebacks
    source: str
    marker: str | None = None
    reason: str | None = None


def _check_marker(name: str, reason: str | None, page: Path, lineno: int) -> None:
    if name not in MARKERS:
        raise ValueError(f"{page}:{lineno}: unknown docs-test marker {name!r}")
    if name == "skip" and not reason:
        raise ValueError(f"{page}:{lineno}: docs-test: skip requires a reason ('-- <reason>')")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def extract_rst_blocks(text: str, page: Path) -> list[Block]:
    """Extract ``.. code-block:: python`` / ``.. code:: python`` bodies, dedented."""
    lines = text.split("\n")
    blocks: list[Block] = []
    marker: tuple[str, str | None, int] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = RST_MARKER_RE.match(line)
        if m:
            _check_marker(m["name"], m["reason"], page, i + 1)
            marker = (m["name"], m["reason"], i + 1)
            i += 1
            continue
        d = RST_BLOCK_RE.match(line)
        if d is None:
            if line.strip():
                marker = None  # markers survive only blank lines
            i += 1
            continue
        indent = len(d["indent"])
        i += 1
        # Consume option lines (":linenos:" etc.) attached directly to the directive.
        while i < len(lines) and lines[i].strip() and _indent(lines[i]) > indent:
            if not RST_OPTION_RE.match(lines[i]):
                break
            i += 1
        # Body: everything more indented than the directive, up to the indentation boundary.
        body: list[str] = []
        first_code_line = None
        while i < len(lines):
            cur = lines[i]
            if cur.strip():
                if _indent(cur) <= indent:
                    break
                if first_code_line is None:
                    first_code_line = i + 1
            if first_code_line is not None:
                body.append(cur)
            i += 1
        while body and not body[-1].strip():
            body.pop()
        source = textwrap.dedent("\n".join(body))
        blocks.append(
            Block(
                page=page,
                lineno=first_code_line or i,
                source=source,
                marker=marker[0] if marker else None,
                reason=marker[1] if marker else None,
            )
        )
        marker = None
    return blocks


def extract_md_blocks(text: str, page: Path) -> list[Block]:
    """Extract ```python fenced blocks from (MyST) Markdown, dedented."""
    lines = text.split("\n")
    blocks: list[Block] = []
    marker: tuple[str, str | None] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = MD_MARKER_RE.match(line)
        if m:
            _check_marker(m["name"], m["reason"], page, i + 1)
            marker = (m["name"], m["reason"])
            i += 1
            continue
        f = MD_FENCE_RE.match(line)
        if f is None:
            if line.strip():
                marker = None
            i += 1
            continue
        indent = f["indent"]
        i += 1
        first_code_line = i + 1
        body = []
        while i < len(lines) and lines[i].strip() != "```":
            body.append(lines[i].removeprefix(indent))
            i += 1
        i += 1  # closing fence
        blocks.append(
            Block(
                page=page,
                lineno=first_code_line,
                source="\n".join(body),
                marker=marker[0] if marker else None,
                reason=marker[1] if marker else None,
            )
        )
        marker = None
    return blocks


def extract_blocks(page: Path) -> list[Block]:
    text = page.read_text(encoding="utf-8")
    if page.suffix == ".md":
        return extract_md_blocks(text, page)
    return extract_rst_blocks(text, page)


def discover_pages() -> list[Path]:
    pages = [
        p for pattern in ("*.rst", "*.md") for p in DOCS.rglob(pattern) if "_build" not in p.parts
    ]
    return sorted(pages)


PAGES = discover_pages()
ALL_BLOCKS = {page: extract_blocks(page) for page in PAGES}
EXEC_PAGES = [page for page, blocks in ALL_BLOCKS.items() if blocks]
NOTEBOOKS = sorted((DOCS / "user").rglob("*.ipynb"))


def _page_id(page: Path) -> str:
    return str(page.relative_to(REPO))


def _run_block(block: Block, namespace: dict[str, object]) -> None:
    # Pad the source so tracebacks and SyntaxErrors point at the real line in the docs page.
    padded = "\n" * (block.lineno - 1) + block.source
    exec(compile(padded, str(block.page), "exec"), namespace)


@pytest.fixture(scope="session")
def docs_cwd(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared working directory with documentation setup blocks applied."""
    cwd = tmp_path_factory.mktemp("docs-snippets")
    old = os.getcwd()
    os.chdir(cwd)
    try:
        namespace: dict[str, object] = {}
        for page in PAGES:
            for block in ALL_BLOCKS[page]:
                if block.marker == "setup":
                    _run_block(block, namespace)
    finally:
        os.chdir(old)
    return cwd


@pytest.mark.parametrize("page", EXEC_PAGES, ids=_page_id)
def test_page_snippets(page: Path, docs_cwd: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(docs_cwd)
    namespace: dict[str, object] = {}
    try:
        for block in ALL_BLOCKS[page]:
            if block.marker in MARKERS:
                continue
            _run_block(block, namespace)
    finally:
        if "matplotlib.pyplot" in sys.modules:
            sys.modules["matplotlib.pyplot"].close("all")


@pytest.mark.parametrize("page", PAGES, ids=_page_id)
def test_every_python_block_is_extracted(page: Path) -> None:
    """Anything that looks like a Python block must be extracted (or the scanner fixed)."""
    text = page.read_text(encoding="utf-8")
    loose_re = MD_LOOSE_RE if page.suffix == ".md" else RST_LOOSE_RE
    assert len(loose_re.findall(text)) == len(ALL_BLOCKS[page])


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=_page_id)
def test_notebooks_use_portable_markdown(notebook: Path) -> None:
    """Notebook Markdown must render in Jupyter as well as in myst-nb."""
    data = json.loads(notebook.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "markdown"
    )
    assert not re.search(r"^`{3,4}\{", markdown, re.MULTILINE)
    assert not re.search(r"\{(?:doc|class|meth|func|mod|ref|attr|data|exc)\}`", markdown)
    assert not re.search(r"^\([a-z0-9-]+\)=$", markdown, re.MULTILINE)
    assert "../_images/" not in markdown
    assert "<img" not in markdown
    for image in re.findall(r"!\[[^]]*\]\(([^)]+)\)", markdown):
        if "://" not in image:
            assert (notebook.parent / image).is_file(), image
    for cell in data["cells"]:
        source = cell.get("source", [])
        for index, line in enumerate(source):
            if "notebook-admonition-title" not in line or not line.startswith("> "):
                continue
            assert "blockquote:has(.notebook-admonition-title)" in markdown
            assert source[index + 1] == ">\n"
            assert source[index + 2].startswith("> ")


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=_page_id)
def test_notebooks_do_not_commit_outputs(notebook: Path) -> None:
    """Generated notebook outputs belong in the ignored documentation cache."""
    data = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        assert not cell.get("outputs")
        assert cell.get("execution_count") is None


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=_page_id)
def test_notebooks_have_normalized_kernel_metadata(notebook: Path) -> None:
    """Committed notebooks must carry the pinned kernel metadata.

    A notebook re-saved in a local Jupyter kernel picks up a machine-specific ``kernelspec`` and a
    ``language_info`` block. That changes the jupyter-cache key the docs build computes, so the
    build misses the pre-built cache and re-executes the notebook in a CWD without the sample data,
    which fails on CI. ``user_guide/_clear_notebook_outputs.py`` pins this metadata; this test fails
    loudly with a fixup hint if a drifted notebook is committed anyway.
    """
    metadata = json.loads(notebook.read_text(encoding="utf-8")).get("metadata", {})
    fixup = f"run `python user_guide/_clear_notebook_outputs.py {notebook}` to fix"
    assert metadata.get("kernelspec") == {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }, f"{notebook.name} has a non-normalized kernelspec; {fixup}"
    assert "language_info" not in metadata, (
        f"{notebook.name} still carries a language_info block; {fixup}"
    )


def test_plotting_guide_has_one_notebook_per_public_format() -> None:
    plotting = DOCS / "user" / "plotting"
    expected_statistical = {
        "bar.ipynb",
        "box.ipynb",
        "histogram.ipynb",
        "paintball.ipynb",
        "scatter.ipynb",
        "sealevel.ipynb",
        "seats_votes.ipynb",
        "violin.ipynb",
    }
    expected_geographic = {"dot_density.ipynb", "geo.ipynb", "graph.ipynb", "subway.ipynb"}
    assert expected_statistical <= {
        path.name for path in (plotting / "statistical").glob("*.ipynb")
    }
    assert expected_geographic <= {path.name for path in (plotting / "geographic").glob("*.ipynb")}


def test_plotting_guide_has_main_and_advanced_notebooks() -> None:
    plotting = DOCS / "user" / "plotting"
    expected = {
        plotting / "overview.ipynb",
        plotting / "composition.ipynb",
        plotting / "statistical" / "workflow.ipynb",
        plotting / "geographic" / "workflow.ipynb",
    }
    assert all(path.is_file() for path in expected)


def test_plotting_navigation_hubs_cover_every_notebook() -> None:
    """The plotting hubs own the nested toctrees Furo renders in the sidebar."""
    plotting = DOCS / "user" / "plotting"
    hub = (plotting / "index.md").read_text(encoding="utf-8")
    for child in ("overview", "statistical/index", "geographic/index", "composition"):
        assert f"<{child}>" in hub

    for category in ("statistical", "geographic"):
        category_dir = plotting / category
        category_hub = (category_dir / "index.md").read_text(encoding="utf-8")
        for path in category_dir.glob("*.ipynb"):
            assert f"<{path.stem}>" in category_hub, f"{path.name} missing from {category}/index.md"

    root = (DOCS / "index.md").read_text(encoding="utf-8")
    assert "<user/plotting/index>" in root
    assert "user/plotting/statistical/histogram" not in root


def test_latex_guide_has_rendered_examples() -> None:
    """Every committed LaTeX image is referenced by a guide, and every reference resolves."""
    image_dir = DOCS / "_static" / "images" / "latex"
    committed = {path.name for path in image_dir.glob("*.png")}
    assert committed, "no committed LaTeX images found"

    referenced: set[str] = set()
    for notebook_path in sorted((DOCS / "user" / "latex").glob("*.ipynb")):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        for name in committed:
            if f"../../_static/images/latex/{name}" in markdown:
                referenced.add(name)
    assert referenced == committed, (
        f"unreferenced images: {sorted(committed - referenced)}; "
        f"missing files: {sorted(referenced - committed)}"
    )

    manifest = (image_dir / "README.md").read_text(encoding="utf-8")
    for name in sorted(committed):
        assert f"`{name}`" in manifest

    hub = (DOCS / "user" / "latex" / "index.md").read_text(encoding="utf-8")
    for child in ("textable", "tikztable", "tikztable_styling", "paintball", "seats_votes"):
        assert f"<{child}>" in hub


def test_clear_notebook_outputs(tmp_path: Path) -> None:
    path = tmp_path / "example.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "id": "example",
                        "metadata": {},
                        "outputs": [{"name": "stdout", "output_type": "stream", "text": ["1\n"]}],
                        "source": ["print(1)"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    spec = spec_from_file_location("clear_notebook_outputs", DOCS / "_clear_notebook_outputs.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.clear_notebook_outputs(path)
    assert not module.clear_notebook_outputs(path)
    cell = json.loads(path.read_text(encoding="utf-8"))["cells"][0]
    assert cell["outputs"] == []
    assert cell["execution_count"] is None


def test_notebook_cache_matches_source_content(tmp_path: Path) -> None:
    path = tmp_path / "example.ipynb"
    spec = spec_from_file_location("refresh_notebooks", DOCS / "_refresh_notebooks.py")
    assert spec is not None and spec.loader is not None
    refresh = module_from_spec(spec)
    spec.loader.exec_module(refresh)

    source = refresh.nbformat.v4.new_notebook(cells=[refresh.nbformat.v4.new_code_cell("print(1)")])
    refresh.nbformat.write(source, path)
    executed = refresh.nbformat.read(path, as_version=4)
    executed.cells[0].execution_count = 1
    executed.cells[0].outputs = [refresh.nbformat.v4.new_output("stream", text="1\n")]
    cache = refresh.get_cache(str(tmp_path / "cache"))
    cache.cache_notebook_bundle(refresh.CacheBundleIn(executed, str(path)), check_validity=False)

    assert refresh.notebook_is_cached(path, cache)
    source.cells[0].source = "print(2)"
    refresh.nbformat.write(source, path)
    assert not refresh.notebook_is_cached(path, cache)


def test_notebook_execution_preserves_source_metadata(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "example.ipynb"
    spec = spec_from_file_location("refresh_notebooks", DOCS / "_refresh_notebooks.py")
    assert spec is not None and spec.loader is not None
    refresh = module_from_spec(spec)
    spec.loader.exec_module(refresh)

    source = refresh.nbformat.v4.new_notebook(
        cells=[refresh.nbformat.v4.new_code_cell("print(1)")],
        metadata={
            "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.0"},
        },
    )
    refresh.nbformat.write(source, path)

    def fake_execute(client):
        client.nb.metadata["language_info"]["version"] = "3.11.0"
        client.nb.cells[0].execution_count = 1
        client.nb.cells[0].outputs = [refresh.nbformat.v4.new_output("stream", text="1\n")]
        return client.nb

    monkeypatch.setattr(refresh.NotebookClient, "execute", fake_execute)

    executed = refresh.execute(path)
    cache = refresh.get_cache(str(tmp_path / "cache"))
    cache.cache_notebook_bundle(
        refresh.CacheBundleIn(executed, str(path)),
        check_validity=False,
    )

    assert executed.metadata == source.metadata
    assert refresh.notebook_is_cached(path, cache)


# ---------------------------------------------------------------------------
# Extractor unit tests
# ---------------------------------------------------------------------------

FAKE = Path("fake.rst")


def rst_blocks(text: str) -> list[Block]:
    return extract_rst_blocks(textwrap.dedent(text), FAKE)


class TestRstExtractor:
    def test_top_level_block(self) -> None:
        blocks = rst_blocks(
            """\
            Some prose.

            .. code-block:: python

                x = 1
                y = x + 1

            More prose.
            """
        )
        assert len(blocks) == 1
        assert blocks[0].source == "x = 1\ny = x + 1"
        assert blocks[0].lineno == 5

    def test_code_directive_variant_and_trailing_whitespace(self) -> None:
        blocks = rst_blocks(".. code:: python  \n\n    x = 1\n")
        assert len(blocks) == 1
        assert blocks[0].source == "x = 1"

    def test_nested_block_terminates_at_directive_indent(self) -> None:
        blocks = rst_blocks(
            """\
            .. admonition:: Note

                Some explanation.

                .. code-block:: python

                    x = 1

                This prose is back at the admonition level, not code.
            """
        )
        assert len(blocks) == 1
        assert blocks[0].source == "x = 1"

    def test_adjacent_blocks(self) -> None:
        blocks = rst_blocks(
            """\
            .. code-block:: python

                x = 1

            .. code-block:: python

                y = 2
            """
        )
        assert [b.source for b in blocks] == ["x = 1", "y = 2"]

    def test_interior_blank_lines_preserved(self) -> None:
        blocks = rst_blocks(".. code-block:: python\n\n    x = 1\n\n    y = 2\n")
        assert blocks[0].source == "x = 1\n\ny = 2"

    def test_option_lines_are_not_code(self) -> None:
        blocks = rst_blocks(
            """\
            .. code-block:: python
                :linenos:

                x = 1
            """
        )
        assert blocks[0].source == "x = 1"

    def test_skip_marker_with_reason(self) -> None:
        blocks = rst_blocks(
            """\
            .. docs-test: skip -- needs network access

            .. code-block:: python

                x = 1
            """
        )
        assert blocks[0].marker == "skip"
        assert blocks[0].reason == "needs network access"

    def test_skip_without_reason_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="requires a reason"):
            rst_blocks(".. docs-test: skip\n\n.. code-block:: python\n\n    x = 1\n")

    def test_unknown_marker_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="unknown docs-test marker"):
            rst_blocks(".. docs-test: sikp -- oops\n\n.. code-block:: python\n\n    x = 1\n")

    def test_setup_marker(self) -> None:
        blocks = rst_blocks(".. docs-test: setup\n\n.. code-block:: python\n\n    x = 1\n")
        assert blocks[0].marker == "setup"

    def test_marker_does_not_survive_intervening_prose(self) -> None:
        blocks = rst_blocks(
            """\
            .. docs-test: skip -- for the next block

            Some prose in between.

            .. code-block:: python

                x = 1
            """
        )
        assert blocks[0].marker is None

    def test_non_python_blocks_ignored(self) -> None:
        assert rst_blocks(".. code-block:: console\n\n    $ ls\n") == []

    def test_extracted_source_is_executable(self) -> None:
        blocks = rst_blocks(
            """\
            .. code-block:: python

                def f():
                    return 41

                x = f() + 1
            """
        )
        namespace: dict[str, object] = {}
        _run_block(blocks[0], namespace)
        assert namespace["x"] == 42


class TestMdExtractor:
    def test_fence_with_marker(self) -> None:
        text = textwrap.dedent(
            """\
            Some prose.

            <!-- docs-test: skip -- interactive only -->

            ```python
            x = 1
            ```
            """
        )
        blocks = extract_md_blocks(text, Path("fake.md"))
        assert len(blocks) == 1
        assert blocks[0].source == "x = 1"
        assert blocks[0].marker == "skip"
        assert blocks[0].reason == "interactive only"

    def test_plain_fence_and_non_python_ignored(self) -> None:
        text = "```python\nx = 1\ny = 2\n```\n\n```console\n$ ls\n```\n"
        blocks = extract_md_blocks(text, Path("fake.md"))
        assert len(blocks) == 1
        assert blocks[0].source == "x = 1\ny = 2"
        assert blocks[0].marker is None
