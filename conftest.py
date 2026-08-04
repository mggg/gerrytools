import re

import pytest

OPT_IN_MARKERS = ("latex", "snapshot")
_MARKEXPR_TOKEN_RE = re.compile(r"[A-Za-z_]+")


def pytest_addoption(parser):
    group = parser.getgroup("gerrytools")
    group.addoption(
        "--with-latex",
        action="store_true",
        default=False,
        help="Include tests marked 'latex'.",
    )
    group.addoption(
        "--with-snapshot",
        action="store_true",
        default=False,
        help="Include tests marked 'snapshot'.",
    )


def pytest_runtestloop(session):
    markexpr = (session.config.option.markexpr or "").strip()
    mentioned_markers = set(_MARKEXPR_TOKEN_RE.findall(markexpr)).intersection(OPT_IN_MARKERS)
    missing_flags = [
        marker
        for marker in mentioned_markers
        if not session.config.getoption(f"--with-{marker}")
        and not (marker == "snapshot" and session.config.getoption("--with-latex"))
        and any(item.get_closest_marker(marker) is not None for item in session.items)
    ]
    if missing_flags:
        flags = " ".join(f"--with-{marker}" for marker in sorted(missing_flags))
        raise pytest.UsageError(f"-m selected opt-in tests; pass {flags} to run them")


def pytest_collection_modifyitems(config, items):
    include_latex = config.getoption("--with-latex")
    include_snapshot = config.getoption("--with-snapshot")

    skip_latex = pytest.mark.skip(reason="need --with-latex to run")
    skip_snapshot = pytest.mark.skip(reason="need --with-snapshot or --with-latex to run")

    for item in items:
        has_snapshot_marker = item.get_closest_marker("snapshot") is not None
        has_latex_marker = item.get_closest_marker("latex") is not None

        if has_snapshot_marker:
            if not (include_snapshot or include_latex):
                item.add_marker(skip_snapshot)
            continue

        if has_latex_marker and not include_latex:
            item.add_marker(skip_latex)
