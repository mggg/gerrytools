import json
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
TUTORIALS = REPO / "user_guide" / "user" / "mgrp"
STATIC = REPO / "user_guide" / "_static"


@pytest.mark.parametrize("name", ["recom", "forest", "smc"])
def test_mgrp_tutorial_code_uses_a_shipped_input(name, monkeypatch):
    notebook = json.loads((TUTORIALS / f"{name}.ipynb").read_text())
    namespace = {}
    monkeypatch.chdir(STATIC)

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{name}.ipynb", "exec"), namespace)

    assert Path(namespace["config"].host_graph_path).is_file()
