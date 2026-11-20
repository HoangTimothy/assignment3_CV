import ast
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("environment", ["colab", "kaggle"])
def test_notebook_structure_and_code(environment):
    notebook = json.loads(Path(f"notebooks/{environment}_runner.ipynb").read_text())
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert len(cells) >= 20
    assert len({c["id"] for c in cells}) == len(cells)
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            code = "".join(line for line in cell["source"] if not line.startswith(("!", "%")))
            ast.parse(code)
    text = json.dumps(notebook)
    assert "--accept-terms" in text
    assert "RUN_FINAL_TEST = False" in text
