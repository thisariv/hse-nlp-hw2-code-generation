"""nbformat validation + complete CPU dry-run in a fresh Jupyter kernel."""

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

SUBMISSION = Path(__file__).resolve().parents[1]


def main():
    notebook = nbformat.read(SUBMISSION / "inference.ipynb", as_version=4)
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix="hw2-notebook-") as directory:
        root = Path(directory)
        kernel = root / "kernels/hw2-clean-cpu"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                dict(
                    argv=[
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    display_name="HW2 CPU check",
                    language="python",
                )
            )
        )
        environment = dict(
            JUPYTER_PATH=str(root),
            JUPYTER_RUNTIME_DIR=str(root / "runtime"),
            IPYTHONDIR=str(root / "ipython"),
            HW2_DRY_RUN="1",
            HW2_CODE_ROOT=str(SUBMISSION / "code"),
            HW2_OUTPUT=str(root / "output"),
            CUDA_VISIBLE_DEVICES="",
        )
        old = {k: os.environ.get(k) for k in environment}
        os.environ.update(environment)
        try:
            NotebookClient(
                notebook,
                timeout=60,
                kernel_name="hw2-clean-cpu",
                resources={"metadata": {"path": str(SUBMISSION)}},
            ).execute()
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        nbformat.validate(notebook)
        rows = [
            json.loads(line)
            for line in (root / "output/answers.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 8
        assert all(
            set(row) == {"code"} and isinstance(row["code"], str) for row in rows
        )
        errors = [
            o
            for c in notebook.cells
            if c.cell_type == "code"
            for o in c.outputs
            if o.output_type == "error"
        ]
        assert not errors
        nbformat.write(notebook, root / "inference-cpu.executed.ipynb")
        for cell in notebook.cells:
            if cell.cell_type == "code":
                for output in cell.outputs:
                    if output.output_type == "stream":
                        print(output.text.strip())
        print("nbformat valid; fresh CPU kernel; 8 answers; GPU not used")


if __name__ == "__main__":
    main()
