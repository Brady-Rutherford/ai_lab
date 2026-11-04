"""Run the live notebook on a rented GPU, without Colab, and save it with its outputs.

    HF_TOKEN=$(cat) python demo/run_with_outputs.py demo/brady_gpt_live.ipynb demo_data.json out.ipynb < token

Every cell runs exactly as written. The only stand-in is google.colab: "upload" returns the data file,
and the secret comes from the environment. The output contains class text, so keep it under data/ and
never commit it.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

STUB = '''
import json, os, pathlib, types
files = types.SimpleNamespace(upload=lambda: {"demo_data.json": pathlib.Path(os.environ["DEMO_DATA"]).read_bytes()})
userdata = types.SimpleNamespace(get=lambda name: os.environ[name])
def _refuse():
    raise RuntimeError("this notebook run must not wipe anything")
runtime = types.SimpleNamespace(unassign=_refuse)
'''


def main():
    source, data, out = map(Path, sys.argv[1:4])
    nb = nbformat.read(source, as_version=4)
    token = os.environ["HF_TOKEN"]
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "google/colab").mkdir(parents=True)
        (Path(tmp) / "google/colab/__init__.py").write_text(STUB)
        os.environ["PYTHONPATH"] = tmp
        os.environ["DEMO_DATA"] = str(data.resolve())
        start = time.time()

        def done(cell, cell_index, execute_reply):
            print(f"{time.time() - start:6.0f}s  cell {cell_index + 1} of {len(nb.cells)} done", flush=True)

        failed = None
        try:
            NotebookClient(nb, timeout=None, kernel_name="python3", on_cell_executed=done).execute()
        except Exception as error:  # keep what ran, so a failure can be read
            failed = error
    shown = json.dumps([cell.get("outputs", []) for cell in nb.cells])
    if token and token in shown:
        sys.exit("The login appeared in an output, so nothing was saved.")
    nb.metadata.colab.private_outputs = False  # this copy is for viewing; it stays on my machines
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    print(f"saved {out}")
    if failed:
        print(f"A cell failed: {type(failed).__name__}: {str(failed)[-1500:]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
