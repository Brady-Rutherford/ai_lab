"""Fetch a model into the Hugging Face cache and verify every file, once, before training.

    uv run python download_model.py                       # the OLMo 2 1B base
    uv run python download_model.py allenai/OLMo-3-...    # any other model id

The other scripts run offline (common.py), so this is the only one that downloads. It
checks each file's size against the Hub, because an interrupted download can leave a
partial shard behind that only surfaces later as a stalled training run.
"""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

from common import BASE_MODEL  # noqa: E402

PATTERNS = ["*.json", "*.safetensors", "*.txt"]


def main(model_id):
    path = Path(snapshot_download(model_id, allow_patterns=PATTERNS))
    info = HfApi().model_info(model_id, files_metadata=True)
    wanted = [s for s in info.siblings if s.rfilename.endswith((".json", ".safetensors", ".txt"))]
    bad = [s.rfilename for s in wanted
           if not (path / s.rfilename).exists() or (path / s.rfilename).stat().st_size != s.size]
    if bad:
        print(f"INCOMPLETE {model_id}: {bad}")
        return 1
    print(f"verified {model_id}: {len(wanted)} files, "
          f"{sum(s.size or 0 for s in wanted) / 1e9:.2f} GB at {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else BASE_MODEL))
