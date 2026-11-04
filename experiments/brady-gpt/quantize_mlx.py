"""Make 8-bit MLX copies of merged models, to run them on the Mac with chat.py.

    uv run --extra mac python quantize_mlx.py bradyrutherford/minerva-olmo3-7b-B \
        --out checkpoints/7b/mlx8-dpobest-B [--revision sft] [--bits 8]

The full bf16 model comes from my private HF repo (or a local folder) and is never changed.
transformers 5 saves the RoPE settings per attention type under `rope_parameters`; mlx_lm's
OLMo 3 reads the older top-level `rope_theta` and `rope_scaling` (YaRN, which it applies to the
full-attention layers only). So the model folder is mirrored with symlinks and only config.json
gets those two keys, copied from rope_parameters. Before trusting the copy, score it with
eval_compare.py against the full model's per-answer losses. Uploading is a separate step: mlx_lm's
own --upload-repo creates the repo public.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")  # the full model downloads on first use


def mlx_config(config):
    """Add the top-level RoPE keys mlx_lm's OLMo 3 expects, from transformers 5's rope_parameters."""
    rope = config.get("rope_parameters")
    if not rope or "rope_theta" in config:
        return config
    full, sliding = rope["full_attention"], rope["sliding_attention"]
    assert full["rope_theta"] == sliding["rope_theta"], "mlx_lm uses one rope_theta for both"
    assert sliding.get("rope_type", "default") == "default", "mlx_lm's sliding layers use plain RoPE"
    return {**config, "rope_theta": full["rope_theta"], "rope_scaling": full}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF repo id or local folder of a merged (full) model")
    ap.add_argument("--revision")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()
    src = Path(args.model)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(args.model, revision=args.revision, ignore_patterns=["adapters/*"]))
    with tempfile.TemporaryDirectory() as tmp:
        for f in src.iterdir():
            if f.name != "config.json":
                (Path(tmp) / f.name).symlink_to(f.resolve())
        config = json.loads((src / "config.json").read_text())
        (Path(tmp) / "config.json").write_text(json.dumps(mlx_config(config), indent=2))
        subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", tmp,
                        "--mlx-path", str(args.out), "-q", "--q-bits", str(args.bits),
                        "--q-group-size", str(args.group_size)], check=True)
    print(f"{args.bits}-bit MLX copy of {args.model} -> {args.out}")


if __name__ == "__main__":
    main()
