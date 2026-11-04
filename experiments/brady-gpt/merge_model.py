"""Fold LoRA adapters into the base weights and save one full model folder.

    uv run python merge_model.py checkpoints/sft-A checkpoints/dpo-A --out checkpoints/final-A

vLLM serves one LoRA adapter on top of a base model, but the DPO adapter sits on top of the
merged SFT model. Folding both in gives vLLM (and generate.py) a single model to load. The
folder is gitignored like every checkpoint: it can regurgitate training text.
"""

import argparse
from pathlib import Path

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", type=Path, nargs="+", help="applied in order, e.g. sft then dpo")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    model = common.load_model(*args.adapters, merge_last=True)
    model.save_pretrained(args.out)
    common.load_tokenizer().save_pretrained(args.out)  # with the chat template we trained on
    print(f"merged {' + '.join(map(str, args.adapters))} -> {args.out}")


if __name__ == "__main__":
    main()
