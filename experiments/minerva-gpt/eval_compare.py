"""Head-to-head on the same footing: how well each model predicts MY held-out answers.

    MINERVA_BASE_MODEL=allenai/Olmo-3-1025-7B uv run python eval_compare.py \
        checkpoints/7b/sft-A checkpoints/7b/sft-B [checkpoints/7b/final-dpobest-A ...]

The val loss train_sft.py logs is not comparable across variants: B's val set includes
classmates' answers. This scores every model on the same rows, data/sft/val.jsonl (my answers,
prompts tagged "Answer as: Brady"): per-token loss on each answer, averaged over answers. Lower
= it predicts what I actually said better. LoRA adapters and merged models both work.
"""

import argparse
from pathlib import Path

import torch

import common
from voice_score import mean_logprobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    models = args.models
    tok = common.load_tokenizer()
    rows = common.read_jsonl(common.DATA / "sft/val.jsonl")[:args.limit]
    pairs = [(r["prompt"], r["completion"][0]["content"]) for r in rows]
    for path in map(Path, models):
        model = (common.load_model(name=str(path)) if common.is_merged(path) else common.load_model(path))
        model.eval()
        lp = mean_logprobs(model, tok, pairs, batch=8)
        print(f"{str(path):40s} loss on my {len(pairs)} held-out answers: {-sum(lp) / len(lp):.4f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
