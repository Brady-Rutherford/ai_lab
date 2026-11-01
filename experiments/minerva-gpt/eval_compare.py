"""Head-to-head on the same footing: how well each model predicts MY held-out answers.

    MINERVA_BASE_MODEL=allenai/Olmo-3-1025-7B uv run python eval_compare.py \
        checkpoints/7b/sft-A checkpoints/7b/sft-B [checkpoints/7b/final-dpobest-A ...]

The val loss train_sft.py logs is not comparable across variants: B's val set includes
classmates' answers. This scores every model on the same rows, data/sft/val.jsonl (my answers,
prompts tagged "Answer as: Brady"): per-token loss on each answer, averaged over answers. Lower
= it predicts what I actually said better. LoRA adapters and merged models both work, and so
does an MLX copy (quantized, for the Mac), scored with the same tokens and the same arithmetic.
"""

import argparse
import json
from pathlib import Path

import common


def is_mlx(path):
    """A quantized MLX folder (mlx_lm.convert -q) rather than a transformers model."""
    config = Path(path) / "config.json"
    return config.exists() and "quantization" in json.loads(config.read_text())


def mlx_logprobs(path, pairs):
    """voice_score.mean_logprobs for an MLX model, one answer at a time (no padding)."""
    import mlx.core as mx
    from mlx_lm import load

    model, wrapper = load(str(path))
    tok = wrapper._tokenizer
    out = []
    for prompt, answer in pairs:
        p = tok(tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True),
                add_special_tokens=False).input_ids
        a = tok(answer + tok.eos_token, add_special_tokens=False).input_ids
        seq = (p + a)[:common.MAX_LENGTH]
        start = min(len(p), common.MAX_LENGTH - 1)
        logits = model(mx.array(seq)[None])[0, start - 1:len(seq) - 1].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        out.append(mx.take_along_axis(logp, mx.array(seq[start:])[:, None], axis=-1).mean().item())
    return out


def torch_logprobs(path, pairs):
    import torch

    from voice_score import mean_logprobs

    model = (common.load_model(name=str(path)) if common.is_merged(path) else common.load_model(path))
    model.eval()
    lp = mean_logprobs(model, common.load_tokenizer(), pairs, batch=8)
    del model
    torch.cuda.empty_cache()
    return lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--save", type=Path, help="append each model's per-answer losses here (jsonl)")
    args = ap.parse_args()
    rows = common.read_jsonl(common.DATA / "sft/val.jsonl")[:args.limit]
    pairs = [(r["prompt"], r["completion"][0]["content"]) for r in rows]
    for path in map(Path, args.models):
        lp = (mlx_logprobs if is_mlx(path) else torch_logprobs)(path, pairs)
        print(f"{str(path):40s} loss on my {len(pairs)} held-out answers: {-sum(lp) / len(lp):.4f}")
        if args.save:
            with args.save.open("a") as f:
                f.write(json.dumps({"model": str(path), "losses": [-x for x in lp]}) + "\n")


if __name__ == "__main__":
    main()
