"""Voice score: how much an answer sounds like me, as a log-probability contrast.

    uv run python voice_score.py --adapter checkpoints/sft-B --variant B --validate val
    uv run python voice_score.py --adapter checkpoints/sft-B --variant B --score data/generations/B-val.jsonl

Variant B was trained with speaker tags, so for any answer we can ask how likely it is after
"Answer as: Brady" versus after "Answer as: a classmate". The difference is the log-odds that
the answer came from me rather than from a typical classmate; fluency, length and topic appear
in both terms and cancel, so a bland but probable answer scores near zero. Not "highest
log-prob wins": that would favour generic, short text.

Variant A has no classmate tag and must stay free of classmate data, so its score is the same
answer under the A adapter minus under the plain base model: how much more likely training on
me made it. It is noisier, since it also rewards anything else SFT changed.

Scores are per-token means. `--validate SPLIT` checks the score before it is trusted: on real
held-out answers, mine should outscore classmates' (AUC >= 0.8). Classmates' answers are only
read there, as negatives; nothing is trained on them.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import common

AUC_GATE = 0.8


def with_tag(prompt, tag):
    first, _, rest = prompt[0]["content"].partition("\n")
    assert first in (common.TAG_ME, common.TAG_CLASSMATE), "prompt has no speaker tag"
    return [{"role": "user", "content": f"{tag}\n{rest}"}]


def mean_logprobs(model, tok, pairs, batch=4):
    """Mean log p(answer tokens | prompt) for each (prompt messages, answer) pair."""
    out = []
    for i in range(0, len(pairs), batch):
        seqs, starts = [], []
        for prompt, answer in pairs[i:i + batch]:
            p = tok(tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False).input_ids
            a = tok(answer + tok.eos_token, add_special_tokens=False).input_ids
            seqs.append((p + a)[:common.MAX_LENGTH])
            starts.append(min(len(p), common.MAX_LENGTH - 1))
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), tok.pad_token_id)
        mask = torch.zeros_like(ids)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        ids, mask = ids.to(model.device), mask.to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        for j, s in enumerate(seqs):
            span = logits[j, starts[j] - 1:len(s) - 1].float().log_softmax(-1)
            target = ids[j, starts[j]:len(s)]
            out.append(span.gather(-1, target[:, None]).mean().item())
    return out


def voice_scores(model, tok, variant, pairs):
    """Voice score for each (prompt, answer); higher = more like me."""
    if variant == "B":
        me = mean_logprobs(model, tok, [(with_tag(p, common.TAG_ME), a) for p, a in pairs])
        them = mean_logprobs(model, tok, [(with_tag(p, common.TAG_CLASSMATE), a) for p, a in pairs])
        return [m - t for m, t in zip(me, them, strict=True)]
    trained = mean_logprobs(model, tok, pairs)
    with model.disable_adapter():
        base = mean_logprobs(model, tok, pairs)
    return [t - b for t, b in zip(trained, base, strict=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--variant", choices=["A", "B"], required=True)
    ap.add_argument("--data", type=Path, default=common.DATA)
    ap.add_argument("--validate", metavar="SPLIT")
    ap.add_argument("--score", type=Path, help="a generations file; writes <name>.voice.jsonl")
    ap.add_argument("--n", type=int, default=300, help="answers per side when validating")
    args = ap.parse_args()
    tok, model = common.load_tokenizer(), common.load_model(args.adapter)
    model.eval()

    if args.validate:
        rng = random.Random(0)
        mine = [r for r in common.read_jsonl(args.data / f"sft/{args.validate}.jsonl")
                if r["source"] == "class_turn"]
        theirs = common.read_jsonl(args.data / f"sft_classmates/{args.validate}.jsonl")
        mine, theirs = (rng.sample(x, min(args.n, len(x))) for x in (mine, theirs))
        pairs = [(r["prompt"], r["completion"][0]["content"]) for r in mine + theirs]
        scores = voice_scores(model, tok, args.variant, pairs)
        auc = roc_auc_score([1] * len(mine) + [0] * len(theirs), scores)
        verdict = "PASS" if auc >= AUC_GATE else "FAIL: rely on Brady's own picks"
        print(f"voice score AUC on real {args.validate} answers ({len(mine)} mine vs "
              f"{len(theirs)} classmates'): {auc:.3f}  {verdict}")
        # The score has an offset (the tag seen more often in training lifts every answer a
        # little), so the automatic gate comes from my own answers, not from 0: the score
        # that 90% of what I really said clears.
        gate = float(np.percentile(scores[:len(mine)], 10))
        print(f"suggested --voice-gate for train_dpo.py: {gate:.4f} "
              f"(10th percentile of my real answers' scores)")

    if args.score:
        rows = common.read_jsonl(args.score)
        pairs = [(r["prompt"], s) for r in rows for s in r["samples"]]
        scores = iter(voice_scores(model, tok, args.variant, pairs))
        for r in rows:
            r["voice"] = [round(next(scores), 4) for _ in r["samples"]]
        out = args.score.with_suffix(".voice.jsonl")
        common.write_jsonl(out, rows)
        print(f"voice scores for {len(pairs)} answers -> {out}")


if __name__ == "__main__":
    main()
