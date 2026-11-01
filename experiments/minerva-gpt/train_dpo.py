"""DPO on top of the SFT adapter: "your voice decides, grades refine, retrieval informs".

    uv run python train_dpo.py --variant A --spin data/generations/A-train.jsonl
    uv run python train_dpo.py --variant B --rankings data/rankings.jsonl
    uv run python train_dpo.py --variant B --rankings data/rankings.jsonl \
        --auto data/generations/B-val.voice.grade.jsonl --voice-gate 0.5

Pairs, as (prompt, chosen, rejected):
- SPIN (self-play fine-tuning, 2024), from generations on train asks: what I really said beats what the
  SFT model said to the same ask. No labeling needed, and it pushes toward my actual
  reasoning rather than between near-identical samples. Answers under 15 words (often
  transcription fragments) are skipped, and so are model answers cut off at the length cap:
  the chat template still ends them with eos, so they would teach "stopping loses" and DPO
  learned to ramble (session 2, 1B: 66% of answers hit the cap). These are answers SFT already
  trained on, so check eval_memorization.py and eval_drift.py afterwards.
- gold, from my rankings (rank_server.py). My pick beats every answer I marked "not my voice"
  and every other answer I marked as my voice. Between two non-picked answers that both sound
  like me, the one the grade model scores higher wins. Voice is a hard gate: an answer I marked
  "not my voice" is never chosen, whatever its grade. Gold pairs count 3x.
- automatic (optional), from generations scored by voice_score.py and reward_model.py: among
  samples whose voice score passes `--voice-gate`, the best-graded beats the sample with the
  lowest voice score.

The SFT adapter is merged into the base first and a fresh LoRA adapter is trained on top, so
DPO's reference model (the same weights with the new adapter off) is the SFT model, i.e. me.
`--beta` is the leash: higher keeps the model closer to that SFT model. `--sft-weight` adds
an SFT loss on the chosen answer (the RPO recipe), which keeps my real answers, endings and
all, likely while DPO pushes the rejected ones down. Each DPO comparison
carries at most one bit, so this stage steers the voice the SFT stage taught; it can't
teach new things about me.
"""

import argparse
from pathlib import Path

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

import common
from reward_model import MODEL_PATH, GradeModel
from retrieve import ask_text

GOLD_REPEATS = 3
SPIN_MIN_WORDS = 15


def answer(text):
    return [{"role": "assistant", "content": text}]


def gold_pairs(rankings, grade_model=None):
    pairs = []
    for r in rankings:
        texts, voice, best = [a["text"] for a in r["answers"]], r["voice"], r.get("best")
        if best is None or not voice[best]:
            continue
        for i, text in enumerate(texts):
            if i != best:
                pairs.append((r["prompt"], texts[best], text))
        others = [i for i, ok in enumerate(voice) if ok and i != best]
        if grade_model is not None and len(others) >= 2:
            scores = grade_model.predict([ask_text(r)] * len(others), [texts[i] for i in others])
            ranked = [others[i] for i in sorted(range(len(others)), key=lambda k: -scores[k])]
            pairs.append((r["prompt"], texts[ranked[0]], texts[ranked[-1]]))
    return pairs


def auto_pairs(rows, voice_gate):
    pairs = []
    for r in rows:
        samples = list(zip(r["samples"], r["voice"], r["grade_pred"], strict=True))
        passing = [s for s in samples if s[1] >= voice_gate]
        if not passing:
            continue
        chosen = max(passing, key=lambda s: s[2])
        rejected = min(samples, key=lambda s: s[1])
        if rejected is not chosen and rejected[1] < voice_gate:
            pairs.append((r["prompt"], chosen[0], rejected[0]))
    return pairs


def finished(row, i=0):
    """Whether sample i ended on its own; older files without the flag fall back to whether it
    ends a sentence."""
    if "finished" in row:
        return row["finished"][i]
    return row["samples"][i].rstrip()[-1:] in tuple(".?!\"')")


def spin_pairs(rows, length_match=None):
    """My real answer (chosen) vs the model's first sample (rejected), per train ask.
    `length_match` (e.g. 0.3) keeps only pairs within that fraction of each other's length, so
    length can't be what separates winner from loser."""
    pairs, cut, unmatched = [], 0, 0
    for r in rows:
        real, model = r["real"], r["samples"][0]
        if len(real.split()) < SPIN_MIN_WORDS or not model.strip() or model.strip() == real.strip():
            continue
        if not finished(r):
            cut += 1
            continue
        a, b = len(real.split()), len(model.split())
        if length_match is not None and abs(a - b) > length_match * max(a, b):
            unmatched += 1
            continue
        pairs.append((r["prompt"], real, model))
    print(f"SPIN: {len(pairs)} pairs; {cut} skipped because the model's answer was cut off"
          + (f", {unmatched} because the lengths differ by more than {length_match:.0%}" if length_match else ""))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B"], required=True)
    ap.add_argument("--rankings", type=Path, help="my picks from rank_server.py")
    ap.add_argument("--spin", type=Path, help="generations on train asks, for SPIN pairs")
    ap.add_argument("--auto", type=Path, help="generations with voice and grade_pred scores")
    ap.add_argument("--voice-gate", type=float, default=0.0)
    ap.add_argument("--sft", type=Path, help="default checkpoints/sft-{variant}")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5, help="~10x a full-fine-tuning DPO rate")
    ap.add_argument("--loss", choices=["sigmoid", "sigmoid_norm"], default="sigmoid",
                    help="sigmoid_norm compares per-token log-probs, so length can't buy reward")
    ap.add_argument("--ld-alpha", type=float, help="LD-DPO: weight on tokens beyond the shared length")
    ap.add_argument("--length-match", type=float, help="keep SPIN pairs within this length ratio")
    ap.add_argument("--sft-weight", type=float, default=1.0,
                    help="weight of the SFT loss on chosen answers (RPO); 0 = plain DPO")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--grade-model", type=Path, default=MODEL_PATH)
    args = ap.parse_args()
    if not (args.rankings or args.spin):
        ap.error("give --rankings, --spin, or both")
    sft = args.sft or common.CHECKPOINTS / f"sft-{args.variant}"
    out = args.out or common.CHECKPOINTS / f"dpo-{args.variant}"

    grade_model = GradeModel.load(args.grade_model) if args.grade_model.exists() else None
    gold = gold_pairs(common.read_jsonl(args.rankings), grade_model) if args.rankings else []
    auto = auto_pairs(common.read_jsonl(args.auto), args.voice_gate) if args.auto else []
    spin = spin_pairs(common.read_jsonl(args.spin), args.length_match) if args.spin else []
    pairs = gold * GOLD_REPEATS + auto + spin
    if not pairs:
        raise SystemExit("no DPO pairs: no rankings, or every SPIN sample already equals my real "
                         "answer (the model has memorized them; see eval_memorization.py)")
    print(f"{len(gold)} gold pairs (x{GOLD_REPEATS}) + {len(auto)} automatic + {len(spin)} SPIN pairs"
          + ("" if grade_model or not gold else " (no grade model: gold pairs only use my picks)"))
    train = Dataset.from_list([{"prompt": p, "chosen": answer(c), "rejected": answer(r)}
                               for p, c, r in pairs]).shuffle(seed=0)

    config = DPOConfig(
        output_dir=str(out),
        beta=args.beta,
        loss_type=[args.loss, "sft"] if args.sft_weight else [args.loss],
        ld_alpha=args.ld_alpha,
        loss_weights=[1.0, args.sft_weight] if args.sft_weight else None,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_length=common.MAX_LENGTH,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    trainer = DPOTrainer(model=common.load_model(sft, merge_last=True), args=config,
                         train_dataset=train, processing_class=common.load_tokenizer(),
                         peft_config=common.lora_config())
    trainer.train()
    trainer.save_model(str(out))
    print(f"DPO adapter saved to {out}; load with common.load_model({sft}, {out})")


if __name__ == "__main__":
    main()
