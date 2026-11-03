"""Does the trained model use the whole class? Same model, three prompts per held-out spoken answer.

    uv run python eval_context.py --model ~/models/dpobest-B [--limit 100] [--tail-tokens 4000]

For each of my spoken answers in the validation classes (data/sft/val.jsonl, class_turn), with its
class so far from data/context/val.jsonl (written by prepare_data.py):

  window   the prompt exactly as trained: class header, readings, my PCW, the ask + replies since
  tail     the same, with the last ~4,000 tokens of the class so far inserted before the ask
  full     the same, with the whole class so far inserted before the ask

The inserted block is `Class so far:` plus the class's earlier turns as the window renders them
(`Who: text`, masked). It sits between the context and `Professor: <ask>`, so every prompt still
ends exactly as in training. If tail or full matches or beats the window, the live bot can carry
the class in context with no retraining; if they are worse, the full-class retraining is needed.

Scored on (1) loss on my real answer, paired answer by answer against the window, with a class
bootstrap; (2) one sampled answer per prompt: my habits per 1,000 words, length, cut-offs, loops,
transcript breaks (the answer tries to write another speaker's line: cut there by a stop rule,
and counted) and overlap with what I actually said. Runs on the GPU pod through vLLM (vllm_sample.py, in ~/vllm-env). Writes
data/generations/7b/context-test.json, plus one generation file per variant
(B-val-ctx-<variant>.jsonl) that rank_server.py can show blind.
"""

import argparse
import json
import os
import random
import re
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

import common
from generate import SAMPLING, prompt_ids

VARIANTS = ("window", "tail", "full")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
HABITS = {"like": r"\blike\b", "I think": r"\bi think\b",
          "So at sentence start": r"(?:^|[.?!]\s+)so\b", "yeah / yep": r"\b(yeah|yep)\b"}
LOOP = re.compile(r"\b(\w+)( \1\b){2,}")
BREAK = re.compile(r"^\s*(Professor|Student \d+|Speaker|Brady):")  # an answer that opens as a turn
# The guard the live bot uses too: an answer ends where it would start another speaker's line, so
# the bot can never say someone else's words. The test counts how often it fired.
STOPS = ["\nProfessor:", "\nStudent ", "\nSpeaker:", "\nBrady:"]
FILLER = {"like", "yeah", "yep", "um", "uh", "okay", "ok", "kinda", "gonna", "wanna", "really",
          "just", "thing", "things", "think", "know", "mean", "sort", "kind", "lot"}


def insert_history(content, block):
    """`Class so far:` + block, right before the body (`Professor: <ask>`); unchanged if empty."""
    k = content.find("\n\nProfessor: ")
    if not block or k < 0:
        return content
    return f"{content[:k]}\n\nClass so far:\n{block}{content[k:]}"


def last_tokens(tok, text, n):
    """The last n tokens of text, starting at a line break so no turn is cut mid-way."""
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) <= n:
        return text
    tail = tok.decode(ids[-n:])
    nl = tail.find("\n")
    return tail[nl + 1:] if 0 <= nl < len(tail) - 1 else tail


def run_vllm(job):
    python = os.environ.get("VLLM_PYTHON", str(Path.home() / "vllm-env/bin/python"))
    with tempfile.TemporaryDirectory() as tmp:
        job_path, out = Path(tmp) / "job.json", Path(tmp) / "out.json"
        job_path.write_text(json.dumps(job))
        subprocess.run([python, str(Path(__file__).with_name("vllm_sample.py")), str(job_path), str(out)],
                       check=True)
        return json.loads(out.read_text())


def content_words(text):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return [w for w in WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS and w not in FILLER and len(w) > 2]


def f1(a, b):
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    total = sum(ca.values()) + sum(cb.values())
    return 2 * sum((ca & cb).values()) / total if total else 0.0


def per_1000(pattern, texts):
    words = sum(len(WORD.findall(t.lower())) for t in texts)
    return 1000 * sum(len(re.findall(pattern, t, re.I)) for t in texts) / max(1, words)


def class_bootstrap(diffs, classes, reps=2000, seed=0):
    """95% interval of the mean paired difference, resampling whole classes."""
    by_class = {}
    for d, c in zip(diffs, classes):
        by_class.setdefault(c, []).append(d)
    groups, rng, means = list(by_class.values()), random.Random(seed), []
    for _ in range(reps):
        pick = [rng.choice(groups) for _ in groups]
        flat = [d for g in pick for d in g]
        means.append(sum(flat) / len(flat))
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="merged full model folder (SFT + DPO)")
    ap.add_argument("--data", type=Path, default=common.DATA)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tail-tokens", type=int, default=4000)
    ap.add_argument("--max-class-tokens", type=int, default=28000, help="cap on the full class so far")
    ap.add_argument("--n", type=int, default=1, help="sampled answers per prompt")
    ap.add_argument("--out", type=Path, default=common.DATA / "generations/7b")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model))
    if not tok.chat_template:
        tok.chat_template = common.CHAT_TEMPLATE
    rows = [r for r in common.read_jsonl(args.data / f"sft/{args.split}.jsonl") if r["source"] == "class_turn"]
    history = {h["id"]: h["history"] for h in common.read_jsonl(args.data / f"context/{args.split}.jsonl")}
    rows = rows[:args.limit]
    missing = [r for r in rows if common.prompt_key(r["prompt"]) not in history]
    if missing:
        sys.exit(f"{len(missing)} answers have no class so far: rebuild data with prepare_data.py")

    prompts = {v: [] for v in VARIANTS}
    for r in rows:
        content, h = r["prompt"][0]["content"], history[common.prompt_key(r["prompt"])]
        blocks = {"window": "", "tail": last_tokens(tok, h, args.tail_tokens),
                  "full": last_tokens(tok, h, args.max_class_tokens)}
        for v in VARIANTS:
            prompts[v].append([{"role": "user", "content": insert_history(content, blocks[v])}])
    ids = {v: [prompt_ids(tok, p) for p in prompts[v]] for v in VARIANTS}
    answers = [tok(r["completion"][0]["content"] + tok.eos_token, add_special_tokens=False).input_ids for r in rows]
    longest = max(len(p) for v in VARIANTS for p in ids[v]) + max(len(a) for a in answers)
    max_len = max(4096, min(65536, 1024 * (longest // 1024 + 2)))
    print(f"{len(rows)} spoken answers; prompt tokens (median): "
          + ", ".join(f"{v} {st.median(len(p) for p in ids[v]):,.0f}" for v in VARIANTS)
          + f"; vLLM max_model_len {max_len:,}", flush=True)

    # 1. loss on what I actually said, under each prompt
    items = [{"prompt": p, "answer": a} for v in VARIANTS for p, a in zip(ids[v], answers)]
    means = run_vllm({"adapter": str(args.model.resolve()), "score": items, "max_model_len": max_len})
    loss = {v: [-m for m in means[i * len(rows):(i + 1) * len(rows)]] for i, v in enumerate(VARIANTS)}

    # 2. one sampled answer per prompt
    flat = [p for v in VARIANTS for p in ids[v]]
    out = run_vllm({"adapter": str(args.model.resolve()), "prompts": flat, "n": args.n,
                    "temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"],
                    "max_new_tokens": SAMPLING["max_new_tokens"], "eos_token_id": tok.eos_token_id,
                    "max_model_len": max_len, "stop": STOPS})
    samples = {v: out[i * len(rows):(i + 1) * len(rows)] for i, v in enumerate(VARIANTS)}

    real = [r["completion"][0]["content"] for r in rows]
    classes = [r["class_id"] for r in rows]
    rng = random.Random(0)
    other = [(i + rng.randrange(1, len(rows))) % len(rows) for i in range(len(rows))] if len(rows) > 1 else [0]
    sem = None
    try:
        from sentence_transformers import SentenceTransformer

        sem = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    except Exception as e:  # noqa: BLE001 - the embedder is optional
        print(f"(meaning similarity skipped: {e.__class__.__name__})")
    real_emb = sem.encode(real, normalize_embeddings=True) if sem else None

    report = {"answers": len(rows), "tail_tokens": args.tail_tokens, "variants": {}}
    for v in VARIANTS:
        texts = [a[0] for per in samples[v] for a in per]
        own = [i for i, per in enumerate(samples[v]) for _ in per]
        d = [a - b for a, b in zip(loss[v], loss["window"])]
        res = {"loss": st.mean(loss[v]), "prompt_tokens_median": st.median(len(p) for p in ids[v])}
        if v != "window":
            lo, hi = class_bootstrap(d, classes)
            res |= {"loss_vs_window": st.mean(d), "ci95": [lo, hi],
                    "better_on_share": sum(x < 0 for x in d) / len(d)}
        res["habits_per_1000"] = {h: per_1000(p, texts) for h, p in HABITS.items()}
        res["length_vs_mine"] = st.median(len(t.split()) for t in texts) / st.median(len(t.split()) for t in real)
        res["cut_off_share"] = sum(not a[1] for per in samples[v] for a in per) / len(texts)
        res["loops"] = sum(bool(LOOP.search(t.lower())) for t in texts)
        res["transcript_breaks"] = (sum(a[2] for per in samples[v] for a in per)
                                    + sum(bool(BREAK.match(t)) for t in texts))
        res["content_overlap"] = st.mean(f1(content_words(t), content_words(real[i])) for t, i in zip(texts, own))
        res["content_overlap_other_answer"] = st.mean(f1(content_words(t), content_words(real[other[i]]))
                                                      for t, i in zip(texts, own))
        if sem:
            e = sem.encode(texts, normalize_embeddings=True)
            same = [float(e[j] @ real_emb[i]) for j, i in enumerate(own)]
            diff = [float(e[j] @ real_emb[other[i]]) for j, i in enumerate(own)]
            res["meaning_closer_to_mine"] = sum(a > b for a, b in zip(same, diff)) / len(same)
        report["variants"][v] = res
        common.write_jsonl(args.out / f"B-val-ctx-{v}.jsonl", [
            {"id": common.prompt_key(r["prompt"]), "class_id": r["class_id"], "source": r["source"],
             "los": r["los"], "prompt": r["prompt"], "real": real[i], "variant": "B",
             "samples": [a[0] for a in samples[v][i]], "finished": [a[1] for a in samples[v][i]]}
            for i, r in enumerate(rows)])
    report["habits_mine_per_1000"] = {h: per_1000(p, real) for h, p in HABITS.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "context-test.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'':36s}" + "".join(f"{v:>15s}" for v in VARIANTS) + f"{'me':>10s}")
    V = report["variants"]
    line = lambda name, f, mine="": print(f"{name:36s}" + "".join(f"{f(V[v]):>15s}" for v in VARIANTS) + f"{mine:>10s}")
    line("prompt tokens (median)", lambda r: f"{r['prompt_tokens_median']:,.0f}")
    line("loss on my answer (lower=better)", lambda r: f"{r['loss']:.4f}")
    line("  vs window", lambda r: f"{r['loss_vs_window']:+.4f}" if "loss_vs_window" in r else "")
    line("  95% interval (class bootstrap)", lambda r: f"{r['ci95'][0]:+.3f}/{r['ci95'][1]:+.3f}" if "ci95" in r else "")
    line("  better than window on", lambda r: f"{100 * r['better_on_share']:.0f}%" if "better_on_share" in r else "")
    for h in HABITS:
        line(f"'{h}' per 1,000", lambda r, h=h: f"{r['habits_per_1000'][h]:.1f}",
             f"{report['habits_mine_per_1000'][h]:.1f}")
    line("length vs mine (median)", lambda r: f"{r['length_vs_mine']:.2f}x")
    line("cut off at the length limit", lambda r: f"{100 * r['cut_off_share']:.0f}%")
    line("answers stuck in a loop", lambda r: f"{r['loops']}")
    line("tried to write another speaker", lambda r: f"{r['transcript_breaks']}")
    line("content overlap with mine", lambda r: f"{100 * r['content_overlap']:.1f}%")
    line("  with an unrelated answer", lambda r: f"{100 * r['content_overlap_other_answer']:.1f}%")
    if sem:
        line("meaning closer to mine than other", lambda r: f"{100 * r['meaning_closer_to_mine']:.0f}%")
    for v in ("tail", "full"):
        lo, hi = V[v]["ci95"]
        verdict = "better than" if hi < 0 else "worse than" if lo > 0 else "no clear difference from"
        print(f"\n{v}: loss {verdict} the window ({V[v]['loss_vs_window']:+.4f}, 95% {lo:+.4f} to {hi:+.4f})")
    print(f"\nwrote {args.out / 'context-test.json'} and B-val-ctx-<variant>.jsonl")


if __name__ == "__main__":
    main()
