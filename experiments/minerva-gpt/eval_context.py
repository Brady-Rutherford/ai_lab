"""Does the trained model use the whole class? Same model, four prompts per held-out spoken answer.

    uv run python eval_context.py --model ~/models/dpobest-B --crosscheck 20   # first: vLLM right?
    uv run python eval_context.py --model ~/models/dpobest-B [--limit 30] [--tail-tokens 2500]

For each of my spoken answers in the validation classes (data/sft/val.jsonl, class_turn), with its
class so far from data/context/val.jsonl (written by prepare_data.py):

  window    the prompt exactly as trained: class header, readings, my PCW, the ask + replies since
  tail      the same, with the last ~2,500 tokens of the class so far inserted before the ask
  full      the same, with the whole class so far inserted before the ask
  full-top  the whole class so far at the very top, then the trained prompt untouched

Where the class goes matters. 24 of OLMo 3 7B's 32 layers see only the last 4,096 tokens, and the
prompt's first line is the voice tag (`Answer as: Brady`). In `full`, most prompts push the tag and
my PCW out of those layers' reach. At 2,500 tokens `tail` fits inside the window, and `full-top`
keeps the whole trained prompt inside it, so length and placement can be told apart. The block is
`Class so far:` plus the class's earlier turns as the window renders them (`Who: text`, masked).
Every prompt still ends exactly as in training.

Scored on (1) loss on my real answer, paired answer by answer against the window, with a class
bootstrap, overall and by prompt length; (2) one sampled answer per prompt: my habits, length,
cut-offs, loops, transcript breaks (the answer tries to write another speaker's line: cut there by a
stop rule, and counted) and overlap with what I actually said. Each variant gets a verdict: see
verdict() and NEXT_SESSION.md. Runs on the GPU pod through vLLM (vllm_sample.py, in ~/vllm-env).

--crosscheck scores a spread of prompts through vLLM and through plain Transformers on the same GPU,
judged separately on the long ones, since a fault there is what would bend this test. It also compares
greedy answers on long prompts, with vLLM's sampling settings and a second pass from its prefix
cache. A control with the 4,096 limit off shows how far off a vLLM that ignored the limit would be.
It exits with 3 if vLLM disagrees, before the long run is paid for.

Writes to data/generations/7b: context-test.json; context-test-answers.jsonl (every answer's loss
under each prompt, saved as soon as scoring ends); context-scores-<key>.json/.partial (reused by a
rerun of the same job, so a crash costs only the unfinished part); context-crosscheck.json; and one
generation file per variant (B-val-ctx-<variant>.jsonl) that rank_server.py can show blind.
"""

import argparse
import hashlib
import json
import math
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

VARIANTS = ("window", "tail", "full", "full-top")
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
# Bins by whole `full` prompt + answer: inside the sliding window; past it but inside the 8,192 the
# base model was pretrained on; past that, where only the long-context stage (8 layers) applies.
LENGTH_BINS = (("up to 4,096", 0, 4096), ("4,097-8,192", 4096, 8192), ("over 8,192", 8192, math.inf))
# The decision rule. Set from the earlier runs: B vs A differ by 0.029 in loss, so half of that; two
# runs of the same model differ by about 6% in length (runs that shared one seed, so probably less
# now); on the trained prompt, none of 1,236 sampled answers tried to write another speaker.
LOSS_MARGIN = 0.015
LENGTH_BAND = 0.10  # log ratio, about 10%
SPEAKER_MARGIN = 0.02  # share of answers
# vLLM vs Transformers, per group of prompts. For scale: the 8-bit MLX copy and the full model differ
# by 0.003 per answer on average, and by 0.0001 once signed.
CROSSCHECK_LIMIT = 0.01  # mean absolute difference in loss
CROSSCHECK_BIAS = 0.005  # mean signed difference: a one-way shift is what would move the paired result
GREEDY_PROMPTS, GREEDY_TOKENS, GREEDY_AGREE = 6, 8, 2 / 3


def insert_history(content, block, top=False):
    """`Class so far:` + block, before the body (`Professor: <ask>`) or, with `top`, before
    everything. Unchanged if there is no class so far."""
    if not block:
        return content
    if top:
        return f"Class so far:\n{block}\n\n{content}"
    k = content.find("\n\nProfessor: ")
    return content if k < 0 else f"{content[:k]}\n\nClass so far:\n{block}{content[k:]}"


def last_tokens(tok, text, n):
    """The last n tokens of text, starting at a line break so no turn is cut mid-way."""
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) <= n:
        return text
    tail = tok.decode(ids[-n:])
    nl = tail.find("\n")
    return tail[nl + 1:] if 0 <= nl < len(tail) - 1 else tail


def build_prompts(tok, rows, history, tail_tokens, max_class_tokens):
    prompts = {v: [] for v in VARIANTS}
    for r in rows:
        content, h = r["prompt"][0]["content"], history[common.prompt_key(r["prompt"])]
        whole = last_tokens(tok, h, max_class_tokens)
        texts = {"window": content, "tail": insert_history(content, last_tokens(tok, h, tail_tokens)),
                 "full": insert_history(content, whole), "full-top": insert_history(content, whole, top=True)}
        for v in VARIANTS:
            prompts[v].append([{"role": "user", "content": texts[v]}])
    return prompts


def run_vllm(job):
    python = os.environ.get("VLLM_PYTHON", str(Path.home() / "vllm-env/bin/python"))
    with tempfile.TemporaryDirectory() as tmp:
        job_path, out = Path(tmp) / "job.json", Path(tmp) / "out.json"
        job_path.write_text(json.dumps(job))
        subprocess.run([python, str(Path(__file__).with_name("vllm_sample.py")), str(job_path), str(out)],
                       check=True)
        return json.loads(out.read_text())


def score_all(args, ids, answers, max_len):
    """Mean log-prob of each answer under each variant, in VARIANTS order. Longest prompts first, so
    running out of memory shows in the first minutes; saved, so a rerun skips what is done."""
    items = [{"prompt": p, "answer": a} for v in VARIANTS for p, a in zip(ids[v], answers)]
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["prompt"]))
    job_items = [items[i] for i in order]
    key = hashlib.sha1(json.dumps([str(args.model.resolve()), args.attention_backend, job_items]).encode()).hexdigest()[:16]
    done = args.out / f"context-scores-{key}.json"
    if done.exists():
        print(f"scores from an earlier run of this exact job: {done.name}", flush=True)
        ordered = json.loads(done.read_text())
    else:
        ordered = run_vllm({"adapter": str(args.model.resolve()), "score": job_items, "max_model_len": max_len,
                            "progress": str(args.out / f"context-scores-{key}.partial"),
                            "attention_backend": args.attention_backend})
        if not all(map(math.isfinite, ordered)):
            sys.exit(f"{sum(not math.isfinite(x) for x in ordered)} answers scored NaN or infinite: "
                     "these numbers can't be used. Nothing was saved as finished.")
        tmp = done.with_suffix(".tmp")
        tmp.write_text(json.dumps(ordered))
        tmp.replace(done)
    means = [0.0] * len(items)
    for position, i in enumerate(order):
        means[i] = ordered[position]
    return means


def hf_means(model, items):
    """The same number vllm_sample.py computes, through plain Transformers, one sequence at a time."""
    import torch

    out = []
    for it in items:
        seq = torch.tensor([it["prompt"] + it["answer"]], device=model.device)
        with torch.no_grad():
            logits = model(input_ids=seq, logits_to_keep=len(it["answer"]) + 1, use_cache=False).logits[0, :-1]
        target = seq[0, -len(it["answer"]):]
        out.append(logits.float().log_softmax(-1).gather(-1, target[:, None]).mean().item())
    return out


def crosscheck_items(ids, answers, n):
    """A spread over prompt length: mostly the long variants, and always the longest prompt."""
    by_length = sorted(range(len(answers)), key=lambda i: len(ids["full"][i]))
    n_long = max(2, n * 4 // 5)
    picks = [by_length[round(k * (len(by_length) - 1) / (n_long - 1))] for k in range(n_long)]
    labelled = [("full-top" if k % 2 else "full", i) for k, i in enumerate(picks)]
    labelled += [("tail" if k % 2 else "window", picks[k * len(picks) // max(1, n - n_long)])
                 for k in range(n - n_long)]
    return [{"variant": v, "prompt": ids[v][i], "answer": answers[i]} for v, i in labelled]


def same_answer(a, b):
    """Greedy answers match as far as both go (the two engines may stop at different end tokens)."""
    k = min(len(a), len(b))
    return k > 0 and a[:k] == b[:k]


def crosscheck(args, tok, ids, answers, max_len):
    """0 if vLLM matches plain Transformers, 3 if not."""
    import torch

    items = crosscheck_items(ids, answers, args.crosscheck)
    size = lambda it: len(it["prompt"]) + len(it["answer"])  # noqa: E731
    window = json.loads((args.model / "config.json").read_text())["sliding_window"]
    long_items = [k for k, it in enumerate(items) if size(it) > window]
    step = max(1, len(long_items) // GREEDY_PROMPTS)
    greedy_items = [items[k] for k in long_items[::-step][:GREEDY_PROMPTS]]
    job = {"adapter": str(args.model.resolve()), "max_model_len": max_len, "attention_backend": args.attention_backend}
    vllm = run_vllm({**job, "score": [{"prompt": it["prompt"], "answer": it["answer"]} for it in items]})
    passes = run_vllm({**job, "prompts": [it["prompt"] for it in greedy_items], "n": 1, "temperature": 0.0, "top_p": 1.0,
                       "max_new_tokens": GREEDY_TOKENS, "eos_token_id": tok.eos_token_id, "greedy_passes": 2})

    model = common.load_model(name=str(args.model), attn="sdpa").eval()
    hf = hf_means(model, items)
    hf_greedy = [model.generate(torch.tensor([it["prompt"]], device=model.device), max_new_tokens=GREEDY_TOKENS,
                                do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
                                )[0, len(it["prompt"]):].tolist() for it in greedy_items]
    model.config.sliding_window = 10 ** 9  # the control: every layer sees everything
    no_window = hf_means(model, [items[k] for k in long_items[::2]])
    del model
    torch.cuda.empty_cache()

    print(f"\n{'variant':10s}{'tokens':>8s}{'vLLM':>10s}{'plain':>10s}{'diff':>10s}")
    for it, a, b in zip(items, vllm, hf):
        print(f"{it['variant']:10s}{size(it):8,d}{-a:10.4f}{-b:10.4f}{a - b:+10.4f}")
    groups = {"all": range(len(items)), f"over {window:,}": long_items,
              f"over {2 * window:,}": [k for k in long_items if size(items[k]) > 2 * window]}
    res = {"attention_backend": args.attention_backend or "auto", "groups": {}, "limit": CROSSCHECK_LIMIT, "bias_limit": CROSSCHECK_BIAS}
    passed = True
    print()
    for name, members in groups.items():
        if not members:
            continue
        d = [vllm[k] - hf[k] for k in members]
        g = {"items": len(d), "mean_abs_diff": st.mean(abs(x) for x in d), "mean_diff": st.mean(d), "max_abs_diff": max(abs(x) for x in d)}
        g["passed"] = g["mean_abs_diff"] <= CROSSCHECK_LIMIT and abs(g["mean_diff"]) <= CROSSCHECK_BIAS
        passed = passed and g["passed"]
        res["groups"][name] = g
        print(f"{name:12s} {g['items']:3d} prompts: mean difference {g['mean_abs_diff']:.4f} (limit {CROSSCHECK_LIMIT}), signed "
              f"{g['mean_diff']:+.4f} (limit {CROSSCHECK_BIAS}), largest {g['max_abs_diff']:.4f}{'' if g['passed'] else '  !! too far apart'}")
    if greedy_items:
        shares = [sum(map(same_answer, answers_, hf_greedy)) / len(greedy_items) for answers_ in passes]
        res["greedy_agreement"] = {"first_pass": shares[0], "from_prefix_cache": shares[1], "prompts": len(greedy_items)}
        passed = passed and min(shares) >= GREEDY_AGREE
        print(f"greedy answers on {len(greedy_items)} long prompts match: {100 * shares[0]:.0f}% on the first pass, "
              f"{100 * shares[1]:.0f}% from the prefix cache (need {100 * GREEDY_AGREE:.0f}%)")
    if no_window:
        control = st.mean(abs(hf[k] - x) for k, x in zip(long_items[::2], no_window))
        res["control_mean_abs_diff"] = control
        print(f"with the 4,096 limit off, plain Transformers moves by {control:.4f}"
              + ("" if control > CROSSCHECK_LIMIT else ": the limit barely changes these numbers, so a vLLM ignoring it wouldn't either"))
    res["passed"] = passed
    (args.out / "context-crosscheck.json").write_text(json.dumps(res, indent=2))
    if not passed:
        print("!! vLLM and plain Transformers disagree: the test's numbers can't be trusted on this setup")
        return 3
    print("crosscheck passed")
    return 0


def content_words(text):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return [w for w in WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS and w not in FILLER and len(w) > 2]


def f1(a, b):
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    total = sum(ca.values()) + sum(cb.values())
    return 2 * sum((ca & cb).values()) / total if total else 0.0


def n_words(text):
    return len(WORD.findall(text.lower()))


def per_1000(pattern, texts):
    words = sum(n_words(t) for t in texts)
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


def habit_shift(pattern, mine, texts, base, classes, reps=2000, seed=0):
    """How much further from my rate this variant is than the window, per 1,000 words (positive =
    further), with a 95% interval resampling whole classes. `texts`, `base`: answers per prompt."""
    per_class = {}
    for answer, control, c in zip(texts, base, classes):
        s = per_class.setdefault(c, [0, 0, 0, 0])
        for t in answer:
            s[0] += len(re.findall(pattern, t, re.I))
            s[1] += n_words(t)
        for t in control:
            s[2] += len(re.findall(pattern, t, re.I))
            s[3] += n_words(t)

    def shift(groups):
        hits, words, base_hits, base_words = (sum(col) for col in zip(*groups))
        return abs(1000 * hits / max(1, words) - mine) - abs(1000 * base_hits / max(1, base_words) - mine)

    groups, rng = list(per_class.values()), random.Random(seed)
    draws = sorted(shift([rng.choice(groups) for _ in groups]) for _ in range(reps))
    return shift(groups), draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1]


def verdict(res, margin=LOSS_MARGIN):
    """PASS, FAIL or UNSURE for one variant against the window. The interval decides, so the chance
    differences between two runs of one model don't: see NEXT_SESSION.md."""
    lo, hi = res["ci95"]
    flagged = [h for h, s in res["habit_shift"].items() if s["ci95"][0] > 0]
    fails, doubts = [], []
    if lo > margin:
        fails.append(f"loss worse than the limit of +{margin}")
    elif hi >= margin:
        doubts.append(f"loss interval crosses the limit of +{margin}")
    length_lo, length_hi = res["length_vs_window_ci95"]
    if length_lo > LENGTH_BAND or length_hi < -LENGTH_BAND:
        fails.append("answer length moved more than about 10%")
    elif length_lo < -LENGTH_BAND or length_hi > LENGTH_BAND:
        doubts.append("answer length may have moved more than about 10%")
    speaker_lo, speaker_hi = res["speaker_vs_window_ci95"]
    if speaker_lo > SPEAKER_MARGIN:
        fails.append("tries to write other speakers more often")
    elif speaker_hi > SPEAKER_MARGIN:
        doubts.append("may try to write other speakers more often")
    if len(flagged) >= 2:
        fails.append("further from my habits: " + ", ".join(flagged))
    elif flagged:
        doubts.append("further from my habits: " + flagged[0])
    return {"verdict": "FAIL" if fails else "UNSURE" if doubts else "PASS", "why": fails + doubts}


def baseline_check(args, positions, window_loss):
    """The window's losses against an earlier plain Transformers run of the same model and answers:
    a second check on vLLM that costs nothing. None if there is no such run here."""
    if args.split != "val" or not args.baseline.exists():
        return None
    runs = [p["losses"] for p in common.read_jsonl(args.baseline) if p["model"].endswith(args.baseline_model)]
    if not runs:
        return None
    gaps = [window_loss[k] - runs[0][i] for k, i in enumerate(positions)]
    return {"mean_abs_diff": st.mean(abs(g) for g in gaps), "mean_diff": st.mean(gaps),
            "max_abs_diff": max(abs(g) for g in gaps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="merged full model folder (SFT + DPO)")
    ap.add_argument("--data", type=Path, default=common.DATA)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tail-tokens", type=int, default=2500)
    ap.add_argument("--max-class-tokens", type=int, default=28000, help="cap on the full class so far")
    ap.add_argument("--n", type=int, default=1, help="sampled answers per prompt")
    ap.add_argument("--out", type=Path, default=common.DATA / "generations/7b")
    ap.add_argument("--crosscheck", type=int, help="compare vLLM with Transformers on this many prompts, then stop")
    ap.add_argument("--attention-backend", help="vLLM attention backend, e.g. TRITON_ATTN (default: vLLM's choice)")
    ap.add_argument("--baseline", type=Path, default=common.ROOT / "runs/perrow-7b.jsonl",
                    help="eval_compare.py --save output for the same answers")
    ap.add_argument("--baseline-model", default="final-dpobest-B")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model))
    if not tok.chat_template:
        tok.chat_template = common.CHAT_TEMPLATE
    all_rows = common.read_jsonl(args.data / f"sft/{args.split}.jsonl")
    positions = [i for i, r in enumerate(all_rows) if r["source"] == "class_turn"][:args.limit]
    rows = [all_rows[i] for i in positions]
    history = {h["id"]: h["history"] for h in common.read_jsonl(args.data / f"context/{args.split}.jsonl")}
    missing = [r for r in rows if common.prompt_key(r["prompt"]) not in history]
    if missing:
        sys.exit(f"{len(missing)} answers have no class so far: rebuild data with prepare_data.py")

    prompts = build_prompts(tok, rows, history, args.tail_tokens, args.max_class_tokens)
    ids = {v: [prompt_ids(tok, p) for p in prompts[v]] for v in VARIANTS}
    answers = [tok(r["completion"][0]["content"] + tok.eos_token, add_special_tokens=False).input_ids for r in rows]
    longest = max(len(p) for v in VARIANTS for p in ids[v]) + max(len(a) for a in answers)
    max_len = max(4096, min(65536, 1024 * (longest // 1024 + 2)))
    print(f"{len(rows)} spoken answers; prompt tokens (median): "
          + ", ".join(f"{v} {st.median(len(p) for p in ids[v]):,.0f}" for v in VARIANTS)
          + f"; vLLM max_model_len {max_len:,}", flush=True)

    if args.crosscheck is not None:
        if args.crosscheck < 5:
            sys.exit("--crosscheck needs at least 5 prompts")
        sys.exit(crosscheck(args, tok, ids, answers, max_len))

    # 1. loss on what I actually said, under each prompt
    means = score_all(args, ids, answers, max_len)
    loss = {v: [-m for m in means[i * len(rows):(i + 1) * len(rows)]] for i, v in enumerate(VARIANTS)}
    classes = [r["class_id"] for r in rows]
    total = [len(p) + len(a) for p, a in zip(ids["full"], answers)]
    common.write_jsonl(args.out / "context-test-answers.jsonl", [
        {"id": common.prompt_key(r["prompt"]), "class_id": r["class_id"], "answer_tokens": len(answers[i]),
         "prompt_tokens": {v: len(ids[v][i]) for v in VARIANTS}, "loss": {v: loss[v][i] for v in VARIANTS}}
        for i, r in enumerate(rows)])

    report = {"answers": len(rows), "classes": len(set(classes)), "tail_tokens": args.tail_tokens,
              "attention_backend": args.attention_backend or "auto", "loss_margin": LOSS_MARGIN,
              "baseline": baseline_check(args, positions, loss["window"]), "by_length": {}, "variants": {}}
    for v in VARIANTS:
        d = [a - b for a, b in zip(loss[v], loss["window"])]
        res = {"loss": st.mean(loss[v]), "prompt_tokens_median": st.median(len(p) for p in ids[v]),
               "fits_sliding_window": sum(len(p) + len(a) <= 4096 for p, a in zip(ids[v], answers)) / len(rows)}
        if v != "window":
            lo, hi = class_bootstrap(d, classes)
            res |= {"loss_vs_window": st.mean(d), "ci95": [lo, hi],
                    "better_on_share": sum(x < 0 for x in d) / max(1, sum(x != 0 for x in d))}
        report["variants"][v] = res
    for name, low, high in LENGTH_BINS:
        members = [i for i, t in enumerate(total) if low < t <= high]
        if not members:
            continue
        report["by_length"][name] = {"answers": len(members), "classes": len({classes[i] for i in members}), "variants": {}}
        for v in VARIANTS[1:]:
            d = [loss[v][i] - loss["window"][i] for i in members]
            report["by_length"][name]["variants"][v] = {
                "loss_vs_window": st.mean(d), "ci95": list(class_bootstrap(d, [classes[i] for i in members]))}
    if report["baseline"]:
        b = report["baseline"]
        print(f"\nwindow vs an earlier plain Transformers run, answer by answer: mean difference "
              f"{b['mean_abs_diff']:.4f}, largest {b['max_abs_diff']:.4f}"
              + ("" if b["mean_abs_diff"] <= CROSSCHECK_LIMIT else "  !! larger than expected: check before trusting this"))
    print_loss(report)
    (args.out / "context-test.json").write_text(json.dumps(report, indent=2))  # the main result is safe now

    # 2. one sampled answer per prompt
    flat = [p for v in VARIANTS for p in ids[v]]
    out = run_vllm({"adapter": str(args.model.resolve()), "prompts": flat, "n": args.n,
                    "temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"],
                    "max_new_tokens": SAMPLING["max_new_tokens"], "eos_token_id": tok.eos_token_id,
                    "max_model_len": max_len, "stop": STOPS, "attention_backend": args.attention_backend,
                    # one seed per answer, the same under each prompt: independent across answers, paired across prompts
                    "seeds": [i * args.n for _ in VARIANTS for i in range(len(rows))]})
    samples = {v: out[i * len(rows):(i + 1) * len(rows)] for i, v in enumerate(VARIANTS)}

    real = [r["completion"][0]["content"] for r in rows]
    rng = random.Random(0)
    other = [(i + rng.randrange(1, len(rows))) % len(rows) for i in range(len(rows))] if len(rows) > 1 else [0]
    sem = None
    try:
        from sentence_transformers import SentenceTransformer

        sem = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    except Exception as e:  # noqa: BLE001 - the embedder is optional
        print(f"(meaning similarity skipped: {e.__class__.__name__})")
    real_emb = sem.encode(real, normalize_embeddings=True) if sem else None

    mine = {h: per_1000(p, real) for h, p in HABITS.items()}
    report["habits_mine_per_1000"] = mine
    tried = lambda a: bool(a[2] or BREAK.match(a[0]))  # noqa: E731
    answers_of = {v: [[a[0] for a in per] for per in samples[v]] for v in VARIANTS}
    tries_of = {v: [sum(tried(a) for a in per) / max(1, len(per)) for per in samples[v]] for v in VARIANTS}
    for v in VARIANTS:
        texts = [a[0] for per in samples[v] for a in per]
        own = [i for i, per in enumerate(samples[v]) for _ in per]
        res = report["variants"][v]
        res["habits_per_1000"] = {h: per_1000(p, texts) for h, p in HABITS.items()}
        res["length_vs_mine"] = st.median(len(t.split()) for t in texts) / st.median(len(t.split()) for t in real)
        res["cut_off_share"] = sum(not a[1] for per in samples[v] for a in per) / len(texts)
        res["loops"] = sum(bool(LOOP.search(t.lower())) for t in texts)
        res["transcript_breaks"] = sum(tried(a) for per in samples[v] for a in per)
        res["content_overlap"] = st.mean(f1(content_words(t), content_words(real[i])) for t, i in zip(texts, own))
        res["content_overlap_other_answer"] = st.mean(f1(content_words(t), content_words(real[other[i]]))
                                                      for t, i in zip(texts, own))
        if sem:
            e = sem.encode(texts, normalize_embeddings=True)
            same = [float(e[j] @ real_emb[i]) for j, i in enumerate(own)]
            diff = [float(e[j] @ real_emb[other[i]]) for j, i in enumerate(own)]
            res["meaning_closer_to_mine"] = sum(a > b for a, b in zip(same, diff)) / len(same)
        if v != "window":
            res["habit_shift"] = {}
            for h, p in HABITS.items():
                point, lo, hi = habit_shift(p, mine[h], answers_of[v], answers_of["window"], classes)
                res["habit_shift"][h] = {"further_than_window": point, "ci95": [lo, hi]}
            words = lambda per: max(1.0, st.mean(len(t.split()) for t in per))  # noqa: E731
            ratio = [math.log(words(a) / words(b)) for a, b in zip(answers_of[v], answers_of["window"])]
            res["length_vs_window"] = st.mean(ratio)
            res["length_vs_window_ci95"] = list(class_bootstrap(ratio, classes))
            more = [a - b for a, b in zip(tries_of[v], tries_of["window"])]
            res["speaker_vs_window"] = st.mean(more)
            res["speaker_vs_window_ci95"] = list(class_bootstrap(more, classes))
            res["decision"] = verdict(res)
        common.write_jsonl(args.out / f"B-val-ctx-{v}.jsonl", [
            {"id": common.prompt_key(r["prompt"]), "class_id": r["class_id"], "source": r["source"],
             "los": r["los"], "prompt": prompts[v][i], "real": real[i], "variant": "B",
             "samples": [a[0] for a in samples[v][i]], "finished": [a[1] for a in samples[v][i]]}
            for i, r in enumerate(rows)])
    (args.out / "context-test.json").write_text(json.dumps(report, indent=2))
    print_all(report)
    if args.limit:
        print(f"\nA quick pass of {args.limit} answers: this checks the pipeline. Don't read anything into the verdicts.")
    print(f"\nwrote {args.out / 'context-test.json'}, context-test-answers.jsonl and B-val-ctx-<variant>.jsonl")


def table_line(name, cells, last=""):
    print(f"{name:36s}" + "".join(f"{c:>15s}" for c in cells) + f"{last:>10s}")


def print_loss(report):
    V = report["variants"]
    print(f"\n{report['answers']} answers in {report['classes']} classes")
    table_line("", VARIANTS)
    line = lambda name, f: table_line(name, [f(V[v]) for v in VARIANTS])  # noqa: E731
    line("prompt tokens (median)", lambda r: f"{r['prompt_tokens_median']:,.0f}")
    line("prompt + answer within 4,096", lambda r: f"{100 * r['fits_sliding_window']:.0f}%")
    line("loss on my answer (lower=better)", lambda r: f"{r['loss']:.4f}")
    line("  vs window", lambda r: f"{r['loss_vs_window']:+.4f}" if "ci95" in r else "")
    line("  95% interval (class bootstrap)", lambda r: f"{r['ci95'][0]:+.3f}/{r['ci95'][1]:+.3f}" if "ci95" in r else "")
    line("  better than window on", lambda r: f"{100 * r['better_on_share']:.0f}%" if "ci95" in r else "")
    print("\nloss vs window by length of the full prompt + answer")
    for name, b in report["by_length"].items():
        cells = [""] + [f"{b['variants'][v]['loss_vs_window']:+.4f}" for v in VARIANTS[1:]]
        table_line(f"{name}: {b['answers']} answers, {b['classes']} classes", cells)
        cells = [""] + ["{:+.3f}/{:+.3f}".format(*b["variants"][v]["ci95"]) for v in VARIANTS[1:]]
        table_line("  95% interval", cells)


def print_all(report):
    V, mine = report["variants"], report["habits_mine_per_1000"]
    print_loss(report)
    print()
    table_line("", VARIANTS, "me")
    line = lambda name, f, last="": table_line(name, [f(V[v]) for v in VARIANTS], last)  # noqa: E731
    for h in HABITS:
        line(f"'{h}' per 1,000", lambda r, h=h: f"{r['habits_per_1000'][h]:.1f}", f"{mine[h]:.1f}")
    line("length vs mine (median)", lambda r: f"{r['length_vs_mine']:.2f}x")
    line("cut off at the length limit", lambda r: f"{100 * r['cut_off_share']:.0f}%")
    line("answers stuck in a loop", lambda r: f"{r['loops']}")
    line("tried to write another speaker", lambda r: f"{r['transcript_breaks']}")
    line("content overlap with mine", lambda r: f"{100 * r['content_overlap']:.1f}%")
    line("  with an unrelated answer", lambda r: f"{100 * r['content_overlap_other_answer']:.1f}%")
    if "meaning_closer_to_mine" in V["window"]:
        line("meaning closer to mine than other", lambda r: f"{100 * r['meaning_closer_to_mine']:.0f}%")
    print()
    for v in VARIANTS[1:]:
        r, (lo, hi) = V[v], V[v]["ci95"]
        d = r["decision"]
        print(f"{v}: {d['verdict']}  loss {r['loss_vs_window']:+.4f} (95% {lo:+.4f} to {hi:+.4f}, limit +{report['loss_margin']})"
              + (": " + "; ".join(d["why"]) if d["why"] else ""))


if __name__ == "__main__":
    main()
