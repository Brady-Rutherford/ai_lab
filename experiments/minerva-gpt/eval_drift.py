"""Drift check: do the model's answers still look like mine in shape?

    uv run python eval_drift.py data/generations/7b/A-val-final-plain.jsonl

Compares the model's answers with my real answers to the same asks: median length, how many
were cut off at the length cap instead of ending, distinct openings, and repeated 4-word
phrases. Session 2 caught DPO pushing 1B answers from 75 to 191 words, with 66% cut off; this
makes that a gate. Exit 1 when the median length is off by more than half, or when more than
MAX_CUT of the answers were cut off.
"""

import re
import statistics
import sys
from collections import Counter

import common

LENGTH_BAND = 0.5
MAX_CUT = 0.25


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def repeated(s):
    w = words(s)
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / max(1, len(grams))


def shape(texts):
    lengths = [len(words(t)) for t in texts]
    return {"median": statistics.median(lengths),
            "openings": len(Counter(" ".join(words(t)[:4]) for t in texts)) / len(texts),
            "repeated": statistics.mean(map(repeated, texts))}


def main(path):
    rows = common.read_jsonl(path)
    model = [s for r in rows for s in r["samples"]]
    flags = [f for r in rows for f in r.get("finished", [True] * len(r["samples"]))]
    mine, me = shape(model), shape([r["real"] for r in rows])
    cut = 1 - sum(flags) / len(flags)
    for name, s in (("model", mine), ("me (real)", me)):
        print(f"{name:10s} median {s['median']:.0f} words | distinct openings {s['openings']:.0%} | "
              f"repeated 4-grams {s['repeated']:.1%}")
    print(f"cut off at the length cap: {cut:.0%}")
    ratio = mine["median"] / max(me["median"], 1)
    ok = abs(ratio - 1) <= LENGTH_BAND and cut <= MAX_CUT
    print(f"drift: {'PASS' if ok else 'FAIL'} (median length {ratio:.2f}x mine, need within "
          f"{1 - LENGTH_BAND:.1f}-{1 + LENGTH_BAND:.1f}x; cut off {cut:.0%}, need <= {MAX_CUT:.0%})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
