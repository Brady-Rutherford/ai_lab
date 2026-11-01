"""Memorization check: is the model learning how I answer, or reciting the answers it trained on?

    uv run python eval_memorization.py data/generations/A-train-plain.jsonl data/generations/A-val-plain.jsonl

Both files are generations from the same model, without retrieval: one on train asks (answers it
trained on), one on val asks (never seen). For each generated answer:
- recall: it contains a 12-word stretch of my real answer to that ask, word for word;
- copy: it contains a 12-word stretch of its own prompt, e.g. pasted from my pre-class work.
Recall on val asks is the honest baseline (shared phrases, a quoted reading); a train rate far
above it means memorizing. Exit 1 when the gap exceeds MAX_GAP.
"""

import re
import sys
from pathlib import Path

import common

WINDOW = 12
MAX_GAP = 0.10
WORD = re.compile(r"[a-z0-9']+")


def windows(text):
    w = WORD.findall(text.lower())
    return {tuple(w[i:i + WINDOW]) for i in range(len(w) - WINDOW + 1)}


def rates(path):
    rows = common.read_jsonl(path)
    recall = copy = n = 0
    for r in rows:
        real, prompt = windows(r["real"]), windows(r["prompt"][0]["content"])
        for sample in r["samples"]:
            mine = windows(sample)
            n += 1
            recall += bool(mine & real)
            copy += bool(mine & prompt)
    return recall / max(n, 1), copy / max(n, 1), n


def main(train_path, val_path):
    tr_recall, tr_copy, tr_n = rates(Path(train_path))
    va_recall, va_copy, va_n = rates(Path(val_path))
    gap = tr_recall - va_recall
    print(f"recall of my real answer (12 words verbatim): train asks {tr_recall:.1%} (n={tr_n}), "
          f"never-seen asks {va_recall:.1%} (n={va_n}), gap {gap:+.1%}")
    print(f"copied 12+ words from its own prompt: train {tr_copy:.1%}, never-seen {va_copy:.1%}")
    verdict = "PASS" if gap <= MAX_GAP else "FAIL: it is reciting training answers"
    print(f"memorization: {verdict} (gap must be <= {MAX_GAP:.0%})")
    return 0 if gap <= MAX_GAP else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
