"""Regurgitation test: does the model repeat, word for word, what someone else said in class?

    uv run python eval_regurgitation.py data/generations/B-val.jsonl [more generation files]

Takes every 12-word window of everyone's speech except mine (classmates and professors) from
the masked class documents in data/transcripts/*.parquet, and checks each generated answer for
any window it reproduces exactly (lower case, punctuation ignored). Exit 1 on any match.
A window that is also in the answer's own prompt (the ask, the reading, the discussion so far) is
the model quoting what it was shown, not recalling training data: it is listed and counted
separately, and does not fail the test.
Variant B trains on classmates' answers, so its outputs must pass this before they are trusted;
A is checked too, since classmates' words sit in its prompts.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

import common

WINDOW = 12
WORD = re.compile(r"[a-z0-9']+")
SPEAKER = re.compile(r"^(Professor|Student \d+|Speaker): (.*)$")


def windows(text):
    words = WORD.findall(text.lower())
    return {hash(tuple(words[i:i + WINDOW])) for i in range(len(words) - WINDOW + 1)}


def others_windows(data):
    seen = set()
    for path in sorted((data / "transcripts").glob("*.parquet")):
        for doc in pd.read_parquet(path)["text"]:
            for line in doc.split("\n"):
                m = SPEAKER.match(line)
                if m:
                    seen |= windows(m.group(2))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generations", type=Path, nargs="+")
    ap.add_argument("--data", type=Path, default=common.DATA)
    args = ap.parse_args()
    seen = others_windows(args.data)
    print(f"{len(seen):,} distinct {WINDOW}-word windows of others' speech")
    checked = hits = quoted = 0
    for path in args.generations:
        for row in common.read_jsonl(path):
            shown = windows("\n".join(m["content"] for m in row["prompt"]))
            for i, sample in enumerate(row["samples"]):
                checked += 1
                found = windows(sample) & seen
                if found - shown:
                    hits += 1
                    print(f"{path.name}: ask {row['id']} sample {i} repeats a {WINDOW}-word passage")
                elif found:
                    quoted += 1
                    print(f"{path.name}: ask {row['id']} sample {i} quotes its own prompt (not counted)")
    print(f"regurgitation: {hits} of {checked} answers repeat someone else's words verbatim; "
          f"{quoted} more quote a passage from their own prompt")
    return 1 if hits or not checked else 0


if __name__ == "__main__":
    sys.exit(main())
