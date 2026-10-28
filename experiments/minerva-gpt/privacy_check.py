"""Privacy gate for the SFT data: every roster name and hashed id, checked field by field.

oncue's own scanner treats .jsonl as binary and skips it, so scanning data/sft with it
reports "0 hits" without reading a single file. This uses the same needles (names.py) on
every prompt and answer, and sorts what it finds:

    definite   any roster name, short form of one, hashed id or the ID salt
    ambiguous  a name that is also a common word, at a sentence start with no name signal
               after it ("Will we...", "May I..."): usually the word, not the person

    uv run --project ~/cs156-ml python privacy_check.py data/sft data/sft_classmates

Exit code 0 only when files were scanned and nothing definite was found. Ambiguous hits are
counted and a few are shown, redacted, with the words that follow, for a human to judge.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from names import is_ambiguous, is_common_word_name, needles, nickname_re

SAMPLES = 12


def redact(term):
    return term[:1] + "*" * (len(term) - 1)


def fields(row):
    head, _, body = row["prompt"][0]["content"].partition("\n\n")
    return (("header", head), ("prompt", body), ("answer", row["completion"][0]["content"]))


def main(paths):
    nd = needles()
    files = sorted(f for p in map(Path, paths) for f in (p.glob("*.jsonl") if p.is_dir() else [p]))
    definite, ambiguous, samples, by_term = Counter(), Counter(), [], Counter()
    for path in files:
        for n, line in enumerate(open(path), start=1):
            for field, text in fields(json.loads(line)):
                for kind, term, start in nd.find(text):
                    common = kind == "name" and is_common_word_name(term)
                    if common and (field == "header" or is_ambiguous(text, start, start + len(term))):
                        ambiguous[field] += 1
                        by_term[term.lower()] += 1
                        if len(samples) < SAMPLES:
                            after = " ".join(text[start + len(term):].split()[:3])
                            samples.append(f"  {field}: {redact(term)} {after}")
                    else:
                        definite[(field, kind)] += 1
                        print(f"{path}:{n}: {field} {kind} {redact(term)}")
                if field != "header" and nickname_re():
                    for m in nickname_re().finditer(text):
                        definite[(field, "short_name")] += 1
                        print(f"{path}:{n}: {field} short_name {redact(m.group(0))}")
    print(f"privacy scan: {len(files)} files | definite hits: {sum(definite.values())} "
          f"{dict(definite)} | ambiguous: {sum(ambiguous.values())} {dict(ambiguous)}")
    if by_term:
        top = ", ".join(f"{redact(t)} {n}" for t, n in by_term.most_common(10))
        print(f"ambiguous by term ({len(by_term)} distinct; top 10): {top}")
    if samples:
        print("ambiguous samples (redacted name + the words after it):", *samples, sep="\n")
    if not files:
        print("FAIL: nothing was scanned")
        return 1
    return 1 if definite else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
