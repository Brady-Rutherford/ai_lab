"""Build the private file that brady_gpt_live.ipynb asks you to upload. It stays off GitHub.

    uv run python demo/make_demo_data.py            # picks the moments at random
    uv run python demo/make_demo_data.py --pick ID,ID,ID,ID,ID

Reads the masked held-out results on this Mac and writes data/demo/demo_data.json (gitignored):
  moments    five held-out class moments: the ask, my real answer, the saved model answer, the prompt
  scorecard  other held-out prompts (20 by default) with my real answers and the full-precision loss on each
  my_answers my own spoken answers from the training split, for the predictive-text baseline
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data/generations/7b"
OUT = ROOT / "data/demo/demo_data.json"
SEED = 156
N_SCORECARD = 20
MAX_PROMPT_CHARS = 14_000  # about 3,500 tokens, which leaves room on a T4 beside the 8-bit model


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def by_id(variant):
    return {r["id"]: r for r in read(GEN / f"B-val-ctx-{variant}.jsonl")}


def n_words(text):
    return len(text.split())


def course_of(content):
    """'CS156 - some title' from the line 'Class discussion in CS156 - some title'."""
    line = content.split("\n")[1]
    return line.removeprefix("Class discussion in ").strip()


def the_ask(content):
    """The professor's lines just before my turn, from the prompt's last block. Classmates' lines stay out."""
    block = content.rsplit("\n\n", 2)[-2]
    ask = []
    for line in block.split("\n"):
        if line.startswith("Professor:"):
            ask.append(line.removeprefix("Professor:").strip())
    return " ".join(ask)


def eligible(window, tail, saved):
    real = window["real"]
    ask = the_ask(window["prompt"][0]["content"])
    return (
        25 <= n_words(real) <= 120
        and n_words(ask) >= 15
        and tail["finished"][0]
        and saved["finished"][0]
        and 15 <= n_words(saved["samples"][0]) <= 150
        and len(tail["prompt"][0]["content"]) <= MAX_PROMPT_CHARS
        and not re.search(r"(?i)\b(my (mom|dad|mother|father|family|health)|depress|diagnos|therapy)\b", real)
    )


def moment(window, tail, saved):
    content = window["prompt"][0]["content"]
    return {
        "id": window["id"],
        "course": course_of(content),
        "ask": the_ask(content),
        "real": window["real"],
        "saved": saved["samples"][0],
        "prompt": tail["prompt"][0]["content"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", help="comma-separated ids instead of the random pick")
    ap.add_argument("--show", action="store_true", help="print the picked moments")
    ap.add_argument("--scorecard", type=int, default=N_SCORECARD, help="how many scorecard questions; 20 suits a T4")
    args = ap.parse_args()

    window, tail, saved = by_id("window"), by_id("tail"), by_id("full-pcw")
    ok = [i for i in window if eligible(window[i], tail[i], saved[i])]
    rng = random.Random(SEED)
    rng.shuffle(ok)
    if args.pick:
        picked = args.pick.split(",")
    else:  # one per class and, while they last, one per course
        picked, courses, classes = [], set(), set()
        for i in ok:
            course = course_of(window[i]["prompt"][0]["content"]).split(" - ")[0]
            if window[i]["class_id"] not in classes and course not in courses:
                picked.append(i)
                courses.add(course)
                classes.add(window[i]["class_id"])
            if len(picked) == 5:
                break
        for i in ok:
            if len(picked) == 5:
                break
            if window[i]["class_id"] not in classes:
                picked.append(i)
                classes.add(window[i]["class_id"])

    # The full-precision loss on each spoken answer, to check the T4's 4-bit copy against.
    val = read(ROOT / "data/sft/val.jsonl")
    losses = next(r["losses"] for r in read(ROOT / "runs/perrow-7b.jsonl") if r["model"].endswith("final-dpobest-B"))
    loss_of = {val[n]["prompt"][0]["content"]: loss for n, loss in enumerate(losses)}

    others = [i for i in window if i not in picked and window[i]["prompt"][0]["content"] in loss_of]
    rng.shuffle(others)
    scorecard = [
        {"course": course_of(window[i]["prompt"][0]["content"]), "ask": the_ask(window[i]["prompt"][0]["content"]),
         "prompt": window[i]["prompt"][0]["content"], "real": window[i]["real"],
         "full_loss": loss_of[window[i]["prompt"][0]["content"]]}
        for i in others[:args.scorecard]
    ]
    mine = [r["completion"][0]["content"] for r in read(ROOT / "data/sft/train.jsonl") if r["source"] == "class_turn"]

    data = {
        "moments": [moment(window[i], tail[i], saved[i]) for i in picked],
        "scorecard": scorecard,
        "my_answers": mine,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False))

    print(f"{len(ok)} moments eligible of {len(window)}; picked: {','.join(picked)}")
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.1f} MB, {len(mine)} of my answers, "
          f"median {statistics.median(n_words(a) for a in mine):.0f} words")
    if args.show:
        for m in data["moments"]:
            print(f"\n--- {m['id']} | {m['course']} | prompt {len(m['prompt'])} chars")
            print("ASK:", m["ask"])
            print("ME:", m["real"])
            print("MODEL:", m["saved"])


if __name__ == "__main__":
    main()
