"""Pick any class and any point in it, and get the text to paste into brady_gpt_ask_live.ipynb.

    uv run python demo/pick_moment.py                  # list the held-out classes
    uv run python demo/pick_moment.py bayes            # only those matching a word
    uv run python demo/pick_moment.py 12 --show        # class 12, with every turn numbered
    uv run python demo/pick_moment.py 12 --at 40       # everything said before turn 40, copied to the clipboard

It builds the prompt layout the model trained on. If turn 40 is mine, it also prints what I really said.
Class text only goes to this terminal and the clipboard, so it never reaches a file or GitHub.
Held-out classes are the default because the model never trained on them. --split train shows the
others, which it has seen, so answers there prove less. The test split is kept for the live replay.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ASK_WORDS, REPLY_WORDS = 150, 200  # the same limits prepare_data.py used in training
HISTORY_CHARS = 10_000  # about 2,500 tokens, the `tail` prompt, which fits on a T4
TURN = re.compile(r"^(Professor|Brady|Speaker|Student \d+): (.*)$")


def read_classes(split):
    return [doc for doc in pd.read_parquet(ROOT / f"data/transcripts/{split}.parquet")["text"]]


def parse(doc):
    """(course, title, turns), with each turn as (who, text, activity)."""
    lines = doc.split("\n")
    course, title = lines[0], re.sub(r" \(\d[^)]*\)$", "", lines[1])
    turns, activity = [], None
    for line in lines[2:]:
        match = TURN.match(line)
        if match:
            turns.append((match[1], match[2], activity))
        elif line.startswith("[") and line.endswith("]"):
            activity = line[1:-1]
    return course, title, turns


def last_words(text, n):
    words = text.split()
    return " ".join(words[-n:])


def header_for(course, title, split):
    """The readings, outcomes and my pre-class work from the training data, when this class has them."""
    line = f"Class discussion in {course} - {title}"
    with open(ROOT / f"data/sft/{split}.jsonl") as f:
        for row in f:
            content = json.loads(row)["prompt"][0]["content"]
            if content.split("\n")[1] == line and "\n\nProfessor: " in content:
                return content.split("\n\nProfessor: ")[0]
    return f"Answer as: Brady\n{line}"


def build(course, title, turns, at, split):
    profs = [i for i in range(at) if turns[i][0] == "Professor"]
    if not profs:
        sys.exit(f"No professor turn before turn {at}: pick a later one.")
    ask = profs[-1]
    history = "\n".join(f"{who}: {text}" for who, text, _ in turns[:ask])[-HISTORY_CHARS:]
    if history:
        history = history[history.find("\n") + 1:] if len(history) == HISTORY_CHARS else history
    body = [f"Professor: {last_words(turns[ask][1], ASK_WORDS)}"]
    replies = " ".join(text for _, text, _ in turns[ask + 1:at])
    if replies:
        keep = last_words(replies, REPLY_WORDS).split()
        kept, count = [], 0  # keep the speaker labels, trimming from the oldest reply
        for who, text, _ in reversed(turns[ask + 1:at]):
            words = text.split()
            if count + len(words) > len(keep):
                words = words[-(len(keep) - count):] if len(keep) > count else []
            if words:
                kept.append(f"{who}: {' '.join(words)}")
            count += len(words)
        body += reversed(kept)

    header = header_for(course, title, split)
    head_lines = [l for l in header.split("\n") if not l.startswith(("Activity:", "On screen"))]
    if turns[at - 1][2]:
        head_lines.insert(2, f"Activity: {turns[at - 1][2]}")
    head = "\n".join(head_lines)
    so_far = f"\n\nClass so far:\n{history}" if history else ""
    return f"{head}{so_far}\n\n" + "\n".join(body) + "\n\nWhat do you say?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="a class number, or a word to search for")
    ap.add_argument("--show", action="store_true", help="print the class with its turns numbered")
    ap.add_argument("--at", type=int, help="stop just before this turn and build the prompt")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    args = ap.parse_args()
    docs = read_classes(args.split)

    if args.target is None or not args.target.isdigit():
        word = (args.target or "").lower()
        for n, doc in enumerate(docs):
            course, title, turns = parse(doc)
            if word in f"{course} {title}".lower():
                mine = sum(who == "Brady" for who, _, _ in turns)
                print(f"{n:3d}  {title[:70]:70s}  {len(turns):4d} turns, {mine:3d} mine")
        return

    course, title, turns = parse(docs[int(args.target)])
    print(f"{course} - {title}: {len(turns)} turns")
    if args.show or args.at is None:
        for n, (who, text, _) in enumerate(turns):
            print(f"{n:4d} {'*' if who == 'Brady' else ' '} {who}: {text[:110]}")
        print("\n* marks my turns. Pick one and run this again with --at NUMBER.")
        return
    if not 0 < args.at <= len(turns):
        sys.exit(f"--at must be between 1 and {len(turns)}.")

    prompt = build(course, title, turns, args.at, args.split)
    subprocess.run("pbcopy", input=prompt.encode(), check=False)
    print(f"Copied to the clipboard: {len(prompt.split()):,} words. Paste it into the live notebook.\n")
    print("Last thing said:", prompt.rsplit("\n\nWhat do you say?", 1)[0].rsplit("\n", 1)[-1][-400:])
    if args.at < len(turns) and turns[args.at][0] == "Brady":
        print("\nWhat I really said:", turns[args.at][1])
    elif args.at < len(turns):
        print(f"\nI didn't speak next: {turns[args.at][0]} did.")


if __name__ == "__main__":
    main()
