"""Fake data in the real schema, for smoke-testing the pipeline without touching class data.

    uv run python synthetic.py <dir>      # writes <dir>/sft/*.jsonl and <dir>/sft_classmates/*.jsonl

The fake "Brady" always opens with "Honestly," and the fake classmates never do, so even a
20-step run shows whether training picks up a voice and whether the speaker tag matters.
"""

import random
import sys
from pathlib import Path

import pandas as pd

from common import TAG_CLASSMATE, TAG_ME, write_jsonl

TOPICS = ["pricing", "feedback loops", "market research", "sampling bias", "incentives",
          "network effects", "brand trust", "causal diagrams"]
HCS = [("#complexcausality", "Identify ways that multiple causes interact."),
       ("#correlation", "Distinguish correlation and causation."),
       ("#breakitdown", "Organize problems into tractable components.")]


def row(rng, speaker, split):
    topic, (tag, desc) = rng.choice(TOPICS), rng.choice(HCS)
    who = TAG_ME if speaker == "brady" else TAG_CLASSMATE
    prompt = (f"{who}\nClass discussion in X101 Fake Course - Session {rng.randint(1, 20)}\n"
              f"Focused LOs/HCs:\n- {tag}: {desc}\n\n"
              f"Professor: How does {tag} help us think about {topic}?\n\nWhat do you say?")
    if speaker == "brady":
        answer = (f"Honestly, I think {topic} is mostly about how the pieces feed into each other, "
                  f"so {tag} makes us look at the loop instead of one cause.")
    else:
        answer = f"I would say {topic} shows that {desc.lower()}"
    grade = round(rng.uniform(2, 4), 1) if speaker == "brady" and rng.random() < 0.3 else None
    return {"prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": answer}],
            "source": "class_turn", "speaker": speaker, "class_id": rng.randint(1, 50),
            "los": [tag], "grade": grade, "grade_z": None if grade is None else round(grade - 3, 2)}


def main(out):
    rng = random.Random(0)
    for split, n in (("train", 16), ("val", 4), ("test", 4)):
        write_jsonl(out / f"sft/{split}.jsonl", [row(rng, "brady", split) for _ in range(n)])
        write_jsonl(out / f"sft_classmates/{split}.jsonl", [row(rng, "classmate", split) for _ in range(n)])
        docs = [f"X101 Fake Course\nSession {i}\n\nProfessor: What drives {t}?\n"
                f"Student 1: I think {t} depends on several interacting causes in the system.\n"
                f"Brady: Honestly, I think {t} is mostly a loop." for i, t in enumerate(TOPICS)]
        (out / "transcripts").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": docs}).to_parquet(out / f"transcripts/{split}.parquet")
    print(f"synthetic data written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
