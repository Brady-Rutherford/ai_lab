"""Grade model: predicts how a professor would grade an answer's content.

    uv run python reward_model.py --train
    uv run python reward_model.py --score data/generations/B-val.voice.jsonl

Trained on my answers that professors graded (polls and spoken turns), target `grade_z`: the
grade normalized per grader. Features are bge-small embeddings of the ask, of the answer, and
of their product, with HC hashtags stripped first, so naming an HC can't buy a score; applying
it has to. A ridge regression, since there are only about 1,100 graded answers.

Two checks print with --train, and the grade model is only used in the preference stage if both
pass:
- gate: Spearman correlation with real grades on the val split of at least 0.3;
- label-dropping: appending the LO's own description to an answer ("This shows how multiple
  causes interact...") may raise its predicted grade by at most 0.1 z on average. If it does
  more, the model rewards reciting rubric language, which DPO would learn to exploit.
Saved to checkpoints/grade_model.pkl (gitignored).
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV

import common
from retrieve import EMBEDDER, ask_text

SPEARMAN_GATE = 0.3
LABEL_DROP_MAX = 0.1
MODEL_PATH = common.CHECKPOINTS / "grade_model.pkl"
HASHTAG = re.compile(r"#[\w-]+")
LO_LINE = re.compile(r"^- (#[\w-]+): (.+)$", re.MULTILINE)


def strip_tags(text):
    return " ".join(HASHTAG.sub(" ", text).split())


class GradeModel:
    def __init__(self, ridge=None):
        self.encoder = SentenceTransformer(EMBEDDER, device=common.device(), local_files_only=True)
        self.ridge = ridge

    @classmethod
    def load(cls, path=MODEL_PATH):
        """Only the fitted ridge is saved (pickling the class itself would tie the file to
        whichever script saved it); the encoder reloads from the HF cache."""
        return cls(pickle.loads(Path(path).read_bytes()))

    def save(self, path=MODEL_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(pickle.dumps(self.ridge))

    def features(self, asks, answers):
        e = lambda xs: self.encoder.encode([strip_tags(x) for x in xs], normalize_embeddings=True,
                                           batch_size=64)
        a, b = e(asks), e(answers)
        return np.hstack([a, b, a * b])

    def fit(self, rows):
        x = self.features([ask_text(r) for r in rows], [r["completion"][0]["content"] for r in rows])
        self.ridge = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(x, [r["grade_z"] for r in rows])
        return self

    def predict(self, asks, answers):
        return self.ridge.predict(self.features(asks, answers))


def graded(data, split):
    return [r for r in common.read_jsonl(data / f"sft/{split}.jsonl") if r.get("grade_z") is not None]


def label_drop(model, rows):
    """Mean change in predicted grade when the LO's description is appended to the answer."""
    asks, plain, padded = [], [], []
    for r in rows:
        los = LO_LINE.findall(r["prompt"][0]["content"])
        if los:
            answer = r["completion"][0]["content"]
            asks.append(ask_text(r))
            plain.append(answer)
            padded.append(f"{answer} This shows how we {los[0][1][0].lower()}{los[0][1][1:]}")
    if not asks:
        return float("nan")
    return float(np.mean(model.predict(asks, padded) - model.predict(asks, plain)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=common.DATA)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score", type=Path, help="a generations file; writes <name>.grade.jsonl")
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    args = ap.parse_args()

    if args.train:
        train, val = graded(args.data, "train"), graded(args.data, "val")
        model = GradeModel().fit(train)
        print(f"grade model fit on {len(train)} graded answers (ridge alpha {model.ridge.alpha_:.3g})")
        if len(val) >= 10:
            rho = spearmanr(model.predict([ask_text(r) for r in val],
                                          [r["completion"][0]["content"] for r in val]),
                            [r["grade_z"] for r in val]).statistic
            print(f"gate: Spearman {rho:.3f} on {len(val)} val answers -> "
                  f"{'PASS' if rho >= SPEARMAN_GATE else 'FAIL: keep grades for SFT weighting only'}")
        else:
            print(f"gate: only {len(val)} graded val answers, too few to judge")
        drop = label_drop(model, val or train)
        print(f"label-dropping: appending the LO description moves the grade {drop:+.3f} z -> "
              f"{'PASS' if drop <= LABEL_DROP_MAX else 'FAIL: it rewards rubric language'}")
        model.save(args.model)
        print(f"saved to {args.model}")

    if args.score:
        model = GradeModel.load(args.model)
        rows = common.read_jsonl(args.score)
        for r in rows:
            r["grade_pred"] = [round(float(g), 4) for g in
                               model.predict([ask_text(r)] * len(r["samples"]), r["samples"])]
        out = args.score.with_suffix(".grade.jsonl")
        common.write_jsonl(out, rows)
        print(f"grades predicted for {sum(len(r['samples']) for r in rows)} answers -> {out}")


if __name__ == "__main__":
    main()
