"""Retrieval for the preference stage: my own best-graded past answers on the same LOs/HCs.

Next to the ask, the model gets a few things I actually said or wrote about the LOs/HCs that
class focused on, preferring the answers professors graded highest. That supplies the
"rightness" (how I applied the HC when it went well) while the model still writes a new
answer in my words. Retrieval only draws from the train split, so a held-out class never sees
its own answers, and it always skips the class being asked about.

Score = 0.4 BM25 + 0.4 embedding similarity (bge-small, each scaled to [0, 1])
      + 0.3 if the answer shares an LO/HC with the ask + up to 0.2 for a high grade.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import common

EMBEDDER = "BAAI/bge-small-en-v1.5"  # already cached by the CS156 pipeline
K = 2
MIN_WORDS = 15   # skip one-liners: they show no reasoning to borrow
MAX_WORDS = 120  # per retrieved answer, to keep the prompt short


def ask_text(row):
    """The ask itself: the prompt without its header (tag, course, LO list)."""
    return row["prompt"][0]["content"].partition("\n\n")[2]


class Retriever:
    def __init__(self, data=common.DATA):
        self.rows = [r for r in common.read_jsonl(data / "sft/train.jsonl")
                     if len(r["completion"][0]["content"].split()) >= MIN_WORDS]
        texts = [r["completion"][0]["content"] for r in self.rows]
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        self.encoder = SentenceTransformer(EMBEDDER, device=common.device(), local_files_only=True)
        self.emb = self.encoder.encode(texts, normalize_embeddings=True, batch_size=64)
        self.bonus = np.array([0.2 * min(max(r["grade_z"] or 0.0, 0.0), 2.0) / 2 for r in self.rows])

    def search(self, row, k=K):
        query = ask_text(row)
        bm = self.bm25.get_scores(query.lower().split())
        bm = bm / (bm.max() or 1.0)
        cos = (self.emb @ self.encoder.encode([query], normalize_embeddings=True)[0] + 1) / 2
        los = set(row["los"])
        shared = np.array([0.3 * bool(los & set(r["los"])) for r in self.rows])
        score = 0.4 * bm + 0.4 * cos + shared + self.bonus
        score[np.array([r["class_id"] == row["class_id"] for r in self.rows])] = -1.0
        return [self.rows[i] for i in np.argsort(-score)[:k] if score[i] > 0]

    def augment(self, row):
        """The row's prompt with my past answers inserted after the header."""
        hits = self.search(row)
        if not hits:
            return row["prompt"]
        past = "\n".join("- " + " ".join(h["completion"][0]["content"].split()[:MAX_WORDS])
                         for h in hits)
        head, sep, body = row["prompt"][0]["content"].partition("\n\n")
        return [{"role": "user", "content": f"{head}\nMy past answers on these LOs/HCs:\n{past}{sep}{body}"}]
