"""Hybrid retriever: dense cosine similarity + boost when a paragraph title appears in the query."""
import json
import re

import numpy as np

import config
from src import embedder

TITLE_BOOST = 0.3


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


class Retriever:
    def __init__(self):
        emb_path = config.INDEX_DIR / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError("Index not found. Run:  python build_index.py")
        self.emb = np.load(emb_path)
        with open(config.INDEX_DIR / "docs.json", "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        with open(config.INDEX_DIR / "questions.json", "r", encoding="utf-8") as f:
            self.questions = json.load(f)
        # "Ed Wood (film)" -> " ed wood " so it matches queries that mention "Ed Wood"
        self.padded_titles = []
        for d in self.docs:
            t = _norm(re.sub(r"\s*\(.*?\)\s*$", "", d["title"]))
            self.padded_titles.append(f" {t} " if len(t) >= 4 else None)

    def search(self, query, k=config.TOP_K):
        k = min(k, len(self.docs))
        q = embedder.embed([query])[0]
        scores = self.emb @ q
        qn = f" {_norm(query)} "
        boost = np.array([TITLE_BOOST if (t and t in qn) else 0.0 for t in self.padded_titles])
        scores = scores + boost
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [dict(self.docs[i], score=float(scores[i])) for i in idx]