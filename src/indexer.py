"""Embed the corpus once and save it to disk."""
import json

import numpy as np

import config
from src import data_loader, embedder


def build_index():
    data = data_loader.load_hotpot()
    docs, questions = data_loader.build_corpus(data)
    print(f"{len(questions)} questions -> {len(docs)} unique paragraphs. Embedding (first run is slow)...")
    emb = embedder.embed([d["text"] for d in docs], show_progress=True).astype("float32")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_DIR / "embeddings.npy", emb)
    with open(config.INDEX_DIR / "docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f)
    with open(config.INDEX_DIR / "questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f)
    print(f"Saved index to {config.INDEX_DIR}")


if __name__ == "__main__":
    build_index()
