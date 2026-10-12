"""Local sentence-transformers embeddings (no API needed)."""
from sentence_transformers import SentenceTransformer

import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed(texts, batch_size=64, show_progress=False):
    return get_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=show_progress, convert_to_numpy=True,
    )
