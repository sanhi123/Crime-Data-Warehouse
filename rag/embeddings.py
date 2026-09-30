"""Embeddings with a lightweight local model and an offline deterministic fallback."""
from __future__ import annotations
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", allow_download: bool = False):
        self.name = model_name
        self._model = None
        self._fallback = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, local_files_only=not allow_download)
        except Exception:
            from sklearn.feature_extraction.text import HashingVectorizer
            self.name = "hashing-fallback-384"
            self._fallback = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2", stop_words="english")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype="float32")
        return self._fallback.transform(texts).toarray().astype("float32")
