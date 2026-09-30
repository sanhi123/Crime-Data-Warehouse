"""FAISS persistence and transparent top-k evidence retrieval."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import faiss
from .document_builder import AnalyticalDocument
from .embeddings import EmbeddingModel


@dataclass
class RetrievedDocument:
    text: str
    metadata: dict
    score: float


class FaissRetriever:
    def __init__(self, index_dir: str | Path = "rag_index", embedder: EmbeddingModel | None = None):
        self.index_dir = Path(index_dir)
        self.embedder = embedder or EmbeddingModel()
        self.index = None
        self.documents: list[AnalyticalDocument] = []

    @property
    def index_path(self) -> Path: return self.index_dir / "crime.faiss"
    @property
    def metadata_path(self) -> Path: return self.index_dir / "documents.json"

    def build(self, documents: list[AnalyticalDocument]) -> None:
        if not documents:
            raise ValueError("Cannot build a RAG index without analytical documents.")
        vectors = self.embedder.encode([d.text for d in documents])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents = documents
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps([d.to_dict() for d in documents], indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("RAG index is missing. Run `python -m rag.build_index` first.")
        self.index = faiss.read_index(str(self.index_path))
        self.documents = [AnalyticalDocument(**item) for item in json.loads(self.metadata_path.read_text(encoding="utf-8"))]

    @staticmethod
    def _terms(text: str) -> set[str]:
        """Light normalization keeps the offline fallback useful for plurals."""
        words = re.findall(r"[a-z]{3,}", text.lower())
        return {word[:-1] if word.endswith("s") and len(word) > 4 else word for word in words}

    @staticmethod
    def _is_trend_question(question: str) -> bool:
        terms = ("increase", "decrease", "trend", "previous year", "compared with", "year-over-year")
        return any(term in question.lower() for term in terms)

    def _mentioned_city(self, question: str) -> str | None:
        question_lower = question.lower()
        for document in self.documents:
            city = document.metadata.get("city")
            if city and city.lower() in question_lower:
                return city.lower()
        return None

    def search(self, question: str, top_k: int = 5, min_score: float = 0.12) -> list[RetrievedDocument]:
        if self.index is None: self.load()
        if not question.strip() or self.index.ntotal == 0: return []
        trend_question = self._is_trend_question(question)
        # Comparison questions need enough FAISS candidates for metadata-aware reranking.
        candidate_count = self.index.ntotal if trend_question else min(top_k, self.index.ntotal)
        scores, ids = self.index.search(self.embedder.encode([question]), candidate_count)
        candidates = [RetrievedDocument(self.documents[i].text, self.documents[i].metadata, float(score))
                      for score, i in zip(scores[0], ids[0]) if i >= 0 and score >= min_score]
        # Hashing is an offline safety net, not a semantic model. Guard against hash
        # collisions so an unrelated question never gains fabricated evidence.
        if self.embedder._fallback is not None:
            ignored = {"the", "and", "for", "with", "from", "what", "why", "how", "does", "did", "are", "was", "this", "that", "about", "non", "existent"}
            terms = self._terms(question) - ignored
            candidates = [item for item in candidates
                          if terms.intersection(self._terms(item.text))]
        if trend_question:
            city = self._mentioned_city(question)
            def trend_priority(item: RetrievedDocument) -> tuple[int, float]:
                is_city_trend = item.metadata.get("kind") == "city_year_trend"
                if is_city_trend and (city is None or item.metadata.get("city", "").lower() == city):
                    return (0, -item.score)
                if is_city_trend:
                    return (1, -item.score)
                return (2, -item.score)
            candidates.sort(key=trend_priority)
        return candidates[:top_k]
