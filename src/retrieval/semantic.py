from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticRetrieval:
    """
    Cosine similarity over Sentence-BERT embeddings.
    model_name: Hugging Face model identifier
    embeddings: corpus embeddings, shape (n_docs, dim)
    n_docs: number of documents in the corpus
    """
    def __init__(
            self,
            model_name: str = DEFAULT_MODEL,
            device: str | None = None,
            batch_size: int = 64,
            show_progress: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress

        self._model: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None
        self.n_docs = 0

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress,
        )

    def fit(self, corpus: Sequence[str]) -> "SemanticRetrieval":
        if not corpus:
            raise ValueError("Must be at least one document")

        self.embeddings = self.encode(corpus)
        self.n_docs = len(corpus)
        return self

    def validate_fitted(self) -> None:
        if self.embeddings is None or self.n_docs == 0:
            raise RuntimeError("call fit() before scoring")

    def score(self, query: str) -> list[float]:
        """Cosine similarity of ``query`` against every corpus document."""

        self.validate_fitted()

        query_embedding = self.encode([query])[0]
        return (self.embeddings @ query_embedding).astype(float).tolist()

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return ``(doc_id, score)`` for the ``k`` most similar documents."""

        if k <= 0:
            raise ValueError("k must be positive")

        self.validate_fitted()

        scores = np.asarray(self.score(query))
        best = np.argsort(-scores, kind="stable")[:k]

        return [(int(doc_id), float(scores[doc_id])) for doc_id in best]

    def score_pair(self, a: str, b: str) -> float:
        embedding_a, embedding_b = self.encode([a, b])
        return float(embedding_a @ embedding_b)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        pairs = list(pairs)
        if not pairs:
            return []

        texts = [text for pair in pairs for text in pair]
        embeddings = self.encode(texts).reshape(len(pairs), 2, -1)

        return (embeddings[:, 0, :] * embeddings[:, 1, :]).sum(axis=1).tolist()
