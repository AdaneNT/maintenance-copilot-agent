"""Lightweight retrieval over maintenance procedures.

Documents are split into sections (markdown headings) and indexed with a hybrid
TF-IDF representation (word uni/bi-grams + character n-grams). This keeps the demo dependency-light and fully
reproducible; the `Retriever` interface makes it easy to swap in a vector store
(e.g. Azure AI Search) later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def source(self) -> str:
        return f"doc:{self.doc_id}#{_slug(self.section)}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {
            "source": self.chunk.source,
            "title": self.chunk.title,
            "section": self.chunk.section,
            "text": self.chunk.text,
            "score": round(self.score, 3),
        }


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def chunk_markdown(doc_id: str, text: str) -> list[Chunk]:
    """Split a markdown document into one chunk per `##` section."""
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.M)
    title = title_match.group(1).strip() if title_match else doc_id
    parts = re.split(r"^##\s+", text, flags=re.M)
    chunks: list[Chunk] = []
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        body = body.strip()
        if body:
            chunks.append(Chunk(doc_id, title, heading.strip(), body))
    if not chunks:  # document without sections
        chunks.append(Chunk(doc_id, title, "main", text.strip()))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk")
        self.chunks = chunks
        # Title repeated to up-weight it. Word n-grams capture exact terms, character
        # n-grams make matching robust to word forms ("increase" vs "increasing").
        corpus = [f"{c.title} {c.title} {c.section} {c.text}" for c in chunks]
        self._vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self._matrix = self._vec.fit_transform(corpus)

    @classmethod
    def from_dir(cls, docs_dir: Path) -> "Retriever":
        chunks: list[Chunk] = []
        for path in sorted(Path(docs_dir).glob("*.md")):
            chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8")))
        return cls(chunks)

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[Hit]:
        q = self._vec.transform([query])
        scores = cosine_similarity(q, self._matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]
