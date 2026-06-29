"""
PRISM - Semantic Memory Store
Long-term vector store using FAISS HNSW index + MiniLM-L6-v2 embeddings.
Stores user preferences, habits, and identity facts that survive across sessions.
"""

import os
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict
from typing import Optional


MODEL_NAME  = "all-MiniLM-L6-v2"
DIM         = 384          # MiniLM-L6-v2 output dimension
INDEX_PATH  = "data/semantic_index.faiss"
META_PATH   = "data/semantic_meta.json"


@dataclass
class SemanticMemory:
    content: str
    category: str          # "preference", "habit", "identity", "fact"
    timestamp: float
    salience_score: float
    memory_id: str = ""
    confirmed: bool = False

    def __post_init__(self):
        if not self.memory_id:
            self.memory_id = f"sem_{int(self.timestamp * 1000)}"


class SemanticStore:

    def __init__(self):
        self._model   = SentenceTransformer(MODEL_NAME)
        self._index   = faiss.IndexHNSWFlat(DIM, 32)   # HNSW for fast retrieval
        self._meta: list[SemanticMemory] = []
        self._load()

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(self, content: str, category: str = "fact",
            salience: float = 0.6) -> SemanticMemory:
        """Add a new long-term memory. Checks for conflicts first."""
        conflict = self._find_conflict(content)
        if conflict:
            self._resolve_conflict(conflict, content, salience)
            return conflict

        mem = SemanticMemory(
            content=content,
            category=category,
            timestamp=time.time(),
            salience_score=salience
        )
        vec = self._embed(content)
        self._index.add(vec)
        self._meta.append(mem)
        self._save()
        return mem

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list:
        """Retrieve top-k semantically similar memories."""
        if len(self._meta) == 0:
            return []
        vec = self._embed(query)
        k   = min(top_k, len(self._meta))
        distances, indices = self._index.search(vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            mem = self._meta[idx]
            results.append((float(dist), mem))
        return results

    def get_all(self) -> list:
        return list(self._meta)

    # ── Conflict resolution ────────────────────────────────────────────────────

    def _find_conflict(self, content: str, threshold: float = 0.92) -> Optional[SemanticMemory]:
        """Detect if a very similar (potentially conflicting) memory exists."""
        if not self._meta:
            return None
        results = self.search(content, top_k=1)
        if results and results[0][0] < threshold:   # FAISS L2: lower = more similar
            return results[0][1]
        return None

    def _resolve_conflict(self, old: SemanticMemory, new_content: str, salience: float):
        """Replace outdated memory with the newer one (intelligent forgetting)."""
        idx = self._meta.index(old)
        old.content        = new_content
        old.timestamp      = time.time()
        old.salience_score = salience
        # Re-embed and update FAISS (rebuild for simplicity at small scale)
        self._rebuild_index()
        self._save()

    def _rebuild_index(self):
        self._index = faiss.IndexHNSWFlat(DIM, 32)
        if self._meta:
            vecs = np.array([self._embed(m.content)[0] for m in self._meta])
            self._index.add(vecs)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self._index, INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump([asdict(m) for m in self._meta], f, indent=2)

    def _load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self._index = faiss.read_index(INDEX_PATH)
            with open(META_PATH) as f:
                raw = json.load(f)
            self._meta = [SemanticMemory(**r) for r in raw]

    # ── Util ───────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], normalize_embeddings=True)
        return vec.astype(np.float32)

    def __len__(self):
        return len(self._meta)

    def __repr__(self):
        return f"SemanticStore({len(self._meta)} memories)"
