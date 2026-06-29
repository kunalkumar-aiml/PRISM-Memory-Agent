"""
PRISM - Salience Scorer
Computes salience score = 0.4*recency_decay + 0.3*access_frequency + 0.3*task_relevance
Only memories above threshold theta=0.6 get promoted to semantic store.
"""

import time
import math
from src.memory.episodic import MemoryEvent


THETA = 0.6          # Promotion threshold
DECAY_HALF_LIFE = 3600  # 1 hour in seconds — memories decay over time


class SalienceScorer:

    def compute(self, event: MemoryEvent, query: str = "", max_access: int = 10) -> float:
        recency   = self._recency_decay(event.timestamp)
        frequency = self._frequency_score(event.access_count, max_access)
        relevance = self._task_relevance(event.content, query)

        score = 0.4 * recency + 0.3 * frequency + 0.3 * relevance
        return round(score, 4)

    def _recency_decay(self, timestamp: float) -> float:
        """Exponential decay — recent memories score higher."""
        elapsed = time.time() - timestamp
        return math.exp(-elapsed / DECAY_HALF_LIFE)

    def _frequency_score(self, access_count: int, max_access: int) -> float:
        """Normalized access frequency."""
        if max_access == 0:
            return 0.0
        return min(access_count / max_access, 1.0)

    def _task_relevance(self, content: str, query: str) -> float:
        """Simple keyword overlap relevance (fast, no model needed)."""
        if not query:
            return 0.5   # neutral if no query
        content_words = set(content.lower().split())
        query_words   = set(query.lower().split())
        overlap = content_words & query_words
        if not query_words:
            return 0.5
        return min(len(overlap) / len(query_words), 1.0)

    def should_promote(self, score: float) -> bool:
        return score >= THETA

    def score_all(self, events: list, query: str = "") -> list:
        """Return events with their salience scores, sorted descending."""
        max_access = max((e.access_count for e in events), default=1)
        scored = []
        for event in events:
            s = self.compute(event, query, max_access)
            event.salience_score = s
            scored.append((s, event))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
