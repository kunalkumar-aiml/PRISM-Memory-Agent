"""
PRISM - Memory Manager
Orchestrates all three memory tiers: Episodic → Semantic → Procedural
Handles promotion, retrieval, and compute-aware context scaling.
"""

import psutil
from src.memory.episodic   import EpisodicBuffer
from src.memory.semantic   import SemanticStore
from src.memory.procedural import ProceduralMemory
from src.memory.salience   import SalienceScorer


class PRISMMemoryManager:

    def __init__(self):
        self.episodic   = EpisodicBuffer()
        self.semantic   = SemanticStore()
        self.procedural = ProceduralMemory()
        self.scorer     = SalienceScorer()

    # ── Store new interaction ──────────────────────────────────────────────────

    def store(self, content: str, role: str = "user", tags: list = None):
        """Store a new event in episodic buffer and check for promotion."""
        event = self.episodic.add(content, role, tags or [])
        self._maybe_promote(event, content)
        return event

    def _maybe_promote(self, event, query: str):
        """Promote high-salience episodic memories to semantic store."""
        score = self.scorer.compute(event, query)
        if self.scorer.should_promote(score):
            category = self._classify_category(event.content)
            self.semantic.add(event.content, category=category, salience=score)

    def _classify_category(self, content: str) -> str:
        content_lower = content.lower()
        if any(w in content_lower for w in ["prefer", "like", "love", "hate", "always", "never", "want"]):
            return "preference"
        if any(w in content_lower for w in ["my name", "i am", "i'm", "i work", "i live"]):
            return "identity"
        if any(w in content_lower for w in ["usually", "every day", "habit", "routine"]):
            return "habit"
        return "fact"

    # ── Retrieve context ───────────────────────────────────────────────────────

    def retrieve_context(self, query: str) -> dict:
        """
        Build context package for the agent.
        Compute-aware: fewer memories injected on low battery.
        """
        top_k = self._compute_aware_k()

        # Semantic retrieval
        semantic_results = self.semantic.search(query, top_k=top_k)
        semantic_memories = [m.content for _, m in semantic_results]

        # Recent episodic events
        recent = self.episodic.get_recent(n=min(5, top_k))
        episodic_memories = [e.content for e in recent if e.role == "user"]

        # Procedural match
        procedure = self.procedural.match(query)

        return {
            "semantic":    semantic_memories,
            "episodic":    episodic_memories,
            "procedural":  procedure,
            "battery_pct": self._battery_pct(),
            "top_k_used":  top_k
        }

    def _compute_aware_k(self) -> int:
        """Inject fewer memories when battery is low."""
        battery = self._battery_pct()
        if battery < 20:
            return 3
        elif battery < 50:
            return 5
        else:
            return 10

    def _battery_pct(self) -> float:
        try:
            b = psutil.sensors_battery()
            return b.percent if b else 100.0
        except Exception:
            return 100.0

    # ── Build prompt context string ────────────────────────────────────────────

    def build_context_string(self, query: str) -> str:
        ctx = self.retrieve_context(query)
        parts = []

        if ctx["semantic"]:
            parts.append("Long-term memory (user preferences & facts):")
            for m in ctx["semantic"]:
                parts.append(f"  - {m}")

        if ctx["episodic"]:
            parts.append("\nRecent conversation context:")
            for e in ctx["episodic"]:
                parts.append(f"  - {e}")

        if ctx["procedural"]:
            p = ctx["procedural"]
            parts.append(f"\nLearned shortcut '{p.description}':")
            for i, step in enumerate(p.steps, 1):
                parts.append(f"  {i}. {step}")

        return "\n".join(parts) if parts else "No prior memory found."

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "episodic_count":   len(self.episodic),
            "semantic_count":   len(self.semantic),
            "procedural_count": len(self.procedural),
            "battery_pct":      self._battery_pct(),
        }
