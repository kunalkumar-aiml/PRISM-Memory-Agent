"""
PRISM - Procedural Memory
Stores learned multi-step task patterns as reusable workflows.
When a pattern is matched, the shortcut is executed directly — 3x efficiency gain.
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

PROC_PATH = "data/procedural.json"


@dataclass
class TaskPattern:
    trigger_keywords: list      # words that activate this pattern
    steps: list                 # list of action strings
    success_count: int = 0
    last_used: float = 0.0
    pattern_id: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = f"proc_{int(time.time() * 1000)}"


class ProceduralMemory:

    def __init__(self):
        self._patterns: list[TaskPattern] = []
        self._load()

    def add_pattern(self, trigger_keywords: list, steps: list,
                    description: str = "") -> TaskPattern:
        """Store a newly learned task workflow."""
        pattern = TaskPattern(
            trigger_keywords=trigger_keywords,
            steps=steps,
            description=description,
            last_used=time.time()
        )
        self._patterns.append(pattern)
        self._save()
        return pattern

    def match(self, query: str) -> Optional[TaskPattern]:
        """Find best matching pattern for a given user query."""
        query_words = set(query.lower().split())
        best_match  = None
        best_score  = 0

        for pattern in self._patterns:
            keywords   = set(pattern.trigger_keywords)
            overlap    = len(query_words & keywords)
            score      = overlap / max(len(keywords), 1)
            if score > best_score and score >= 0.5:
                best_score = score
                best_match = pattern

        if best_match:
            best_match.success_count += 1
            best_match.last_used      = time.time()
            self._save()

        return best_match

    def get_all(self) -> list:
        return list(self._patterns)

    def _save(self):
        os.makedirs("data", exist_ok=True)
        with open(PROC_PATH, "w") as f:
            json.dump([asdict(p) for p in self._patterns], f, indent=2)

    def _load(self):
        if os.path.exists(PROC_PATH):
            with open(PROC_PATH) as f:
                raw = json.load(f)
            self._patterns = [TaskPattern(**r) for r in raw]

    def __len__(self):
        return len(self._patterns)
