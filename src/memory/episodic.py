"""
PRISM - Episodic Memory Buffer
Short-term ring buffer storing last 50 interactions with timestamps and context tags.
"""

import time
import json
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MemoryEvent:
    content: str
    role: str               # "user" or "agent"
    timestamp: float
    tags: list              # e.g. ["preference", "flight", "food"]
    access_count: int = 0
    salience_score: float = 0.0
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"ep_{int(self.timestamp * 1000)}"


class EpisodicBuffer:
    """
    Ring buffer holding the last MAX_SIZE memory events.
    Oldest events are automatically evicted when buffer is full.
    """

    MAX_SIZE = 50

    def __init__(self):
        self._buffer: deque = deque(maxlen=self.MAX_SIZE)

    def add(self, content: str, role: str, tags: Optional[list] = None) -> MemoryEvent:
        event = MemoryEvent(
            content=content,
            role=role,
            timestamp=time.time(),
            tags=tags or []
        )
        self._buffer.append(event)
        return event

    def get_recent(self, n: int = 10) -> list:
        """Return last n events, newest first."""
        events = list(self._buffer)
        return list(reversed(events))[:n]

    def get_all(self) -> list:
        return list(self._buffer)

    def search_by_tag(self, tag: str) -> list:
        return [e for e in self._buffer if tag in e.tags]

    def increment_access(self, event_id: str):
        for event in self._buffer:
            if event.event_id == event_id:
                event.access_count += 1
                break

    def to_dict_list(self) -> list:
        return [asdict(e) for e in self._buffer]

    def __len__(self):
        return len(self._buffer)

    def __repr__(self):
        return f"EpisodicBuffer({len(self._buffer)}/{self.MAX_SIZE} events)"
