"""
PRISM - ReAct Agent
Custom Observe → Retrieve → Reason → Act → Store loop.
Integrates Phi-3 Mini (via Ollama) with PRISM memory system.
"""

import ollama
from src.memory.manager import PRISMMemoryManager


SYSTEM_PROMPT = """You are PRISM — a smart personal AI assistant with persistent memory.
You remember the user's preferences, habits, and past interactions across sessions.
Always use the provided memory context to give personalized, relevant responses.
Be concise, friendly, and proactive. If you learn something new about the user, acknowledge it."""

MODEL = "phi3:mini"


class PRISMAgent:

    def __init__(self):
        self.memory  = PRISMMemoryManager()
        self._history = []   # in-session conversation history for Ollama

    def chat(self, user_input: str) -> dict:
        """
        Full ReAct cycle:
        1. OBSERVE  — receive user input
        2. RETRIEVE — get relevant memories
        3. REASON   — build augmented prompt
        4. ACT      — call Phi-3 Mini
        5. STORE    — save interaction to memory
        """

        # 1. OBSERVE
        self.memory.store(user_input, role="user",
                          tags=self._extract_tags(user_input))

        # 2. RETRIEVE
        context_str = self.memory.build_context_string(user_input)
        ctx         = self.memory.retrieve_context(user_input)

        # 3. REASON — build augmented system prompt
        augmented_system = SYSTEM_PROMPT
        if context_str.strip() and context_str != "No prior memory found.":
            augmented_system += f"\n\n[MEMORY CONTEXT]\n{context_str}"

        # 4. ACT — call Phi-3 Mini
        messages = [{"role": "system", "content": augmented_system}]
        messages += self._history[-6:]   # last 3 turns for in-session context
        messages.append({"role": "user", "content": user_input})

        response = ollama.chat(model=MODEL, messages=messages)
        reply    = response["message"]["content"]

        # 5. STORE — save agent reply
        self._history.append({"role": "user",      "content": user_input})
        self._history.append({"role": "assistant",  "content": reply})
        self.memory.store(reply, role="agent")

        return {
            "reply":    reply,
            "context":  ctx,
            "memory_stats": self.memory.stats()
        }

    def _extract_tags(self, text: str) -> list:
        """Simple keyword-based tag extraction."""
        tag_map = {
            "food":       ["food", "eat", "restaurant", "order", "pizza", "swiggy", "zomato"],
            "travel":     ["flight", "train", "hotel", "trip", "travel", "book", "cab", "uber"],
            "preference": ["prefer", "like", "love", "hate", "always", "never"],
            "identity":   ["my name", "i am", "i live", "i work", "i study"],
            "reminder":   ["remind", "reminder", "alarm", "schedule", "meeting"],
        }
        text_lower = text.lower()
        tags = []
        for tag, keywords in tag_map.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag)
        return tags

    def learn_procedure(self, trigger_keywords: list, steps: list, description: str):
        """Manually teach the agent a new procedural shortcut."""
        self.memory.procedural.add_pattern(trigger_keywords, steps, description)

    def reset_session(self):
        """Clear in-session history (not persistent memory)."""
        self._history = []

    def get_stats(self) -> dict:
        return self.memory.stats()
