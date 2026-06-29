"""
PRISM - Basic tests
Run: python tests/test_prism.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from src.memory.episodic   import EpisodicBuffer
from src.memory.salience   import SalienceScorer
from src.memory.procedural import ProceduralMemory


def test_episodic():
    print("Testing Episodic Buffer...")
    buf = EpisodicBuffer()
    buf.add("I prefer window seats", "user", ["preference", "travel"])
    buf.add("I love spicy food",     "user", ["preference", "food"])
    assert len(buf) == 2
    recent = buf.get_recent(1)
    assert "spicy food" in recent[0].content
    print("  ✅ Episodic Buffer OK")


def test_salience():
    print("Testing Salience Scorer...")
    scorer = SalienceScorer()
    buf    = EpisodicBuffer()
    ev     = buf.add("I prefer window seats on flights", "user", ["preference"])
    score  = scorer.compute(ev, query="book flight window seat")
    assert 0.0 <= score <= 1.0
    print(f"  ✅ Salience Score: {score:.4f} (threshold=0.6)")


def test_procedural():
    print("Testing Procedural Memory...")
    proc = ProceduralMemory()
    proc.add_pattern(
        trigger_keywords=["order", "food", "usual"],
        steps=["Open Swiggy", "Select restaurant", "Add items", "Place order"],
        description="Usual food order"
    )
    match = proc.match("order my usual food")
    assert match is not None
    assert match.description == "Usual food order"
    print(f"  ✅ Procedural match: '{match.description}'")


def test_all():
    print("\n🔮 Running PRISM tests...\n")
    test_episodic()
    test_salience()
    test_procedural()
    print("\n✅ All tests passed! PRISM components are working.\n")


if __name__ == "__main__":
    test_all()
