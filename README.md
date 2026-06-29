# 🔮 PRISM — Persistent, Retrieval-Indexed, Structured Memory

> **Context-Aware Adaptive Memory Solution for Mobile Agentic Systems**  
> Team MEGATRON · Samsung ennovateX AX Hackathon 2026 · Problem Statement 03

---

## 🧠 What is PRISM?

Today's mobile AI assistants are **fundamentally stateless** — every session starts from scratch. Tell your assistant you prefer window seats on flights today, and tomorrow it has no idea.

**PRISM fixes this.**

PRISM is a three-tier on-device cognitive memory framework that gives mobile AI agents persistent, intelligent memory — inspired by how the human brain actually stores information.

```
User: "I prefer window seats on flights"
← Next session →
User: "Book me a flight"
PRISM Agent: "Booking with window seat preference applied ✓"
```

---

## ✨ Key Innovations

| Innovation | What it does |
|---|---|
| **Salience-Weighted Promotion** | score = recency × frequency × relevance. Only important memories are kept. |
| **Intelligent Forgetting** | Contradictory facts are pruned, not appended. Prevents context bloat. |
| **Compute-Aware Scaling** | Low battery (<20%)? Injects 3 memories. Charging? Full 10. First system to do this. |
| **Procedural Shortcuts** | Learns multi-step task patterns. "Order my usual" → 1 command, done. |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                   USER INPUT                        │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 1: PERCEPTION ENGINE                         │
│  Text + Voice (Whisper-tiny) + App Context          │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 2: PRISM MEMORY ENGINE (Core)                │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │  Episodic    │ │  Semantic    │ │ Procedural  │ │
│  │  Buffer      │→│  Store       │ │ Cache       │ │
│  │  (50 events) │ │  (FAISS+     │ │ (SQLite     │ │
│  │              │ │   MiniLM)    │ │  graphs)    │ │
│  └──────────────┘ └──────────────┘ └─────────────┘ │
│         ↑ Salience Scorer (θ=0.6)                   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 3: REASONING AGENT                           │
│  Phi-3 Mini (3.8B, Q4 quantized) · Custom ReAct    │
│  Observe → Retrieve → Reason → Act → Store          │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 4: ADAPTATION LOOP                           │
│  Task signals → Procedural update                   │
│  Ebbinghaus forgetting curve → Episodic decay       │
│  Battery monitor → Compute-aware context scaling    │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/kunalkumar-aiml/PRISM-Memory-Agent.git
cd PRISM-Memory-Agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull Phi-3 Mini (via Ollama)
```bash
# Install Ollama from https://ollama.com
ollama pull phi3:mini
```

### 4. Run tests
```bash
python tests/test_prism.py
```

### 5. Launch demo
```bash
cd demo
streamlit run app.py
```

---

## 📁 Project Structure

```
PRISM-Memory-Agent/
├── src/
│   ├── memory/
│   │   ├── episodic.py      # Ring buffer (50 events, O(1) write)
│   │   ├── semantic.py      # FAISS HNSW vector store + MiniLM
│   │   ├── procedural.py    # SQLite task pattern store
│   │   ├── salience.py      # Salience scorer (θ=0.6 threshold)
│   │   └── manager.py       # Orchestrator — all 3 tiers
│   └── agent/
│       └── react_agent.py   # Custom ReAct loop + Phi-3 Mini
├── demo/
│   └── app.py               # Streamlit UI (Galaxy AI-style)
├── tests/
│   └── test_prism.py        # Component tests
├── data/                    # Persisted memory (auto-created)
└── requirements.txt
```

---

## 📊 Performance Targets

| Metric | Target | Status |
|---|---|---|
| Memory Recall Accuracy | 87%+ (LOCOMO benchmark) | 🔄 In progress |
| Retrieval Latency | <120ms P95 | 🔄 In progress |
| RAM Footprint | <150MB total | 🔄 In progress |
| Battery Overhead | <3%/hr | 🔄 In progress |
| Task Efficiency Gain | 3.2x on repeated workflows | 🔄 In progress |

---

## 🛠 Tech Stack

- **SLM:** Phi-3 Mini (3.8B, Q4_K_M quantized) via Ollama + llama.cpp
- **Embeddings:** MiniLM-L6-v2 (22MB, INT8 quantized)
- **Vector Store:** FAISS with HNSW indexing
- **Procedural Store:** SQLite graph store
- **Demo UI:** Streamlit
- **Voice (planned):** Whisper-tiny

---

## 👥 Team MEGATRON

| Member | College | Email |
|---|---|---|
| Kunal Kumar | SRM IST, Kattankulathur | kk4468@srmist.edu.in |
| Jyotirgamay Maurya | SRM IST, Ramapuram | jm4677@srmist.edu.in |

---

## 📄 License

MIT License — open for research and educational use.
