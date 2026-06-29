"""
PRISM - Streamlit Demo
Galaxy AI-style interface with live memory visualization.
Shows all 3 memory tiers updating in real-time.
"""

import streamlit as st
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.react_agent import PRISMAgent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRISM — Adaptive Memory Agent",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 20px 24px; border-radius: 12px; margin-bottom: 20px;
}
.main-header h1 { color: #e2e8f0; font-size: 28px; margin: 0; }
.main-header p  { color: #94a3b8; margin: 6px 0 0; font-size: 14px; }
.memory-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;
}
.memory-card .tag {
    display: inline-block; font-size: 10px; padding: 2px 7px;
    border-radius: 20px; margin-right: 4px; margin-bottom: 4px;
}
.tag-preference { background: #14532d; color: #86efac; }
.tag-identity   { background: #1e3a5f; color: #93c5fd; }
.tag-fact       { background: #3b1f5e; color: #d8b4fe; }
.tag-habit      { background: #78350f; color: #fde68a; }
.salience-bar   { height: 4px; border-radius: 2px; background: #334155; margin-top: 6px; }
.salience-fill  { height: 4px; border-radius: 2px; background: #3b82f6; }
.metric-box {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 10px 12px; text-align: center;
}
.metric-box .val { font-size: 22px; font-weight: 600; color: #60a5fa; }
.metric-box .lbl { font-size: 11px; color: #64748b; margin-top: 2px; }
.chat-user  { background: #1e3a5f; border-radius: 10px; padding: 10px 14px; margin: 6px 0; }
.chat-agent { background: #1e293b; border-radius: 10px; padding: 10px 14px; margin: 6px 0; border-left: 3px solid #3b82f6; }
.proc-card  { background: #1a2e1a; border: 1px solid #166534; border-radius: 8px; padding: 10px 12px; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Init agent ────────────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent    = PRISMAgent()
    st.session_state.messages = []
    # Seed some demo procedural memories
    st.session_state.agent.learn_procedure(
        trigger_keywords=["usual", "order", "food", "swiggy"],
        steps=["Open Swiggy", "Select saved restaurant", "Add usual items to cart", "Apply saved address", "Place order"],
        description="Usual food order"
    )
    st.session_state.agent.learn_procedure(
        trigger_keywords=["book", "flight", "travel"],
        steps=["Open travel app", "Select window seat preference", "Apply saved payment method", "Confirm booking"],
        description="Book flight with preferences"
    )

agent: PRISMAgent = st.session_state.agent

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔮 PRISM — Persistent, Retrieval-Indexed, Structured Memory</h1>
    <p>Context-Aware Adaptive Memory for Mobile Agentic Systems · Team MEGATRON · Samsung ennovateX AX Hackathon 2026</p>
</div>
""", unsafe_allow_html=True)

# ── Layout: Chat (left) + Memory Panel (right) ────────────────────────────────
col_chat, col_memory = st.columns([3, 2])

# ── CHAT COLUMN ───────────────────────────────────────────────────────────────
with col_chat:
    st.markdown("### 💬 Chat with PRISM")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-agent">🔮 <b>PRISM:</b> {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Message", placeholder="Talk to PRISM... (e.g. 'I prefer window seats on flights')", label_visibility="collapsed")
        col_send, col_reset = st.columns([4, 1])
        with col_send:
            submitted = st.form_submit_button("Send →", use_container_width=True)
        with col_reset:
            if st.form_submit_button("🔄", use_container_width=True):
                st.session_state.messages = []
                agent.reset_session()
                st.rerun()

    if submitted and user_input.strip():
        with st.spinner("PRISM is thinking..."):
            result = agent.chat(user_input.strip())

        st.session_state.messages.append({"role": "user",  "content": user_input.strip()})
        st.session_state.messages.append({"role": "agent", "content": result["reply"]})
        st.rerun()

    # Demo scenarios
    st.markdown("---")
    st.markdown("**🎯 Demo scenarios — click to try:**")
    demo_scenarios = [
        "My name is Kunal and I prefer window seats on all flights",
        "I love spicy food, especially biryani",
        "Book me a flight to Delhi",
        "Order my usual food",
        "What do you remember about me?",
        "I moved from Delhi to Chennai last month",
    ]
    cols = st.columns(2)
    for i, scenario in enumerate(demo_scenarios):
        with cols[i % 2]:
            if st.button(scenario[:45] + "...", key=f"demo_{i}", use_container_width=True):
                with st.spinner("PRISM is thinking..."):
                    result = agent.chat(scenario)
                st.session_state.messages.append({"role": "user",  "content": scenario})
                st.session_state.messages.append({"role": "agent", "content": result["reply"]})
                st.rerun()

# ── MEMORY PANEL ──────────────────────────────────────────────────────────────
with col_memory:
    stats = agent.get_stats()

    # Metrics
    st.markdown("### 📊 Live Memory Dashboard")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="val">{stats["episodic_count"]}</div><div class="lbl">Episodic</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="val">{stats["semantic_count"]}</div><div class="lbl">Semantic</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="val">{stats["procedural_count"]}</div><div class="lbl">Procedural</div></div>', unsafe_allow_html=True)

    battery = stats["battery_pct"]
    battery_color = "#ef4444" if battery < 20 else "#f59e0b" if battery < 50 else "#22c55e"
    st.markdown(f"🔋 Battery: **{battery:.0f}%** — Context k = **{agent.memory._compute_aware_k()}** memories injected")

    # Semantic memories
    st.markdown("---")
    st.markdown("#### 🧠 Semantic Memory (Long-term)")
    semantic_mems = agent.memory.semantic.get_all()
    if semantic_mems:
        for mem in reversed(semantic_mems[-8:]):
            tag_class = f"tag-{mem.category}"
            bar_width = int(mem.salience_score * 100)
            st.markdown(f"""
<div class="memory-card">
    <span class="tag {tag_class}">{mem.category}</span>
    <span style="color:#e2e8f0;font-size:13px;">{mem.content[:80]}</span>
    <div class="salience-bar"><div class="salience-fill" style="width:{bar_width}%"></div></div>
    <span style="color:#64748b;font-size:10px;">salience: {mem.salience_score:.2f}</span>
</div>""", unsafe_allow_html=True)
    else:
        st.info("No long-term memories yet. Start chatting!")

    # Episodic memories
    st.markdown("#### ⚡ Episodic Buffer (Recent)")
    recent_events = agent.memory.episodic.get_recent(5)
    if recent_events:
        for ev in recent_events:
            icon = "👤" if ev.role == "user" else "🔮"
            st.markdown(f'<div class="memory-card"><span style="color:#94a3b8;font-size:12px;">{icon} {ev.content[:70]}</span></div>', unsafe_allow_html=True)
    else:
        st.info("No recent events yet.")

    # Procedural memory
    st.markdown("#### ⚙️ Procedural Shortcuts")
    procedures = agent.memory.procedural.get_all()
    if procedures:
        for p in procedures:
            st.markdown(f"""
<div class="proc-card">
    <b style="color:#86efac;font-size:13px;">⚡ {p.description}</b><br>
    <span style="color:#64748b;font-size:11px;">Triggers: {', '.join(p.trigger_keywords[:4])} · Used: {p.success_count}x</span>
</div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p style="color:#475569;text-align:center;font-size:12px;">PRISM · Team MEGATRON · SRM Institute of Science and Technology · Samsung ennovateX AX Hackathon 2026</p>', unsafe_allow_html=True)
