# app_custom_ui.py — MCP + Streamlit chat (same logic, custom UI/layout)

import os
import json
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# ─────────────────────────────
# MCP server config (UNCHANGED LOGIC)
# ─────────────────────────────
SERVERS = {
    "Summarize": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "fastmcp",
            "run",
            r"C:\Users\Yash\Desktop\sample_project_1\expense-tracker\main.py",
        ],
    }
}

SYSTEM_PROMPT = (
    "You have access to tools. When you choose to call a tool, do not narrate status updates. "
    "After tools run, return only a concise final answer."
)

# ─────────────────────────────
# Page + global styling (UI ONLY)
# ─────────────────────────────
st.set_page_config(
    page_title="Expense MCP Assistant",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
        .app-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            color: #6b7280;
            margin-bottom: 1.2rem;
        }
        .chat-hint {
            color: #9ca3af;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────
# Sidebar (UI personality)
# ─────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ MCP Assistant")
    st.markdown("Interact with your **expense summarizer** using natural language.")
    st.divider()
    st.markdown("### Examples")
    st.markdown(
        "- *Summarize all expenses*\n"
        "- *What did I spend last month?*\n"
        "- *Give category-wise totals*"
    )
    st.divider()
    st.caption("Built with Streamlit · LangChain · MCP")

# ─────────────────────────────
# Header
# ─────────────────────────────
st.markdown('<div class="app-title">💬 Expense MCP Chat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask questions about your expenses. Tools run silently.</div>',
    unsafe_allow_html=True,
)

load_dotenv()

# ─────────────────────────────
# One-time init (LOGIC UNCHANGED)
# ─────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-5")

    st.session_state.client = MultiServerMCPClient(SERVERS)
    tools = asyncio.run(st.session_state.client.get_tools())
    st.session_state.tools = tools
    st.session_state.tool_by_name = {t.name: t for t in tools}

    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(tools)

    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.initialized = True

# ─────────────────────────────
# Chat history rendering (UI tweaks only)
# ─────────────────────────────
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if getattr(msg, "tool_calls", None):
            continue
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)

# ─────────────────────────────
# Chat input
# ─────────────────────────────
st.markdown('<div class="chat-hint">Tip: Ask high-level questions like summaries or totals.</div>', unsafe_allow_html=True)

user_text = st.chat_input("Ask about your expenses…")

if user_text:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_text)

    st.session_state.history.append(HumanMessage(content=user_text))

    # First LLM pass (LOGIC UNCHANGED)
    first = asyncio.run(
        st.session_state.llm_with_tools.ainvoke(st.session_state.history)
    )
    tool_calls = getattr(first, "tool_calls", None)

    if not tool_calls:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(first.content or "")
        st.session_state.history.append(first)
    else:
        # 1) append assistant tool-call message
        st.session_state.history.append(first)

        # 2) execute tools
        tool_msgs = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            tool = st.session_state.tool_by_name[name]
            res = asyncio.run(tool.ainvoke(args))
            tool_msgs.append(
                ToolMessage(tool_call_id=tc["id"], content=json.dumps(res))
            )

        st.session_state.history.extend(tool_msgs)

        # 3) final response
        final = asyncio.run(st.session_state.llm.ainvoke(st.session_state.history))
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(final.content or "")
        st.session_state.history.append(AIMessage(content=final.content or ""))
