import json
import os
from html import escape
from pathlib import Path
from typing import Any, Iterator

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".md", ".txt", ".html"}

st.set_page_config(
    page_title="Yet Another RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimalist Industrial CSS (No Emojis, High-Contrast Monospace Telemetry)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .sys-header {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        padding-bottom: 0.5rem;
        margin-bottom: 1.25rem;
    }
    [data-testid="stChatMessage"] [data-testid*="Avatar"],
    [data-testid="stChatMessage"] > div:first-child {
        display: none !important;
    }
    [data-testid="stChatMessage"] {
        background: transparent;
        border: 0;
    }
    .user-query-row {
        display: flex;
        justify-content: flex-end;
        width: 100%;
        margin: 0.5rem 0 1rem;
    }
    .user-query-bubble {
        display: inline-block;
        width: fit-content;
        max-width: min(75%, 800px);
        padding: 0.7rem 1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 1.25rem;
        background: rgba(128, 128, 128, 0.12);
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .telemetry-bar {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.78rem;
        color: #888888;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-local {
        display: inline-block;
        padding: 2px 6px;
        font-family: monospace;
        font-size: 0.72rem;
        font-weight: 600;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 3px;
        margin-right: 8px;
    }
    .badge-web {
        display: inline-block;
        padding: 2px 6px;
        font-family: monospace;
        font-size: 0.72rem;
        font-weight: 600;
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 3px;
        margin-right: 8px;
    }
    .source-row {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
        padding: 6px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# API Client Helpers
# ---------------------------------------------------------------------------
def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def trigger_index_sync() -> tuple[bool, str]:
    try:
        resp = requests.post(f"{API_BASE_URL}/index/sync", timeout=300)
        if resp.status_code == 200:
            return True, resp.json().get("message", "Index synchronized.")
        return False, resp.json().get("detail", f"HTTP {resp.status_code}")
    except requests.RequestException as exc:
        return False, str(exc)


def stream_rag_query(
    query: str,
    state_holder: dict[str, Any],
) -> Iterator[str]:
    """
    Consumes FastAPI SSE stream (/query/stream) and populates `state_holder`
    with metadata and latency while yielding text tokens for st.write_stream.
    """
    with requests.post(
        f"{API_BASE_URL}/query/stream",
        json={"query": query},
        stream=True,
        timeout=180,
    ) as resp:
        resp.raise_for_status()
        current_event = None

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("event:"):
                current_event = raw_line[len("event:") :].strip()
            elif raw_line.startswith("data:"):
                payload_str = raw_line[len("data:") :].strip()
                try:
                    data = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                if current_event == "metadata":
                    state_holder["sources"] = data.get("sources", [])
                    state_holder["web_fallback"] = data.get("web_fallback", False)
                    state_holder["rewritten_query"] = data.get("rewritten_query")
                elif current_event == "token":
                    token = data.get("token", "")
                    if token:
                        yield token
                elif current_event == "done":
                    state_holder["latency"] = data.get("latency", 0.0)


def render_telemetry_and_sources(meta: dict[str, Any]) -> None:
    web_fallback = meta.get("web_fallback", False)
    latency = meta.get("latency")
    rewritten = meta.get("rewritten_query")
    sources = meta.get("sources", [])

    badge_html = (
        '<span class="badge-web">WEB FALLBACK</span>'
        if web_fallback
        else '<span class="badge-local">LOCAL INDEX</span>'
    )
    latency_str = f"LATENCY: {latency:.2f}s" if latency is not None else ""
    rewrite_str = f" | SEARCH QUERY: \"{rewritten}\"" if web_fallback and rewritten else ""

    st.markdown(
        f'<div class="telemetry-bar">{badge_html}{latency_str}{rewrite_str}</div>',
        unsafe_allow_html=True,
    )

    if sources:
        with st.expander(f"CITATIONS ({len(sources)})", expanded=False):
            for idx, src in enumerate(sources, start=1):
                if src.get("source_type") == "local":
                    doc = src.get("document", "Unknown")
                    sec = src.get("section") or "Root"
                    score = src.get("score")
                    score_txt = f" | CONFIDENCE: {score:.4f}" if score is not None else ""
                    st.markdown(
                        f'<div class="source-row">[{idx}] <b>{doc}</b> // SECTION: {sec}{score_txt}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    title = src.get("document", "Web Result")
                    url = src.get("url", "#")
                    st.markdown(
                        f'<div class="source-row">[{idx}] <b>{title}</b> // <a href="{url}" target="_blank">{url}</a></div>',
                        unsafe_allow_html=True,
                    )


def render_user_query(query: str) -> None:
    st.markdown(
        f'<div class="user-query-row"><div class="user-query-bubble">{escape(query)}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: System Control & Corpus Management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sys-header">SYSTEM CONTROL</div>', unsafe_allow_html=True)

    api_online = check_api_health()
    status_text = "ONLINE" if api_online else "OFFLINE"
    st.code(f"API STATUS : {status_text}\nENDPOINT   : {API_BASE_URL}", language="text")

    st.markdown("### CORPUS MANAGEMENT")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus_files = sorted(
        [
            f
            for f in RAW_DATA_DIR.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    st.caption(f"INDEXED FILES IN {RAW_DATA_DIR.as_posix()} ({len(corpus_files)})")
    if corpus_files:
        file_lines = [
            f"{f.relative_to(RAW_DATA_DIR).as_posix()} ({f.stat().st_size // 1024} KB)"
            for f in corpus_files
        ]
        st.code("\n".join(file_lines), language="text")
    else:
        st.code("No documents found in data/raw.", language="text")

    uploaded_files = st.file_uploader(
        "UPLOAD DOCUMENTS TO DATA/RAW",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("SAVE & SYNC UPLOADED FILES", use_container_width=True):
            for uf in uploaded_files:
                dest = RAW_DATA_DIR / uf.name
                dest.write_bytes(uf.read())
            with st.spinner("Synchronizing incremental index..."):
                ok, msg = trigger_index_sync()
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(f"Sync failed: {msg}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("SYNC INDEX", use_container_width=True, disabled=not api_online):
            with st.spinner("Syncing..."):
                ok, msg = trigger_index_sync()
            if ok:
                st.success("Synced.")
                st.rerun()
            else:
                st.error(msg)
    with col2:
        if st.button("CLEAR SESSION", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main Console: Query & Telemetry History
# ---------------------------------------------------------------------------
st.title("Yet Another RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user_query(msg["content"])
    else:
        with st.chat_message("assistant", avatar=None):
            st.markdown(msg["content"])
            if "meta" in msg:
                render_telemetry_and_sources(msg["meta"])

if prompt := st.chat_input(
    "Enter technical query...",
    disabled=not api_online,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_user_query(prompt)

    with st.chat_message("assistant", avatar=None):
        stream_meta: dict[str, Any] = {
            "sources": [],
            "web_fallback": False,
            "rewritten_query": None,
            "latency": None,
        }
        try:
            with st.spinner("Generating response..."):
                full_response = st.write_stream(
                    stream_rag_query(prompt, stream_meta)
                )
            render_telemetry_and_sources(stream_meta)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": str(full_response),
                    "meta": stream_meta,
                }
            )
        except Exception as exc:
            st.error(f"STREAM EXECUTION ERROR: {exc}")