"""Streamlit chat UI, deployable standalone to Streamlit Cloud — runs the
LangGraph agent in-process (no FastAPI server needed on Cloud), same pattern
`DECK_GUIDELINE.md` describes. Separate from api/static/chat_test.html, which
talks to a *running FastAPI server* over HTTP/SSE; this app embeds the graph
directly, same idea as api/main.py's own lifespan (one shared checkpointer
connection, compiled once), just inside Streamlit's execution model instead
of FastAPI's. Gated behind the same shared X-API-Key the FastAPI app uses
(entered in the sidebar) -- this is a public URL once on Streamlit Cloud,
with nothing else standing between it and real OpenAI spend.

Streamlit Cloud's base image ships an old system sqlite3 that chromadb's
embedded sqlite usage doesn't support (raises "unsupported version of
sqlite3" at import time) -- confirmed a known, common Streamlit Cloud +
Chroma incompatibility, not specific to this project. Swapping in
pysqlite3-binary's bundled modern sqlite3 before any chromadb import is the
standard fix; must run before `from agents...` below, since agents.subgraphs
.rag imports chromadb at module level.
"""
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # not on Streamlit Cloud -- local system sqlite3 is new enough

import os
import uuid

import streamlit as st

# st.secrets -> os.environ, before any agents.* import: agents.llm.get_llm() and
# agents.subgraphs.rag read these via os.environ (shared with the FastAPI app, which
# loads them from .env instead) -- Streamlit Cloud has no .env file, only its own
# Secrets UI, surfaced as st.secrets. Local runs already have these in os.environ via
# .env + python-dotenv, so this is a no-op locally -- confirmed the hard way: with no
# secrets.toml anywhere, st.secrets.__contains__ raises StreamlitSecretNotFoundError
# instead of just behaving like an empty mapping, so the whole block is skipped
# outright rather than per-key when no secrets file exists at all.
try:
    _has_secrets = len(st.secrets) >= 0  # triggers the same lookup/parse st.secrets does lazily
except Exception:
    _has_secrets = False
if _has_secrets:
    for _key in (
        "LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP",
        "RETRIEVER_TOP_K", "RETRIEVER_MAX_REWRITES",
        "GRAPH_MAX_STEPS", "CONTEXT_TOKEN_THRESHOLD", "CONTEXT_KEEP_TOKENS",
    ):
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = str(st.secrets[_key])

from dotenv import load_dotenv
load_dotenv()  # local runs: picks up .env the same way api/main.py does; no-op on Cloud (no file)

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.orchestrator import CHECKPOINT_DB_PATH, build_graph

st.set_page_config(page_title="InsureX Sales Agent", page_icon="\U0001F4AC", layout="centered")

# API key gate -- same shared X-API-Key convention as api/auth.py and the other two
# ways to drive this agent (dashboard.html, chat_test.html both ask for it top-left).
# Checked, and the rest of the app (including get_graph(), which triggers real OpenAI
# calls building the Chroma index on a cold start) never runs for an unauthorized
# visitor -- this is a public Streamlit Cloud URL once deployed, not gated by anything
# else.
st.sidebar.caption("API Key")
_api_key_input = st.sidebar.text_input(
    "X-API-Key", type="password", label_visibility="collapsed",
    placeholder="วางค่า API_KEY จาก .env / Streamlit secrets",
)
_expected_api_key = os.environ.get("API_KEY")
if not _expected_api_key:
    st.error("Server misconfigured: ไม่ได้ตั้งค่า API_KEY ไว้")
    st.stop()
if _api_key_input != _expected_api_key:
    st.info("กรุณาใส่ API Key ที่แถบด้านซ้ายก่อนเริ่มใช้งาน")
    st.stop()


@st.cache_resource
def get_graph():
    """One checkpointer connection, one compiled graph, cached for the app's whole
    process lifetime -- same reasoning as api/main.py's lifespan, just via
    st.cache_resource instead of an asgi lifespan context manager. The connection is
    never explicitly closed (no app-wide shutdown hook in Streamlit); it lives until
    the process exits, same as any other st.cache_resource singleton.

    The context-manager generator from_conn_string() returns MUST be kept alive for
    as long as the connection needs to stay open -- confirmed the hard way, twice.
    First attempt stored it in a module-level global, which looked right but isn't:
    Streamlit re-executes the ENTIRE script top to bottom on every rerun (every new
    session, every chat message), including top-level statements outside this
    function -- a top-level `_cm = None` initializer ran again on every rerun and
    clobbered the real reference get_graph() had set, and since st.cache_resource
    skips re-running this function's body on a cache hit, nothing ever restored it.
    Once GC'd, the abandoned generator resumes past its `yield` and runs its
    connection-closing cleanup -- every later query then fails with "Cannot operate
    on a closed database," even though setup() ran successfully moments earlier.
    Fix: attach the reference to the cached graph object itself, which both
    survives for exactly as long as the cache entry does and touches no module-level
    statement that a later rerun could ever re-execute."""
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpointer_cm = SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH))
    checkpointer = checkpointer_cm.__enter__()
    checkpointer.setup()
    graph = build_graph().compile(checkpointer=checkpointer)
    graph._checkpointer_cm = checkpointer_cm  # keep it alive -- see note above
    return graph


graph = get_graph()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.caption(f"session_id\n\n`{st.session_state.session_id}`")
    if st.button("เริ่มบทสนทนาใหม่", width="stretch"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

st.title("InsureX Sales Agent")
st.caption("ถามเกี่ยวกับผลิตภัณฑ์ประกัน หรือบอกความสนใจเพื่อให้ทีมขายติดต่อกลับ")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("พิมพ์คำถาม เช่น Health Plus มีแผนอะไรบ้าง")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    def token_stream():
        # Sync port of agents.runner.stream_structured_events()'s token-filtering logic
        # (graph.stream(), not .astream() -- this project's nodes are sync-compatible,
        # already proven by eval/run_full_eval.py's graph.invoke() calls, so there's no
        # need for asyncio inside Streamlit's own execution model). Same two conditions,
        # same reasoning: namespace == () alone isn't enough to isolate supervisor's own
        # reply tokens from extract_lead_info's internal structured-output call, which
        # also has namespace == () -- gated on metadata["langgraph_node"] == "supervisor"
        # too. st.write_stream() wants a plain sync generator of strings.
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        state = {
            "session_id": st.session_state.session_id,
            "messages": [HumanMessage(content=prompt)],
            "query": None, "snippets": None, "step": 0,
            "conversation_summary": None, "trace": [], "last_retrieval_relevant": None,
        }
        for namespace, mode, chunk in graph.stream(
            state, config, stream_mode=["messages", "updates"], subgraphs=True,
        ):
            if mode != "messages":
                continue
            token_chunk, metadata = chunk
            if (
                namespace == ()
                and isinstance(token_chunk, AIMessageChunk)
                and token_chunk.content
                and metadata.get("langgraph_node") == "supervisor"
            ):
                yield token_chunk.content

    with st.chat_message("assistant"):
        reply = st.write_stream(token_stream())
    st.session_state.messages.append({"role": "assistant", "content": reply})
