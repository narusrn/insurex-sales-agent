"""kb_retriever subgraph — Phase 2: agentic RAG, retrieve -> grade -> rewrite loop.
See implementation-plan.md §2/§3.0. Grade/rewrite prompts reused from BBLAgent's
agents/retriever.py near-verbatim (agents/prompts/_grade_rewrite.py).
"""
import os
from pathlib import Path
from typing import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..llm import get_llm
from ..state import RetrieverState
from ..prompts._grade_rewrite import GRADE_PROMPT, REWRITE_PROMPT

PDF_DIR = Path(__file__).resolve().parent.parent.parent / "Data" / "pdfs"
CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "chroma_db"
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
TOP_K = int(os.environ.get("RETRIEVER_TOP_K", "2"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
MAX_REWRITES = int(os.environ.get("RETRIEVER_MAX_REWRITES", "2"))

_vectorstore = None


def _load_pdfs() -> list[Document]:
    from pypdf import PdfReader  # stdlib-adjacent, already a dependency — no langchain-community needed

    docs = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
        docs.append(Document(page_content=text, metadata={"source": pdf_path.name}))
    return docs


def _get_retriever():
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        if CHROMA_DIR.exists():
            _vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(_load_pdfs())
            _vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=str(CHROMA_DIR))
    return _vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def _retrieve_chunks(query: str) -> list[str]:
    docs = _get_retriever().invoke(query)
    # [source] tag kept in each chunk's own text — the only channel `snippets` (§2's bridge
    # field) carries to the parent, and eval/metrics.py's Hit Rate@k/MRR (§5) need to know which
    # PDF each chunk came from. Harmless/plausibly helpful for the LLM's own grounding too.
    return [f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in docs]


@tool
def retrieve_knowledge_base(query: str) -> str:
    """Search the InsureX insurance product knowledge base for passages relevant to the query."""
    return "\n\n".join(_retrieve_chunks(query))


SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยค้นข้อมูลจากฐานความรู้ผลิตภัณฑ์ประกันของ InsureX "
    "ใช้ tool retrieve_knowledge_base เพื่อค้นข้อมูลที่เกี่ยวข้องกับคำถามของลูกค้าเสมอ"
)


def generate_query_or_respond(state: RetrieverState) -> dict:
    history = state.get("retriever_messages") or []
    seed: list = []
    if not history:
        seed = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=state["query"])]
        history = seed
    llm = get_llm(temperature=0).bind_tools([retrieve_knowledge_base])
    response = llm.invoke(history)
    update = {"retriever_messages": seed + [response]}
    if getattr(response, "tool_calls", None):
        q = response.tool_calls[0]["args"].get("query", state["query"])
        update["trace"] = [f'RAG Agent: ค้นข้อมูลด้วยคำค้น "{q}"']
    return update


def route_on_tool_calls(state: RetrieverState) -> str:
    last = state["retriever_messages"][-1]
    return "retrieve" if getattr(last, "tool_calls", None) else END


def retrieve_node(state: RetrieverState) -> dict:
    tool_calls = state["retriever_messages"][-1].tool_calls
    tool_messages = []
    snippets: list[str] = []  # one entry per actual chunk, not per tool call — eval/metrics.py
    # needs this to count each [source]-tagged chunk individually, not one bundled string per call
    for tc in tool_calls:
        chunks = _retrieve_chunks(tc["args"]["query"])
        snippets.extend(chunks)
        tool_messages.append(ToolMessage(content="\n\n".join(chunks), tool_call_id=tc["id"]))
    return {
        "retriever_messages": tool_messages,
        "snippets": snippets,
        "trace": [f"RAG Agent: ค้นพบ {len(snippets)} ข้อความจากฐานความรู้"],
    }


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Are the retrieved docs relevant to the question?")


def grade_documents(state: RetrieverState) -> dict:
    """NODE (not a bare conditional-edge function, unlike BBLAgent's own version) — needs to be
    one so it can write its own trace line (§3.8's "RAG Agent (ตรวจสอบ)" example). `last_retrieval_
    relevant` IS shared with GraphState (Phase 5.4, renamed from `relevant`) — a plain overwrite
    field, same safe pattern as `query`/`snippets` [§2's table], not an accumulator, so none of
    Phase 1's cross-boundary double-merge risk applies. Needed so after_retrieve can tell supervisor
    when it's handing over snippets that were never actually graded relevant (retries exhausted)."""
    context = "\n\n".join(state.get("snippets") or [])
    grader = get_llm(temperature=0).with_structured_output(GradeDocuments)
    result = grader.invoke(GRADE_PROMPT.format(question=state["query"], context=context))
    relevant = result.binary_score == "yes"
    retries = state.get("retries", 0)
    if relevant:
        line = "RAG Agent (ตรวจสอบ): ข้อมูลที่ได้เกี่ยวข้องเพียงพอ ✅"
    else:
        line = f"RAG Agent (ตรวจสอบ): ข้อมูลที่ได้ไม่เกี่ยวข้องเพียงพอ ❌ — ลองใหม่ ({min(retries + 1, MAX_REWRITES)}/{MAX_REWRITES})"
    return {"last_retrieval_relevant": relevant, "trace": [line]}


def route_after_grade(state: RetrieverState) -> str:
    if state.get("last_retrieval_relevant") or state.get("retries", 0) >= MAX_REWRITES:
        return END
    return "rewrite_question"


def rewrite_question(state: RetrieverState) -> dict:
    """Appends the rewritten query as a new HumanMessage into `retriever_messages` — the ORIGINAL
    `state["query"]` is never mutated (grade_documents keeps grading against the real user intent,
    not the rewritten search text; same reasoning BBLAgent's own version uses). On the next pass,
    generate_query_or_respond sees a non-empty history and just continues the conversation with
    this new message, no re-seeding needed."""
    llm = get_llm(temperature=0)
    response = llm.invoke(REWRITE_PROMPT.format(question=state["query"]))
    new_query = str(response.content).strip()
    return {
        "retriever_messages": [HumanMessage(content=new_query)],
        "retries": state.get("retries", 0) + 1,
        "trace": [f'RAG Agent: เขียนคำถามใหม่เป็น "{new_query}"'],
    }


builder = StateGraph(RetrieverState)
builder.add_node("generate_query_or_respond", generate_query_or_respond)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade_documents", grade_documents)
builder.add_node("rewrite_question", rewrite_question)
builder.add_edge(START, "generate_query_or_respond")
builder.add_conditional_edges("generate_query_or_respond", route_on_tool_calls, {"retrieve": "retrieve", END: END})
builder.add_edge("retrieve", "grade_documents")
builder.add_conditional_edges("grade_documents", route_after_grade, {"rewrite_question": "rewrite_question", END: END})
builder.add_edge("rewrite_question", "generate_query_or_respond")  # the actual cycle

graph = builder.compile()


if __name__ == "__main__":
    # smoke test — Phase 1.2: manually query the retriever, eyeball the chunks
    from dotenv import load_dotenv
    load_dotenv()
    docs = _get_retriever().invoke("Health Plus มีแผนอะไรบ้าง")
    assert docs, "retriever returned nothing"
    for d in docs:
        print("---", d.page_content[:120].replace("\n", " "))
    print(f"rag.py smoke test OK: {len(docs)} chunk(s) retrieved")
