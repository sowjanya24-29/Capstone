"""Zepto policy assistant: local embeddings, LangGraph routing, mock LLM.

MOCK_LLM defaults to 1. That path is deterministic and makes no LLM network
call. Set MOCK_LLM=0 to use the optional Groq path instead.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
CHROMA_PATH = ROOT / "chroma"
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL = "all-MiniLM-L6-v2"

POLICY_KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)

PROMPT_TEMPLATE = """
Role: You are Zepto's policy support assistant. You answer only from the policy excerpts you are given.

Context:
{context}

Task: Answer the customer query using only the context above.
Query: {query}

Format: Return a single JSON object with keys "answer" (string), "sources" (list of document ids drawn from the context), and "confidence" (number from 0 to 1). Do not wrap it in markdown.

Length: Keep the answer under 80 words.

Negative constraint: Do not answer using information not present in the provided context. If the context is not enough, say so in the answer and use an empty sources list.

Few-shot example:
Query: Can I cancel after the order is packed?
Context: [doc_05] Once an order has been packed, it can no longer be cancelled through the app.
JSON: {{"answer": "No. Once the order is packed it can no longer be cancelled in the app.", "sources": ["doc_05"], "confidence": 0.9}}
""".strip()

MOCK_DIRECT_ANSWER = "I can only answer questions about Zepto policies right now."


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    answer: str
    sources: list[str]
    confidence: float


def mock_llm_enabled() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"


_collection = None


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def ingest() -> int:
    """One chunk per policy file. Re-ingests when the collection is short."""
    collection = get_collection()
    files = sorted(DOCS.glob("doc_*.txt"))
    if len(files) != 8:
        raise RuntimeError(f"Expected 8 policy documents, found {len(files)}")
    if collection.count() >= 8:
        return collection.count()
    ids = []
    documents = []
    for path in files:
        ids.append(path.stem)
        documents.append(path.read_text(encoding="utf-8").strip())
    collection.add(ids=ids, documents=documents)
    return collection.count()


def classify_intent(state: GraphState) -> GraphState:
    query = state["query"].lower()
    if mock_llm_enabled():
        intent = (
            "policy_question"
            if any(keyword in query for keyword in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        raw = _generate_with_retry(
            "Classify the query as policy_question or general_question. "
            "Reply with JSON {\"intent\": \"policy_question\"} or "
            "{\"intent\": \"general_question\"} only.\n"
            f"Query: {state['query']}"
        )
        intent = "policy_question" if "policy_question" in raw.lower() else "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    collection = get_collection()
    found = collection.query(query_texts=[state["query"]], n_results=3)
    ids = found["ids"][0]
    documents = found["documents"][0]
    snippet = documents[0][:200]
    context = "\n".join(f"[{doc_id}] {text}" for doc_id, text in zip(ids, documents))
    if mock_llm_enabled():
        answer = f"Based on the retrieved context: {snippet}"
        return {"answer": answer, "sources": ids, "confidence": 1.0}
    prompt = PROMPT_TEMPLATE.format(context=context, query=state["query"])
    parsed = _answer_with_schema(prompt, fallback_sources=ids)
    return parsed


def direct_answer(state: GraphState) -> GraphState:
    if mock_llm_enabled():
        return {"answer": MOCK_DIRECT_ANSWER, "sources": [], "confidence": 1.0}
    prompt = PROMPT_TEMPLATE.format(
        context="No policy excerpts were retrieved because this is not a policy question.",
        query=state["query"],
    )
    parsed = _answer_with_schema(prompt, fallback_sources=[])
    parsed["sources"] = []
    return parsed


def route(state: GraphState) -> str:
    if state.get("intent") == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


def ask(query: str) -> AskResponse:
    ingest()
    result = build_graph().invoke({"query": query})
    return AskResponse(
        answer=result["answer"],
        sources=list(result["sources"]),
        confidence=float(result["confidence"]),
    )


def _call_groq(prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY")
    body = json.dumps(
        {
            "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def _generate_with_retry(prompt: str) -> str:
    """Up to two extra attempts after the first call. Used by the optional real-LLM path."""
    last_error = "no attempt"
    correction = prompt
    for attempt in range(3):
        try:
            return _call_groq(correction)
        except Exception as exc:
            last_error = str(exc)
            correction = (
                prompt
                + "\nThe previous attempt failed. Return only the requested JSON. "
                + f"Failure: {last_error}"
            )
    return json.dumps(
        {
            "answer": f"Error: the model did not return a usable response ({last_error}).",
            "sources": [],
            "confidence": 0.0,
        }
    )


def _answer_with_schema(prompt: str, fallback_sources: list[str]) -> dict:
    last_error = "no attempt"
    correction = prompt
    for _ in range(3):
        raw = _call_groq(correction)
        try:
            parsed = AskResponse.model_validate_json(_extract_json(raw))
            return parsed.model_dump()
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            correction = (
                prompt
                + "\nYour previous reply failed schema validation. "
                + "Return only JSON with keys answer, sources, and confidence. "
                + f"Validation error: {last_error}"
            )
    return {
        "answer": f"Error: the model output failed validation ({last_error}).",
        "sources": fallback_sources,
        "confidence": 0.0,
    }


def _extract_json(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("response did not contain a JSON object")
    return raw[start : end + 1]
