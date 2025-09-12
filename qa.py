# qa.py
from __future__ import annotations
from typing import Any, List, Optional, Literal
import re
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage, Document
from langchain.prompts import ChatPromptTemplate

from retrieval import make_retrievers, top_context_from_sources, is_in_scope

# ----- types -----
AnswerType = Literal["Concise", "Detailed"]
RouteType  = Literal["reuse", "memory", "textbook", "oos"]

@dataclass
class QAResult:
    answer: str
    sources: List[Document]
    context_snippet: str
    route: RouteType
    answer_type: AnswerType
    level: str
    textbook: str


# ----- prompt templates (inlined) -----
_BEGINNER_DETAILED = """You are a patient tutor for {textbook}. Explain clearly to someone new to this subject in ≤20 sentences, no jargon.
Context:
{context}
Question:
{question}
Answer:
"""
_BEGINNER_CONCISE = """You are helping a beginner with {textbook}. Answer in ≤4 simple sentences.
Context:
{context}
Question:
{question}
Answer:
"""
_INTERMEDIATE_DETAILED = """You are an experienced tutor for {textbook}. Explain to someone somewhat familiar with this subject in ≤20 sentences.
Context:
{context}
Question:
{question}
Answer:
"""
_INTERMEDIATE_CONCISE = """Provide a concise answer to someone somewhat familiar with {textbook} in less than 4 sentences.
Context:
{context}
Question:
{question}
Answer:
"""
_ADVANCED_DETAILED = """You are an expert tutor for {textbook}. Provide a rigorous, focused answer in ≤20 sentences.
Context:
{context}
Question:
{question}
Answer:
"""
_ADVANCED_CONCISE = """Reply to an advanced student in {textbook} in ≤4 sentences.
Context:
{context}
Question:
{question}
Answer:
"""

def _get_prompt(level: str, answer_type: AnswerType, textbook: str) -> ChatPromptTemplate:
    lvl = (level or "Intermediate").title()
    concise = (answer_type == "Concise")
    if lvl == "Beginner":
        tpl = _BEGINNER_CONCISE if concise else _BEGINNER_DETAILED
    elif lvl == "Advanced":
        tpl = _ADVANCED_CONCISE if concise else _ADVANCED_DETAILED
    else:
        tpl = _INTERMEDIATE_CONCISE if concise else _INTERMEDIATE_DETAILED
    return ChatPromptTemplate.from_template(tpl).partial(textbook=textbook)


# ----- llm + helpers -----
def _coerce_text(x: Any) -> str:
    if isinstance(x, BaseMessage): return x.content or ""
    if isinstance(x, dict): return x.get("answer") or x.get("result") or x.get("output_text") or str(x)
    return str(x or "")

def _build_llm(openai_api_key: str) -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=openai_api_key)


# ----- vector-based reuse from user_memory (Qdrant only) -----
def _extract_answer_from_doc_text(text: str) -> str:
    """Our memory docs are stored as 'Q: ...\\nA: ...'. Pull just the answer if present."""
    if not text:
        return ""
    m = re.search(r"\bA:\s*(.*)\Z", text, flags=re.DOTALL)
    return (m.group(1).strip() if m else text.strip())

def vector_reuse_from_memory(
    question: str,
    username: str,
    textbook: str,
    level: str,
    memory_db,
    *,
    k: int = 1,
    similarity_threshold: float = 0.78,  # tune 0.68–0.82 for your data
) -> Optional[str]:
    """Return a reused answer from user_memory if the nearest doc is similar enough."""
    user_filter = {
        "username": username,
        "textbook": textbook,
        "experience_level": level,
    }
    try:
        hits = memory_db.similarity_search_with_score(question, k=k, filter=user_filter)
    except Exception:
        return None

    if not hits:
        return None

    # Qdrant with cosine distance returns distance in [0,2]; LC normalizes to [0,1].
    # We convert to similarity ≈ 1 - distance.
    doc, score = hits[0]
    try:
        sim = 1.0 - float(score)
    except Exception:
        # If already similarity, use it; else treat as low sim
        sim = float(score) if isinstance(score, (int, float)) else 0.0

    if sim >= similarity_threshold:
        return _extract_answer_from_doc_text(getattr(doc, "page_content", "") or "")

    return None


# ----- optional compress step for Concise/Voice -----
def compress_answer(llm: ChatOpenAI, text: str, textbook: str, level: str) -> str:
    prompt = (
        f"You are a tutor for {textbook} with a(n) {level} learner. "
        f"Rewrite the answer in ≤2 sentences, clear and jargon-free.\n\nAnswer:\n{text}"
    )
    return _coerce_text(llm.invoke(prompt))


# ----- main entry point -----
def answer(
    *,
    raw_question: str,
    option: str,                         # "Detailed Answer" | "Concise Answer" | "Concise Answer + Voice"
    username: str,
    level: str,
    textbook: str,
    memory,                              # ConversationBufferMemory
    db,                                  # textbook LC Qdrant store
    memory_db,                           # user_memory LC Qdrant store
    logs: list,                          # kept for compatibility; ignored now
    openai_api_key: str,
) -> QAResult:
    ans_type: AnswerType = "Concise" if ("Concise" in option or "Voice" in option) else "Detailed"

    # 0) Subject-scope gate (block early)
    in_scope, _ = is_in_scope(raw_question, textbook, db)
    if not in_scope:
        msg = (
            f"This question looks outside the scope of **{textbook}**. "
            f"Please ask about topics covered in this textbook."
        )
        if ans_type == "Concise":
            msg = f"Outside **{textbook}**. Please ask about topics from this textbook."
        return QAResult(
            answer=msg,
            sources=[],
            context_snippet="",
            route="oos",
            answer_type=ans_type,
            level=level,
            textbook=textbook,
        )

    # 1) vector-based reuse from user_memory (Qdrant only)
    reused = vector_reuse_from_memory(raw_question, username, textbook, level, memory_db)
    if reused:
        final_text = reused
        if ans_type == "Concise":
            final_text = compress_answer(_build_llm(openai_api_key), final_text, textbook, level)
        return QAResult(final_text, [], "", "reuse", ans_type, level, textbook)

    # 2) choose retriever/route
    chosen, has_mem, _ = make_retrievers(raw_question, username, textbook, level, db, memory_db)
    route: RouteType = "memory" if has_mem else "textbook"

    # 3) prompt + chain
    prompt = _get_prompt(level, ans_type, textbook)
    llm = _build_llm(openai_api_key)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=chosen,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    result = chain.invoke({"question": raw_question, "textbook": textbook})
    text = result.get("answer") or result.get("result") or ""
    sources: List[Document] = result.get("source_documents", []) or []

    # 4) compress if needed
    if ans_type == "Concise":
        text = compress_answer(llm, text, textbook, level)

    # 5) context snippet
    snippet = top_context_from_sources(sources, 1200)

    return QAResult(text, sources, snippet, route, ans_type, level, textbook)
