# qa.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import re
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage, Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import QAResult, AnswerType
from retrieval import make_retrievers, top_context_from_sources


# ---------- Public: build a default LLM ----------
def build_llm(openai_api_key: str) -> ChatOpenAI:
    # Keep default model/settings minimal; callers can replace if needed.
    return ChatOpenAI(openai_api_key=openai_api_key)


# ---------- Reuse-from-logs (TF-IDF), mirrors your current behavior ----------
def reuse_answer_from_logs(
    logs: list,
    question: str,
    level: str,
    textbook: str,
    answer_type: AnswerType,
    threshold: float = 0.75,
) -> Optional[str]:
    def preprocess(t: str) -> str:
        return re.sub(r"[^\w\s]", "", (t or "").lower()).strip()

    new_q = preprocess(question)
    questions, answers = [], []

    for session in logs or []:
        for entry in session.get("interactions", []):
            is_same_type = (
                ("Concise" in entry.get("option", "") and answer_type == "Concise") or
                ("Detailed" in entry.get("option", "") and answer_type == "Detailed")
            )
            if (
                entry.get("experience_level") == level and
                entry.get("textbook") == textbook and
                is_same_type
            ):
                questions.append(preprocess(entry.get("question", "")))
                answers.append(entry.get("answer", ""))

    if not questions:
        return None

    vec = TfidfVectorizer().fit(questions + [new_q])
    mat = vec.transform(questions + [new_q])
    sims = cosine_similarity(mat[-1], mat[:-1])[0]
    best = sims.argmax()
    return answers[best] if sims[best] >= threshold else None


# ---------- Prompt selection (embedded for now; can move to prompts.py later) ----------
def _prompt_for(level: str, answer_type: AnswerType, textbook: str) -> ChatPromptTemplate:
    # Your original six templates (abridged only for placement; same tone/limits)
    if answer_type == "Concise":
        if level == "Beginner":
            tpl = """
You are helping a beginner understand {textbook}. Give a short, friendly answer using very simple words — no more than 4 sentences.
Context:
{context}
Question:
{question}
Answer:
"""
        elif level == "Advanced":
            tpl = """
You're replying to an advanced student in {textbook}. Answer in ≤4 sentences, assume familiarity with terms.
Context:
{context}
Question:
{question}
Answer:
"""
        else:
            tpl = """
You are a tutor for {textbook}. Provide a clear, concise answer in ≤4 sentences.
Context:
{context}
Question:
{question}
Answer:
"""
    else:  # Detailed
        if level == "Beginner":
            tpl = """
You are a patient tutor for {textbook}. Explain in clear, simple language (≤20 sentences), no jargon.
Context:
{context}
Question:
{question}
Answer:
"""
        elif level == "Advanced":
            tpl = """
You are an expert tutor for {textbook}. Provide a rigorous, focused answer in ≤20 sentences.
Context:
{context}
Question:
{question}
Answer:
"""
        else:
            tpl = """
You are a tutor for {textbook}. Give a clear, informative answer in ≤20 sentences.
Context:
{context}
Question:
{question}
Answer:
"""
    return ChatPromptTemplate.from_template(tpl).partial(textbook=textbook)


def _coerce_text(x: Any) -> str:
    if isinstance(x, BaseMessage):
        return x.content or ""
    if isinstance(x, dict):
        return x.get("answer") or x.get("result") or x.get("output_text") or str(x)
    return str(x or "")


# ---------- Optional compression pass for Concise/Voice ----------
def compress_answer(llm: ChatOpenAI, text: str, textbook: str, level: str) -> str:
    prompt = (
        f"You are a tutor for {textbook} with a(n) {level} learner. "
        f"Rewrite the answer below in no more than 2 sentences, clear and direct, "
        f"no equations or jargon.\n\nAnswer:\n{text}"
    )
    out = llm.invoke(prompt)
    return _coerce_text(out)


# ---------- Core orchestration ----------
def answer(
    *,
    raw_question: str,
    option: str,                         # "Detailed Answer" | "Concise Answer" | "Concise Answer + Voice"
    username: str,
    level: str,
    textbook: str,
    memory,                              # ConversationBufferMemory (from st.session_state.buffer_memory)
    db,                                  # textbook langchain_qdrant store
    memory_db,                           # user_memory langchain_qdrant store
    logs: list,
    openai_api_key: str,
) -> QAResult:
    """
    Returns a QAResult without mutating session state. Caller can log and update memory as desired.
    """
    # Decide concise vs detailed
    answer_type: AnswerType = "Concise" if ("Concise" in option or "Voice" in option) else "Detailed"

    # 1) Try reuse-from-logs
    reused = reuse_answer_from_logs(logs, raw_question, level, textbook, answer_type)
    if reused:
        final_text = reused
        # Optional: compress reused answer if concise/voice is requested
        if answer_type == "Concise":
            llm = build_llm(openai_api_key)
            final_text = compress_answer(llm, final_text, textbook, level)

        return QAResult(
            answer=final_text,
            sources=[],                                 # reuse path has no fresh sources
            context_snippet="",
            route="reuse",
            answer_type=answer_type,
            level=level,
            textbook=textbook,
        )

    # 2) Build retrievers and choose route
    chosen_retriever, has_memory_match, _ = make_retrievers(
        query=raw_question,
        username=username,
        textbook=textbook,
        level=level,
        db=db,
        memory_db=memory_db,
    )
    route = "memory" if has_memory_match else "textbook"

    # 3) Select prompt
    prompt = _prompt_for(level, answer_type, textbook)

    # 4) Run ConversationalRetrievalChain
    llm = build_llm(openai_api_key)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=chosen_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    result = chain.invoke({"question": raw_question, "textbook": textbook})
    text = result.get("answer") or result.get("result") or ""
    sources: List[Document] = result.get("source_documents", []) or []

    # 5) Optional compress for Concise/Voice
    if answer_type == "Concise":
        text = compress_answer(llm, text, textbook, level)

    # 6) Context snippet for logging/UX
    snippet = top_context_from_sources(sources, max_chars=1200)

    return QAResult(
        answer=text,
        sources=sources,
        context_snippet=snippet,
        route=route,
        answer_type=answer_type,
        level=level,
        textbook=textbook,
    )
