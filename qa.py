# qa.py
from __future__ import annotations
from typing import Any, List, Optional, Literal
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage, Document

from retrieval import make_retrievers, top_context_from_sources

# ----- types -----
AnswerType = Literal["Concise", "Detailed"]
RouteType  = Literal["reuse", "memory", "textbook"]

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
_INTERMEDIATE_CONCISE = """Provide a concise answer to someone somewhat familiar with for {textbook} in less than 4 sentences.
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

from langchain.prompts import ChatPromptTemplate

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


# ----- reuse-from-logs (TF-IDF) -----
def reuse_answer_from_logs(
    logs: list,
    question: str,
    level: str,
    textbook: str,
    answer_type: AnswerType,
    *,
    threshold: float = 0.75,
) -> Optional[str]:
    def prep(t: str) -> str: return re.sub(r"[^\w\s]", "", (t or "").lower()).strip()
    new_q = prep(question); qs, ans = [], []
    for session in logs or []:
        for e in session.get("interactions", []):
            same = (("Concise" in e.get("option","") and answer_type == "Concise")
                    or ("Detailed" in e.get("option","") and answer_type == "Detailed"))
            if e.get("experience_level")==level and e.get("textbook")==textbook and same:
                qs.append(prep(e.get("question",""))); ans.append(e.get("answer",""))
    if not qs: return None
    vec = TfidfVectorizer().fit(qs + [new_q]); M = vec.transform(qs + [new_q])
    sims = cosine_similarity(M[-1], M[:-1])[0]; i = sims.argmax()
    return ans[i] if sims[i] >= threshold else None


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
    logs: list,
    openai_api_key: str,
) -> QAResult:
    ans_type: AnswerType = "Concise" if ("Concise" in option or "Voice" in option) else "Detailed"

    # 1) reuse path
    reused = reuse_answer_from_logs(logs, raw_question, level, textbook, ans_type)
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
