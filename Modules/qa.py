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

from prompts import get_prompt, AnswerType
from retrieval import make_retrievers, top_context_from_sources

RouteType = Literal["reuse", "memory", "textbook"]

@dataclass
class QAResult:
    answer: str
    sources: List[Document]
    context_snippet: str
    route: RouteType
    answer_type: AnswerType
    level: str
    textbook: str

def _coerce_text(x: Any) -> str:
    if isinstance(x, BaseMessage): return x.content or ""
    if isinstance(x, dict): return x.get("answer") or x.get("result") or x.get("output_text") or str(x)
    return str(x or "")

def _build_llm(openai_api_key: str) -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=openai_api_key)

def reuse_answer_from_logs(logs: list, question: str, level: str, textbook: str, answer_type: AnswerType, *, threshold: float = 0.75) -> Optional[str]:
    def prep(t: str) -> str: return re.sub(r"[^\w\s]", "", (t or "").lower()).strip()
    new_q = prep(question); qs, ans = [], []
    for session in logs or []:
        for e in session.get("interactions", []):
            same = (("Concise" in e.get("option","") and answer_type == "Concise") or ("Detailed" in e.get("option","") and answer_type == "Detailed"))
            if e.get("experience_level")==level and e.get("textbook")==textbook and same:
                qs.append(prep(e.get("question",""))); ans.append(e.get("answer",""))
    if not qs: return None
    vec = TfidfVectorizer().fit(qs + [new_q]); M = vec.transform(qs + [new_q]); sims = cosine_similarity(M[-1], M[:-1])[0]
    i = sims.argmax()
    return ans[i] if sims[i] >= threshold else None

def compress_answer(llm: ChatOpenAI, text: str, textbook: str, level: str) -> str:
    prompt = f"You are a tutor for {textbook} with a(n) {level} learner. Rewrite the answer in ≤2 sentences, clear and jargon-free.\n\nAnswer:\n{text}"
    return _coerce_text(llm.invoke(prompt))

def answer(
    *, raw_question: str, option: str, username: str, level: str, textbook: str,
    memory, db, memory_db, logs: list, openai_api_key: str
) -> QAResult:
    ans_type: AnswerType = "Concise" if ("Concise" in option or "Voice" in option) else "Detailed"

    reused = reuse_answer_from_logs(logs, raw_question, level, textbook, ans_type)
    if reused:
        final_text = reused
        if ans_type == "Concise":
            final_text = compress_answer(_build_llm(openai_api_key), final_text, textbook, level)
        return QAResult(final_text, [], "", "reuse", ans_type, level, textbook)

    chosen, has_mem, _ = make_retrievers(raw_question, username, textbook, level, db, memory_db)
    route: RouteType = "memory" if has_mem else "textbook"
    prompt = get_prompt(level, ans_type, textbook)
    llm = _build_llm(openai_api_key)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=chosen, memory=memory, return_source_documents=True, combine_docs_chain_kwargs={"prompt": prompt})
    result = chain.invoke({"question": raw_question, "textbook": textbook})
    text = result.get("answer") or result.get("result") or ""
    sources: List[Document] = result.get("source_documents", []) or []
    if ans_type == "Concise":
        text = compress_answer(llm, text, textbook, level)
    snippet = top_context_from_sources(sources, 1200)

    return QAResult(text, sources, snippet, route, ans_type, level, textbook)
