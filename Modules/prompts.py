# prompts.py
from __future__ import annotations
from typing import Literal
from langchain.prompts import ChatPromptTemplate
AnswerType = Literal["Concise", "Detailed"]

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

def get_prompt(level: str, answer_type: AnswerType, textbook: str) -> ChatPromptTemplate:
    lvl = (level or "Intermediate").title()
    concise = (answer_type == "Concise")
    if lvl == "Beginner": tpl = _BEGINNER_CONCISE if concise else _BEGINNER_DETAILED
    elif lvl == "Advanced": tpl = _ADVANCED_CONCISE if concise else _ADVANCED_DETAILED
    else: tpl = _INTERMEDIATE_CONCISE if concise else _INTERMEDIATE_DETAILED
    return ChatPromptTemplate.from_template(tpl).partial(textbook=textbook)
