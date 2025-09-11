# models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional
from langchain.schema import Document

AnswerType = Literal["Concise", "Detailed"]
RouteType = Literal["memory", "textbook", "reuse"]

@dataclass
class QAResult:
    answer: str
    sources: List[Document]
    context_snippet: str
    route: RouteType             # 'reuse' | 'memory' | 'textbook'
    answer_type: AnswerType      # 'Concise' | 'Detailed'
    level: str                   # Beginner/Intermediate/Advanced
    textbook: str
