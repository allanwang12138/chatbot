# config.py
from __future__ import annotations
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import streamlit as st


# ---------- Public data model ----------
@dataclass(frozen=True)
class Env:
    OPENAI_API_KEY: str
    QDRANT_API_KEY: str
    QDRANT_URL: str
    GITHUB_TOKEN: str | None
    GITHUB_REPO: str | None
    GITHUB_FILE_PATH: str | None


# ---------- Collections map (human name -> qdrant collection) ----------
COLLECTION_MAP: dict[str, str] = {
    "Introductory Macroeconomics": "introductory_macroeconomics_collection",
    "Introductory Microeconomics": "introductory_microeconomics_collection",
    "Statistics For Economics": "statistics_for_economics_collection",
    "MATHEMATICS Textbook for Class IX": "mathematics_textbook_for_class_ix_collection",
    "MATHEMATICS Textbook for Class X": "mathematics_textbook_for_class_x_collection",
    "MATHEMATICS Textbook for Class XI": "mathematics_textbook_for_class_xi_collection",
    "MATHEMATICS Textbook for Class XII PART I": "mathematics_textbook_for_class_xii_part_i_collection",
    "MATHEMATICS Textbook for Class XII PART II": "mathematics_textbook_for_class_xii_part_ii_collection",
}


# ---------- Environment loading ----------
@st.cache_resource(show_spinner=False)
def load_env() -> Env:
    """
    Loads .env once and returns a frozen Env object.
    Caches via Streamlit so all imports share the same values.
    """
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY", "")
    qdrant_key = os.getenv("QDRANT_API_KEY", "")
    qdrant_url = os.getenv("QDRANT_URL", "")

    if not openai_key:
        st.error("❌ Missing OPENAI_API_KEY in environment.")
    if not qdrant_url:
        st.error("❌ Missing QDRANT_URL in environment.")
    if not qdrant_key:
        st.warning("⚠️ QDRANT_API_KEY is empty. If your Qdrant is private, requests may fail.")

    return Env(
        OPENAI_API_KEY=openai_key,
        QDRANT_API_KEY=qdrant_key,
        QDRANT_URL=qdrant_url,
        GITHUB_TOKEN=os.getenv("GITHUB_TOKEN"),
        GITHUB_REPO=os.getenv("GITHUB_REPO"),
        GITHUB_FILE_PATH=os.getenv("GITHUB_FILE_PATH"),
    )
