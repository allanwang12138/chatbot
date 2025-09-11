# voice_speech.py
from __future__ import annotations
import io
import streamlit as st
import openai

DEFAULT_TTS_MODEL = "tts-1"

def synthesize(text: str, voice: str = "alloy", *, model: str = DEFAULT_TTS_MODEL) -> bytes:
    if not text: return b""
    resp = openai.audio.speech.create(model=model, voice=voice, input=text)
    return resp.read()  # MP3 bytes

def play_in_streamlit(audio_bytes: bytes, *, format: str = "audio/mp3") -> None:
    if not audio_bytes:
        st.info("No audio generated.")
        return
    st.audio(io.BytesIO(audio_bytes).read(), format=format)
