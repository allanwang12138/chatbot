import streamlit as st
import openai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
import os
from langchain.prompts import ChatPromptTemplate
from pydub import AudioSegment
from pydub.playback import play
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Load environment variables or set directly

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# CHROMA_PATH = "/Users/wuxiong/Desktop/Research/RAG_econ" # your existing db path
CHROMA_PATH = "/workspaces/chatbot/chroma_db" 

# Load vector store and embedding model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db_chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Define prompt templates
PROMPT_DETAILED = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

PROMPT_CONCISE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a clear and concise summary in no more than 2 sentences.
"""

st.title("📄 Macro Economics Q&A App")

query = st.text_input("Ask a question about Macro Economics:")

# Display options as horizontal buttons
col1, col2, col3 = st.columns(3)
with col1:
    detailed_clicked = st.button("📖 Detailed Answer")
with col2:
    concise_clicked = st.button("✂️ Concise Answer")
with col3:
    voice_clicked = st.button("🔊 Voice Answer")

# Determine selected option
option = None
if detailed_clicked:
    option = "Detailed Answer"
elif concise_clicked:
    option = "Concise Answer"
elif voice_clicked:
    option = "Concise Answer + Voice"

# Handle query and response if an option is chosen
if query and option:
    docs_chroma = db_chroma.similarity_search_with_score(query, k=5)
    context_text = "\n\n".join([doc.page_content for doc, _ in docs_chroma])

    if option == "Detailed Answer":
        prompt_template = ChatPromptTemplate.from_template(PROMPT_DETAILED)
    else:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_CONCISE)

    prompt = prompt_template.format(context=context_text, question=query)
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    with st.spinner("Generating answer..."):
        response = model.predict(prompt)

    # Output handling
    if option in ["Detailed Answer", "Concise Answer"]:
        st.markdown("### Answer")
        st.write(response)
    elif option == "Concise Answer + Voice":
        with st.spinner("Generating voice..."):
            speech_response = openai.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=response
            )
            audio_path = "output.mp3"
            with open(audio_path, "wb") as f:
                f.write(speech_response.read())
            audio_file = open(audio_path, "rb")
            st.audio(audio_file.read(), format="audio/mp3")

    with st.expander("Show Retrieved Context"):
        st.write(context_text)