import streamlit as st
import openai
import os
from dotenv import load_dotenv
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient

# Load secrets
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

openai.api_key = OPENAI_API_KEY
COLLECTION_NAME = "macroecon_collection"

# Embeddings and Qdrant connection
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Connect to existing collection
db = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings  # ✅ Not embedding_function anymore
)

# Prompt templates
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

# Streamlit UI
st.title("📄 Macro Economics Q&A App")

query = st.text_input("Ask a question about Macro Economics:")

col1, col2, col3 = st.columns(3)
with col1:
    detailed_clicked = st.button("📖 Detailed Answer")
with col2:
    concise_clicked = st.button("✂️ Concise Answer")
with col3:
    voice_clicked = st.button("🔊 Voice Answer")

option = None
if detailed_clicked:
    option = "Detailed Answer"
elif concise_clicked:
    option = "Concise Answer"
elif voice_clicked:
    option = "Concise Answer + Voice"

if query and option:
    with st.spinner("Searching context..."):
        docs = db.similarity_search_with_score(query, k=5)
        context_text = "\n\n".join([doc.page_content for doc, _ in docs])

    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_DETAILED if option == "Detailed Answer" else PROMPT_CONCISE
    )
    prompt = prompt_template.format(context=context_text, question=query)
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    with st.spinner("Generating answer..."):
        response = model.predict(prompt)

    if "Voice" in option:
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
    else:
        st.markdown("### Answer")
        st.write(response)

    with st.expander("Show Retrieved Context"):
        st.write(context_text)
