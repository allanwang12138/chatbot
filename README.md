# 🤖 Chatbot Template with RAG + Streamlit

This project is a fully functional **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit**, **OpenAI GPT**, and **Qdrant**. It supports user-level customization, multi-subject tutoring (e.g., Macroeconomics, Microeconomics, and Statistics), and voice-based responses.

---

## 📦 Features

- 🔐 **User login** with custom experience levels (Beginner, Intermediate, Advanced)
- 📚 **Textbook-based Q&A** powered by OpenAI and Qdrant vector search
- 🎯 **Dynamic prompts** based on selected subject and user level
- 📖 **Three response modes**: Detailed, Concise, and Voice (text-to-speech)
- 🧠 **Answer reuse logic** using TF-IDF + cosine similarity
- ☁️ **Session logging** saved directly to GitHub

---

## 🗂️ Project Structure

| File/Folder                        | Purpose                                                                 |
|-----------------------------------|-------------------------------------------------------------------------|
| `streamlit_app.py`                | 🎮 Main Streamlit app with Q&A logic and UI                            |
| `upload_embeddings_to_qdrant.py` | 📤 Upload textbook PDFs to Qdrant as vector embeddings                  |
| `create credentials with experience levels.py` | 🔐 Randomly generate user credentials with experience levels     |
| `sample_credentials_with_levels.csv` | 📁 Sample credential list used for login                          |
| `logs/session_logs.json`          | 📝 Stores interaction history for each session                         |
| `requirements.txt`               | 📦 Python dependencies                                                  |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/allanwang12138/chatbot.git
cd chatbot
