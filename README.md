# 🩺 Medical Encyclopedia RAG Chatbot

An interactive, context-aware AI chatbot designed to answer complex medical queries accurately by leveraging **Retrieval-Augmented Generation (RAG)**. Built using **LangChain**, **Groq LLM (Llama 3)**, **FAISS Vector Store**, and **Streamlit**.

🌐 **Live Demo:** [Click Here to Access the Live App](https://medical-rag-chatbot-vmwao9fe8te3v4hmsg86n8.streamlit.app/)

---

## 🚀 Key Features

- **Context-Grounded QA:** Uses RAG to eliminate AI hallucinations and provide reliable answers strictly based on medical reference literature.
- **Fast Similarity Search:** Powered by FAISS Vector Database for rapid document retrieval.
- **Source Attribution:** Includes an expandable *"View Source Documents"* feature so users can verify reference sources for every response.
- **High-Performance LLM:** Integrated with Groq API (Llama 3) for near-instant inference speed.
- **Interactive UI:** Clean and responsive web interface built with Streamlit.

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3.10+
- **LLM & RAG Framework:** LangChain, Groq API (Llama 3)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Frontend / Web UI:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```text
Medical_Chatbot/
├── data/
│   └── medical.pdf               # Source medical encyclopedia document
├── vectorstore/
│   └── db/
│       ├── index.faiss           # FAISS index file
│       └── index.pkl             # Serialized index metadata
├── app.py                        # Streamlit UI & main application code
├── create_memory_for_llm.py      # Script to process PDF and create FAISS vector store
├── connect_memory_with_llm.py    # Pipeline logic connecting FAISS and Groq LLM
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
