import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Environment variables load karein
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Medical Chatbot", page_icon="💬", layout="centered")

# App Header (Image design ke according)
st.title("Ask Chatbot!")

# Streamlit Cache ka use karke Vectorstore load karein taaki app fast chale
@st.cache_resource(show_spinner="Running get_vectorstore().")
def get_vectorstore():
    vector_store_path = "vectorstore/db"
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        vector_store_path, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})

# Streamlit Cache ka use karke RAG Chain initialize karein
@st.cache_resource(show_spinner="Initializing LLM Chain...")
def get_rag_chain(_retriever):
    GROQ_MODEL_NAME = "openai/gpt-oss-20b"
    
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0.5,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    system_prompt = (
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )

    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    return create_retrieval_chain(_retriever, combine_docs_chain)


# Retriever aur RAG Chain load karein
retriever = get_vectorstore()
rag_chain = get_rag_chain(retriever)

# Chat History ko session state mein maintain karein
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purane messages screen par display karein
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Box
if prompt := st.chat_input("Ask your medical query here..."):
    # User message UI mein dikhayein aur state mein save karein
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Response generate karein
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            
            # Answer display karein
            st.markdown(answer)
            
            # Source documents (Accordion dropdown me show hongi)
            with st.expander("View Source Documents"):
                for doc in response["context"]:
                    st.write(f"**Page {doc.metadata.get('page', 'N/A')}:**")
                    st.caption(doc.page_content[:250] + "...")

    # Assistant response ko chat history mein save karein
    st.session_state.messages.append({"role": "assistant", "content": answer})