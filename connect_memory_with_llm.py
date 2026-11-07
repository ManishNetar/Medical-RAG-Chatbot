import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_classic import hub

from langchain_classic.chains import create_retrieval_chain

from langchain_classic.chains.combine_documents import create_stuff_documents_chain


from dotenv import load_dotenv
load_dotenv()


# Step 1: Setup Groq LLM

api_key_val = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "openai/gpt-oss-20b"  # Change to any supported Groq model

llm = ChatGroq(
    model = GROQ_MODEL_NAME,
    temperature = 0.5,
    max_tokens = 512,
    api_key = api_key_val,
)

# Step 2: Connect LLM with FAISS and Create chain

# Load Database
vector_store_path = "vectorstore/db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# Step 3: Build RAG chain
from langchain_core.prompts import ChatPromptTemplate

# retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat", dangerously_pull_public_prompt=True)

# system_prompt = (
#     "Use the following pieces of retrieved context to answer the question. "
#     "If you don't know the answer, say that you don't know.\n\n"
#     "{context}"
# )

system_prompt = (
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Provide clear, well-formatted plain markdown bullet points instead of complex tables with HTML tags.\n\n"
    "{context}"
)

# # Document combiner chain (stuff documents into prompt)
retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
# Retrieval chain (retriever + doc combiner)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)




# Now invoke with a single query
user_query=input("Write Query Here: ")
response=rag_chain.invoke({'input': user_query})
print("RESULT: ", response["answer"])
print("\nSOURCE DOCUMENTS:")
for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")