from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


# step 1  [load data]
loader = DirectoryLoader(
    path = 'data',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
    )

documents = loader.load()
#print(documents[10].page_contetnt)


# step 2  [create chunks]
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
    )

chunks = splitter.split_documents(documents)
# print(chunks)


# step 3  [create vector embedding]
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Creating FAISS Vector Database... (Isme thoda time lag sakta hai)")
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding = embedding_model
    )
print("Vector DB successfully created!")

vector_store_path = "vectorstore/db"
vector_store.save_local(vector_store_path)