from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
DATA_PATH = "data/raw_notes"
VECTOR_PATH = "data/vectorstore"

os.makedirs(VECTOR_PATH, exist_ok=True)

documents = []

for file in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, file)

    if file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    elif file.endswith(".txt"):
        loader = TextLoader(path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(VECTOR_PATH)

print("✅ Vectorstore created successfully")
