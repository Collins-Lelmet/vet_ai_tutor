from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chunk_documents import chunks
import os

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)

os.makedirs("vectorstore", exist_ok=True)
db.save_local("vectorstore")

print("✅ Vector database built and saved")
