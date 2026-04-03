from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chunk_documents import chunks
import os

# Use the canonical VECTOR_PATH from project config to avoid mismatched locations
from vet_ai_tutor.config import VECTOR_PATH

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)

os.makedirs(VECTOR_PATH, exist_ok=True)
db.save_local(VECTOR_PATH)

print(f"✅ Vector database built and saved to {VECTOR_PATH}")
