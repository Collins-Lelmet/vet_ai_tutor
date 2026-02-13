from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Sample veterinary knowledge
docs = [
    Document(page_content="Rabies is a fatal viral zoonotic disease affecting the central nervous system."),
    Document(page_content="Foot and mouth disease is a highly contagious viral disease of cloven-hoofed animals."),
    Document(page_content="Brucellosis causes reproductive failure such as abortions in livestock."),
]

# Create FAISS index
vectorstore = FAISS.from_documents(docs, embeddings)

# Query
query = "Which disease causes abortions in cattle?"
results = vectorstore.similarity_search(query, k=1)

print("Question:", query)
print("Retrieved context:")
print(results[0].page_content)
