from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

VECTOR_PATH = "data/vectorstore"

# Load embeddings (must match ingest_doc.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vectorstore (NO rereading raw files)
vectorstore = FAISS.load_local(
    VECTOR_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = OllamaLLM(
    model="gemma:2b",
    temperature=0.2
)


def ask_tutor(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY using the information below.
If the answer is not contained in the text, say "Not found in provided material."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response
