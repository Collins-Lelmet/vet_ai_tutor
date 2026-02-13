from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# 1. Load embeddings (must match ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load existing FAISS vector store
db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Create retriever
retriever = db.as_retriever(search_kwargs={"k": 5})

# 4. Load LLM
llm = OllamaLLM(model="gemma:2b", temperature=0.2)

# 5. Question loop
while True:
    question = input("\n🐄 Question (or 'exit'): ")
    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY using the information below.
If the answer is not contained in the text, say "Not found in provided material."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    print("\n📘 Answer:\n", response)
