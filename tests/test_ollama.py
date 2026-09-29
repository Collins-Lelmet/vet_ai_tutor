from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def test_vectorstore_can_be_loaded():
    vector_path = "data/vectorstore" if Path("data/vectorstore/index.faiss").exists() else "vectorstore"
    if not Path(vector_path, "index.faiss").exists():
        pytest.skip("Vectorstore has not been built yet")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke("veterinary anatomy")
    assert docs
    assert any(getattr(doc, "page_content", "") for doc in docs)
