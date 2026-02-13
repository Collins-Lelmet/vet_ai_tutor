import os
import re
from typing import List, Tuple
from vet_ai_tutor.config import VECTOR_PATH, EMBEDDING_MODEL, RETRIEVER_K

try:
    # prefer the same imports used elsewhere in the repo
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


class Tutor:
    """Small wrapper that centralizes embeddings, vectorstore, retriever and LLM calls.

    The `llm` passed in should expose an `invoke(prompt: str) -> str` method.
    If the retriever provides `get_relevant_documents` it will be used; otherwise
    `invoke(question)` is attempted to support older code paths.
    """
    def __init__(self, llm, vector_path: str = None, embedding_model: str = None):
        self.vector_path = vector_path or VECTOR_PATH
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.llm = llm
        # conversation history as list of (role, content) tuples
        self.history: List[tuple] = []

        os.makedirs(self.vector_path, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # load FAISS index saved at ingest time
        self.db = FAISS.load_local(
            self.vector_path, self.embeddings, allow_dangerous_deserialization=True
        )

    def ask(self, question: str, k: int = None, remember: bool = True) -> Tuple[str, List]:
        k = k or RETRIEVER_K
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        # quick rule: handle simple greetings locally
        if re.match(r"^(hi|hello|hey|good\s(morning|afternoon|evening))\b", question.strip().lower()):
            reply = "Hello! I'm your Vet AI Tutor — ask me a veterinary question or say what you'd like to learn."
            if remember:
                self.history.append(("user", question))
                self.history.append(("assistant", reply))
            return reply, []

        if remember:
            self.history.append(("user", question))

        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(question)
        else:
            raise RuntimeError("Retriever does not support known retrieval API")

        context = "\n\n".join(d.page_content for d in docs)

        # Build a prompt that includes brief conversation history for natural chat
        history_text = "\n".join(f"{r}: {c}" for r, c in self.history[-10:])

        if context.strip():
            prompt = f"""
System: You are a veterinary tutor. Use the context when answering veterinary questions.
{history_text}

Context:
{context}

Question:
{question}

Answer ONLY using the information in Context. If not in context, reply "Not found in provided material."
"""
        else:
            prompt = f"""
System: You are a friendly veterinary tutor. Use conversation history for chatty replies.
{history_text}

Question:
{question}

Answer succinctly and helpfully. If unsure, say "I don't know.".
"""

        # expect adapter to expose invoke(prompt) -> str
        if not hasattr(self.llm, "invoke"):
            raise RuntimeError("LLM passed to Tutor must expose an `invoke(prompt)` method")

        answer = self.llm.invoke(prompt)
        if remember:
            self.history.append(("assistant", answer))
        return answer, docs

    def reset_history(self):
        self.history = []

    def get_history(self) -> List[tuple]:
        return list(self.history)
