import os
import re
import logging
from typing import List, Tuple
from vet_ai_tutor.config import VECTOR_PATH, EMBEDDING_MODEL, RETRIEVER_K

logger = logging.getLogger(__name__)


def _build_bm25_retriever(docs, k: int):
    """Build a BM25 retriever from a list of LangChain Documents."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed; hybrid search will use vector-only.")
        return None

    corpus = []
    for d in docs:
        text = (getattr(d, "page_content", "") or "").lower().split()
        corpus.append(text)

    if not corpus:
        return None

    bm25 = BM25Okapi(corpus)
    return bm25, docs


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

        # Lazy imports so importing this module doesn't require heavy deps at import time
        try:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except Exception:
                from langchain_community.embeddings import HuggingFaceEmbeddings
        except Exception:
            HuggingFaceEmbeddings = None
            logger.warning("HuggingFaceEmbeddings not available; embeddings operations may fail")

        try:
            from langchain_community.vectorstores import FAISS
        except Exception:
            FAISS = None
            logger.warning("FAISS vectorstore not available; vector DB operations may fail")

        if HuggingFaceEmbeddings is None or FAISS is None:
            # keep object usable for unit tests / CLI modes that don't load vectorstore
            self.embeddings = None
            self.db = None
            self._bm25_data = None
            logger.warning("Vectorstore not loaded due to missing dependencies.")
            return

        # instantiate embeddings and load FAISS index
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        except Exception as e:
            logger.exception("Failed to initialize embeddings: %s", e)
            self.embeddings = None

        try:
            self.db = FAISS.load_local(
                self.vector_path, self.embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.exception("Failed to load FAISS index: %s", e)
            self.db = None

        # Build BM25 index from all documents in the FAISS store
        self._bm25_data = None
        if self.db is not None:
            try:
                all_docs = list(self.db.docstore._dict.values())
                self._bm25_data = _build_bm25_retriever(all_docs, RETRIEVER_K)
                if self._bm25_data:
                    logger.info("BM25 index built with %d documents for hybrid search.", len(all_docs))
            except Exception as e:
                logger.debug("Could not build BM25 index: %s", e)

        # Best-effort metadata validation; don't raise, only warn
        if self.db is not None:
            try:
                retriever = self.db.as_retriever(search_kwargs={"k": 1})
                docs = []
                if hasattr(retriever, "invoke"):
                    try:
                        docs = retriever.invoke("test")
                    except Exception:
                        docs = []
                elif hasattr(retriever, "get_relevant_documents"):
                    try:
                        docs = retriever.get_relevant_documents("test")
                    except Exception:
                        docs = []

                if docs:
                    meta = getattr(docs[0], "metadata", {}) or {}
                    if not any(k in meta for k in ("source", "chunk", "doc_id")):
                        logger.warning(
                            "Vectorstore documents appear to lack expected metadata keys (source/chunk). "
                            "Consider re-ingesting documents with metadata."
                        )
            except Exception as e:
                logger.debug("Unable to validate vectorstore metadata: %s", e)

    def _hybrid_retrieve(self, question: str, k: int) -> List:
        """Combine FAISS vector search with BM25 keyword search for better recall."""
        # FAISS vector results
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        if hasattr(retriever, "invoke"):
            vector_docs = retriever.invoke(question)
        elif hasattr(retriever, "get_relevant_documents"):
            vector_docs = retriever.get_relevant_documents(question)
        else:
            raise RuntimeError("Retriever does not support known retrieval API")

        if self._bm25_data is None:
            return vector_docs

        # BM25 keyword results
        bm25, all_docs = self._bm25_data
        query_tokens = question.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        # Get top-k BM25 results
        import numpy as np
        top_indices = np.argsort(bm25_scores)[-k:][::-1]
        bm25_docs = [all_docs[i] for i in top_indices if bm25_scores[i] > 0]

        # Merge: vector docs first, then BM25 docs not already present (by content)
        seen_contents = set()
        merged = []
        for d in vector_docs:
            content = (getattr(d, "page_content", "") or "").strip()
            if content not in seen_contents:
                seen_contents.add(content)
                merged.append(d)

        for d in bm25_docs:
            content = (getattr(d, "page_content", "") or "").strip()
            if content not in seen_contents:
                seen_contents.add(content)
                merged.append(d)

        # Return top k from merged results
        return merged[:k]

    def ask(self, question: str, k: int = None, remember: bool = True) -> Tuple[str, List]:
        if self.db is None:
            raise RuntimeError("Vectorstore not loaded; cannot answer questions. Rebuild vectorstore first.")

        k = k or RETRIEVER_K
        # quick rule: handle simple greetings locally
        if re.match(r"^(hi|hello|hey|good\s(morning|afternoon|evening))\b", question.strip().lower()):
            reply = "Hello! I'm your Vet AI Tutor — ask me a veterinary question or say what you'd like to learn."
            if remember:
                self.history.append(("user", question))
                self.history.append(("assistant", reply))
            return reply, []

        if remember:
            self.history.append(("user", question))

        # Enrich vague follow-ups with topic from conversation history.
        # e.g. "How is it treated?" → "milk fever in dairy cattle How is it treated?"
        search_query = question
        if len(question.split()) < 8 and self.history:
            prev_user_msgs = [c for r, c in self.history[:-1] if r == "user"]
            if prev_user_msgs:
                search_query = prev_user_msgs[-1] + " " + question

        # Use hybrid retrieval (FAISS + BM25)
        docs = self._hybrid_retrieve(search_query, k)

        # Build a labeled context so the LLM can cite sources reliably.
        context_parts = []
        citations = []
        for i, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}
            source = meta.get("source") or meta.get("source_file") or meta.get("doc_id") or "unknown"
            chunk = meta.get("chunk") if "chunk" in meta else meta.get("chunk_index")
            label = f"[S{i}]"
            target = f"{source}#{chunk}" if chunk is not None else f"{source}"
            citations.append({"id": label, "target": target})
            text = (getattr(d, "page_content", "") or "").strip()
            context_parts.append(f"{label} {target}\n{text}")

        context = "\n\n".join(context_parts)

        # Build prompt with conversation history (last 4 for memory-constrained LLMs)
        history_text = "\n".join(f"{r}: {c}" for r, c in self.history[-4:])

        if context.strip():
            prompt = (
                "You are a veterinary tutor. Answer from the Context only. "
                "Cite sources as [S1], [S2]. If not found, say so.\n\n"
                f"History:\n{history_text}\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer concisely (under 200 words), then list SOURCES.\n\nAnswer:"
            )
        else:
            prompt = (
                "System: You are a friendly veterinary tutor. "
                "Use conversation history for chatty replies.\n\n"
                f"{history_text}\n\n"
                f"Question: {question}\n\n"
                "Answer succinctly and helpfully. If unsure, say \"I don't know.\".\n\n"
                "Answer:"
            )

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
