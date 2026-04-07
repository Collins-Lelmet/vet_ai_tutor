"""Full pipeline diagnostic for Vet AI Tutor."""
import time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("=" * 60)
print("VET AI TUTOR - FULL PIPELINE DIAGNOSTIC")
print("=" * 60)

# --- Stage 1: Imports ---
t0 = time.time()
from langchain_ollama import OllamaLLM
from vet_ai_tutor.core import Tutor
from vet_ai_tutor.llm_adapter import OllamaAdapter
from vet_ai_tutor.config import VECTOR_PATH, LLM_MODEL, EMBEDDING_MODEL, RETRIEVER_K
t1 = time.time()
print(f"\n[1] Imports: {t1-t0:.2f}s")
print(f"    Config: model={LLM_MODEL}, embedding={EMBEDDING_MODEL}, k={RETRIEVER_K}")

# --- Stage 2: LLM init ---
t2 = time.time()
base_llm = OllamaLLM(model=LLM_MODEL, num_predict=512)
adapter = OllamaAdapter(base_llm)
t3 = time.time()
print(f"[2] LLM init: {t3-t2:.2f}s")

# --- Stage 3: Tutor init (embeddings + FAISS + BM25) ---
tutor = Tutor(llm=adapter, vector_path=VECTOR_PATH)
t4 = time.time()
print(f"[3] Tutor init (embeddings+FAISS+BM25): {t4-t3:.2f}s")
print(f"    DB loaded: {tutor.db is not None}")
print(f"    BM25 loaded: {tutor._bm25_data is not None}")
if tutor.db:
    n_docs = len(tutor.db.docstore._dict)
    print(f"    Docs in FAISS: {n_docs}")

# --- Stage 4: Ask question ---
question = "What is milk fever in dairy cattle?"
print(f'\n[4] Question: "{question}"')

t5 = time.time()
answer, docs = tutor.ask(question, k=RETRIEVER_K)
t6 = time.time()

print(f"[5] Total ask() time: {t6-t5:.2f}s")
print(f"    Retrieved docs: {len(docs)}")
print(f"    Answer length: {len(answer)} chars, {len(answer.split())} words")

# --- Stage 5: Check retrieved docs ---
print(f"\n[6] Retrieved chunks:")
for i, d in enumerate(docs, 1):
    meta = getattr(d, "metadata", {}) or {}
    src = meta.get("source", "unknown")
    content_preview = d.page_content[:80].replace("\n", " ")
    print(f'    [{i}] {src} -> "{content_preview}..."')

# --- Stage 6: Answer ---
print(f"\n[7] Full Answer:")
print("-" * 40)
print(answer)
print("-" * 40)

# --- Stage 7: Second question (tests warm cache + history) ---
question2 = "How is it treated?"
print(f'\n[8] Follow-up: "{question2}"')
t7 = time.time()
answer2, docs2 = tutor.ask(question2, k=RETRIEVER_K)
t8 = time.time()
print(f"[9] Follow-up ask() time: {t8-t7:.2f}s")
print(f"    Retrieved docs: {len(docs2)}")
print(f"    Answer length: {len(answer2)} chars, {len(answer2.split())} words")
print(f"\n[10] Follow-up answer:")
print("-" * 40)
print(answer2)
print("-" * 40)

print(f"\n{'=' * 60}")
print(f"SUMMARY")
print(f"{'=' * 60}")
print(f"  One-time load:     {t4-t0:.2f}s")
print(f"  First query:       {t6-t5:.2f}s")
print(f"  Follow-up query:   {t8-t7:.2f}s")
print(f"  Model:             {LLM_MODEL}")
print(f"  Retriever K:       {RETRIEVER_K}")
