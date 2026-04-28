"""Full assessment of the Vet AI Tutor app."""
import time, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

results = {}

print("=" * 65)
print("VET AI TUTOR - FULL APP ASSESSMENT")
print("=" * 65)

# =====================================================
# TEST 1: Config integrity
# =====================================================
print("\n[TEST 1] Config integrity")
try:
    from vet_ai_tutor.config import (
        VECTOR_PATH, EMBEDDING_MODEL, LLM_MODEL, RETRIEVER_K,
        CHUNK_SIZE, CHUNK_OVERLAP, SKIP_EXTENSIONS, SUPPORTED_EXTENSIONS
    )
    print(f"  VECTOR_PATH:    {VECTOR_PATH}")
    print(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"  LLM_MODEL:      {LLM_MODEL}")
    print(f"  RETRIEVER_K:    {RETRIEVER_K}")
    print(f"  CHUNK_SIZE:     {CHUNK_SIZE}")
    print(f"  CHUNK_OVERLAP:  {CHUNK_OVERLAP}")
    print(f"  SUPPORTED_EXTENSIONS: {len(SUPPORTED_EXTENSIONS)} types")
    results["config"] = "PASS"
except Exception as e:
    print(f"  FAIL: {e}")
    results["config"] = f"FAIL: {e}"

# =====================================================
# TEST 2: Vectorstore load
# =====================================================
print("\n[TEST 2] Vectorstore load")
t0 = time.time()
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_PATH, emb, allow_dangerous_deserialization=True)
    n_docs = len(db.docstore._dict)
    t1 = time.time()
    print(f"  Loaded {n_docs} chunks in {t1-t0:.1f}s")
    if n_docs > 1000:
        results["vectorstore"] = f"PASS ({n_docs} chunks)"
    else:
        results["vectorstore"] = f"WARN: only {n_docs} chunks (expected 30k+)"
except Exception as e:
    t1 = time.time()
    print(f"  FAIL: {e}")
    results["vectorstore"] = f"FAIL: {e}"

# =====================================================
# TEST 3: Embedding dimension check
# =====================================================
print("\n[TEST 3] Embedding dimension consistency")
try:
    test_vec = emb.embed_query("test")
    dim = len(test_vec)
    # Check FAISS index dimension
    faiss_dim = db.index.d
    match = dim == faiss_dim
    print(f"  Embedding dim: {dim}, FAISS index dim: {faiss_dim}, Match: {match}")
    results["embedding_dim"] = "PASS" if match else f"FAIL: {dim} vs {faiss_dim}"
except Exception as e:
    print(f"  FAIL: {e}")
    results["embedding_dim"] = f"FAIL: {e}"

# =====================================================
# TEST 4: BM25 index build
# =====================================================
print("\n[TEST 4] BM25 index build")
t2 = time.time()
try:
    from rank_bm25 import BM25Okapi
    all_docs = list(db.docstore._dict.values())
    corpus = [d.page_content.lower().split() for d in all_docs]
    bm25 = BM25Okapi(corpus)
    t3 = time.time()
    print(f"  Built BM25 index over {len(corpus)} docs in {t3-t2:.1f}s")
    results["bm25"] = "PASS"
except Exception as e:
    t3 = time.time()
    print(f"  FAIL: {e}")
    results["bm25"] = f"FAIL: {e}"

# =====================================================
# TEST 5: FAISS retrieval quality
# =====================================================
print("\n[TEST 5] FAISS retrieval quality")
test_questions = [
    ("What is milk fever in dairy cattle?", ["milk fever", "hypocalc", "parturient"]),
    ("Describe the anatomy of the rumen", ["rumen", "stomach", "ferment"]),
    ("What are the symptoms of East Coast Fever?", ["theileria", "east coast", "tick"]),
]
retrieval_results = []
for q, keywords in test_questions:
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(q)
    combined = " ".join(d.page_content.lower() for d in docs)
    hits = sum(1 for kw in keywords if kw in combined)
    score = hits / len(keywords)
    retrieval_results.append(score)
    status = "PASS" if score >= 0.5 else "WEAK"
    print(f"  Q: '{q[:50]}...' -> {hits}/{len(keywords)} keywords found [{status}]")

avg_retrieval = sum(retrieval_results) / len(retrieval_results) if retrieval_results else 0
results["retrieval"] = f"{'PASS' if avg_retrieval >= 0.5 else 'WEAK'} (avg {avg_retrieval:.0%})"

# =====================================================
# TEST 6: Full Tutor pipeline (LLM inference)
# =====================================================
print("\n[TEST 6] Full Tutor pipeline (LLM inference)")
try:
    from langchain_ollama import OllamaLLM
    from vet_ai_tutor.core import Tutor
    from vet_ai_tutor.llm_adapter import OllamaAdapter

    base_llm = OllamaLLM(model=LLM_MODEL, num_predict=512)
    adapter = OllamaAdapter(base_llm)
    tutor = Tutor(llm=adapter, vector_path=VECTOR_PATH)

    # Q1: Direct question
    print("  Q1: 'What causes mastitis in dairy cows?'")
    t4 = time.time()
    ans1, docs1 = tutor.ask("What causes mastitis in dairy cows?")
    t5 = time.time()
    q1_time = t5 - t4
    q1_words = len(ans1.split())
    q1_has_sources = "[S" in ans1
    q1_relevant = any(kw in ans1.lower() for kw in ["mastitis", "bacteria", "udder", "infection"])
    print(f"    Time: {q1_time:.0f}s | Words: {q1_words} | Sources cited: {q1_has_sources} | Relevant: {q1_relevant}")

    # Q2: Follow-up (tests query enrichment)
    print("  Q2 (follow-up): 'How do you prevent it?'")
    t6 = time.time()
    ans2, docs2 = tutor.ask("How do you prevent it?")
    t7 = time.time()
    q2_time = t7 - t6
    q2_words = len(ans2.split())
    # Check if follow-up docs are about mastitis (not random)
    doc_text = " ".join(d.page_content.lower() for d in docs2)
    q2_on_topic = any(kw in doc_text for kw in ["mastitis", "udder", "hygiene", "milking"])
    print(f"    Time: {q2_time:.0f}s | Words: {q2_words} | On-topic retrieval: {q2_on_topic}")

    # Q3: Greeting handling
    print("  Q3 (greeting): 'Hello'")
    t8 = time.time()
    ans3, docs3 = tutor.ask("Hello")
    t9 = time.time()
    q3_fast = (t9 - t8) < 1.0
    q3_correct = "vet" in ans3.lower() or "tutor" in ans3.lower()
    print(f"    Time: {t9-t8:.2f}s | Handled locally: {q3_fast} | Correct: {q3_correct}")

    results["llm_q1"] = f"{'PASS' if q1_relevant else 'FAIL'} ({q1_time:.0f}s, {q1_words}w)"
    results["llm_followup"] = f"{'PASS' if q2_on_topic else 'FAIL'} ({q2_time:.0f}s, on-topic={q2_on_topic})"
    results["greeting"] = f"{'PASS' if q3_fast and q3_correct else 'FAIL'}"

    # Show answer previews
    print(f"\n  --- Q1 Answer Preview ---")
    print(f"  {ans1[:300]}...")
    print(f"\n  --- Q2 Answer Preview ---")
    print(f"  {ans2[:300]}...")

except Exception as e:
    print(f"  FAIL: {e}")
    results["llm_q1"] = f"FAIL: {e}"
    results["llm_followup"] = "SKIPPED"
    results["greeting"] = "SKIPPED"

# =====================================================
# TEST 7: QA eval pairs integrity
# =====================================================
print("\n[TEST 7] QA eval pairs integrity")
try:
    with open("data/eval/qa_pairs.json", "r", encoding="utf-8") as f:
        qa = json.load(f)
    ids = [item.get("id") for item in qa]
    unique_ids = len(set(ids))
    has_dupes = unique_ids != len(ids)
    all_have_q = all("question" in item and "expected_answer" in item for item in qa)
    print(f"  {len(qa)} pairs, {unique_ids} unique IDs, dupes: {has_dupes}, valid schema: {all_have_q}")
    results["qa_pairs"] = f"PASS ({len(qa)} pairs)" if not has_dupes and all_have_q else "WARN"
except Exception as e:
    print(f"  FAIL: {e}")
    results["qa_pairs"] = f"FAIL: {e}"

# =====================================================
# TEST 8: Streamlit app imports (no runtime test)
# =====================================================
print("\n[TEST 8] Streamlit app imports")
try:
    import streamlit
    # Just check the file parses
    import ast
    with open("vet_ai_tutor_app.py", "r") as f:
        ast.parse(f.read())
    print(f"  vet_ai_tutor_app.py parses OK, streamlit {streamlit.__version__}")
    results["streamlit_app"] = "PASS"
except Exception as e:
    print(f"  FAIL: {e}")
    results["streamlit_app"] = f"FAIL: {e}"

# =====================================================
# SUMMARY
# =====================================================
print(f"\n{'=' * 65}")
print("ASSESSMENT SUMMARY")
print(f"{'=' * 65}")
pass_count = sum(1 for v in results.values() if v.startswith("PASS"))
total = len(results)
for test, result in results.items():
    icon = "PASS" if result.startswith("PASS") else ("WARN" if "WARN" in result else "FAIL")
    print(f"  [{icon:4s}] {test}: {result}")
print(f"\n  Overall: {pass_count}/{total} passed")
print(f"{'=' * 65}")
