import json
import argparse
from pathlib import Path
import time
import numpy as np

try:
    from vet_ai_tutor.config import VECTOR_PATH, EMBEDDING_MODEL
except Exception:
    VECTOR_PATH = "data/vectorstore"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
except Exception:
    # fallbacks used in this repo
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

def normalize(text: str) -> str:
    return (text or "").strip().lower()

def cosine_sim(a, b) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_qas(path: Path):
    return json.loads(path.read_text(encoding="utf8"))

def main():
    p = argparse.ArgumentParser(description="Evaluate retrieval recall against QA pairs (cosine on embeddings)")
    p.add_argument("--qa", default="data/eval/qa_pairs.json", help="QA JSON file")
    p.add_argument("--k", type=int, default=4, help="retrieval k (top-k)")
    p.add_argument("--sim-threshold", type=float, default=0.6, help="low similarity threshold (0..1)")
    args = p.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"QA file not found: {qa_path}")
        return

    qas = load_qas(qa_path)
    print(f"Loaded {len(qas)} QA pairs from {qa_path}")

    print("Loading embeddings and FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": args.k})

    total = len(qas)
    recall_hits = 0
    avg_best_sim = 0.0

    results = []

    for item in qas:
        q = item.get("question", "")
        expected = item.get("answer", "")
        # support different retriever APIs
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(q)
        elif hasattr(retriever, "_get_relevant_documents"):
            try:
                docs = retriever._get_relevant_documents(q, run_manager=None)
            except TypeError:
                docs = retriever._get_relevant_documents(q)
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(q)
        else:
            # fallback to similarity_search if available
            if hasattr(retriever, "similarity_search"):
                docs = retriever.similarity_search(q, k=args.k)
            else:
                raise RuntimeError("Retriever does not support known retrieval API")
        found = False
        best_sim = 0.0

        # exact substring check first
        for d in docs:
            content = d.page_content if hasattr(d, "page_content") else str(d)
            if normalize(expected) in normalize(content):
                found = True
                best_sim = 1.0
                break

        # if not exact, compute cosine similarity using embeddings
        if not found:
            try:
                exp_emb = embeddings.embed_query(expected)
                contents = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
                if contents:
                    doc_embs = embeddings.embed_documents(contents)
                    for emb in doc_embs:
                        s = cosine_sim(exp_emb, emb)
                        if s > best_sim:
                            best_sim = s
            except Exception as e:
                print(f"Embedding error for QA id={item.get('id')}: {e}")

        # dynamic thresholds
        low_thresh = args.sim_threshold
        high_thresh = min(0.95, low_thresh + 0.2)

        exp_len = len(normalize(expected))
        if exp_len > 80:
            high_thresh = max(0.75, high_thresh - 0.05)
        elif exp_len < 20:
            low_thresh = max(low_thresh, 0.75)
            high_thresh = max(high_thresh, 0.9)

        if found or best_sim >= high_thresh:
            status = "FOUND"
            final_found = True
        elif best_sim >= low_thresh:
            status = "PARTIAL"
            final_found = False
        else:
            status = "MISS"
            final_found = False

        recall_hits += 1 if final_found else 0
        avg_best_sim += best_sim

        print(f"[{item.get('id')}] k={args.k} status={status} best_sim={best_sim:.3f} -- Q: {q}")

        results.append({
            "id": item.get("id"),
            "question": q,
            "expected": expected,
            "found": final_found,
            "status": status,
            "best_sim": best_sim
        })

    avg_best_sim = avg_best_sim / total if total else 0.0
    recall = recall_hits / total if total else 0.0

    print("\n=== SUMMARY ===")
    print(f"Total QA pairs: {total}")
    print(f"Recall@{args.k} (FOUND count): {recall_hits}/{total} = {recall:.2%}")
    print(f"Avg best cosine similarity: {avg_best_sim:.3f}")

    print("\nNotes:")
    print("- FOUND = exact match OR cosine >= high threshold")
    print("- PARTIAL = cosine between low and high threshold")
    print("- MISS = cosine < low threshold")

    out_dir = Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_k{args.k}_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf8")
    print(f"Wrote detailed results to {out_path}")

if __name__ == "__main__":
    main()
