#!/usr/bin/env python3
"""Rebuild FAISS vectorstore from RAW_DIR using the shared pipeline."""
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vet_ai_tutor.config import RAW_DIR, VECTOR_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from vet_ai_tutor.pipeline import discover_documents, chunk_documents, build_vectorstore

print("RAW_DIR:", RAW_DIR)
print("VECTOR_PATH:", VECTOR_PATH)
print("EMBEDDING_MODEL:", EMBEDDING_MODEL)
print("CHUNK_SIZE, CHUNK_OVERLAP:", CHUNK_SIZE, CHUNK_OVERLAP)

try:
    print("Discovering documents...")
    docs, stats = discover_documents(return_stats=True)
    print("Discovered items:", len(docs))
    print("Skipped items:", len(stats["skipped"]))
    print("Failed items:", len(stats["failed"]))

    print("Building chunks...")
    chunks = chunk_documents(docs)
    print("Chunks created:", len(chunks))

    print("Building vectorstore...")
    result = build_vectorstore(chunks=chunks)
    print(f"✅ Vector database built and saved to {VECTOR_PATH}")
    print(result)
except Exception as e:
    print("Failed building vectorstore:")
    traceback.print_exc()
    raise

out = os.path.join("archive", "rebuild_summary.json")
try:
    summary = {
        "loaded_items": len(docs),
        "chunks": len(chunks),
        "vector_paths": result.get("vector_paths") if isinstance(result, dict) else None,
        "embedding_model": result.get("embedding_model") if isinstance(result, dict) else EMBEDDING_MODEL,
        "skipped": len(stats["skipped"]),
        "failed": len(stats["failed"]),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary:", out)
except Exception as e:
    print("Could not write summary:", e)

print("Done.")
