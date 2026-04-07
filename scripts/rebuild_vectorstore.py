#!/usr/bin/env python3
"""Rebuild FAISS vectorstore from RAW_DIR using local HuggingFace embeddings.
This script avoids importing the other helper scripts to prevent circular imports
and writes the index to the canonical VECTOR_PATH from `vet_ai_tutor.config`.
"""
import os
import sys
import traceback

try:
    from vet_ai_tutor.config import (
        RAW_DIR, VECTOR_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
        SKIP_EXTENSIONS, SUPPORTED_EXTENSIONS,
    )
except Exception:
    # try to add repo root to path
    sys.path.insert(0, os.getcwd())
    from vet_ai_tutor.config import (
        RAW_DIR, VECTOR_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
        SKIP_EXTENSIONS, SUPPORTED_EXTENSIONS,
    )

print("RAW_DIR:", RAW_DIR)
print("VECTOR_PATH:", VECTOR_PATH)
print("EMBEDDING_MODEL:", EMBEDDING_MODEL)
print("CHUNK_SIZE, CHUNK_OVERLAP:", CHUNK_SIZE, CHUNK_OVERLAP)

# load required libraries
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        UnstructuredPowerPointLoader,
        UnstructuredWordDocumentLoader,
        TextLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception as e:
    print("Required packages missing:", e)
    print("Install requirements (see requirements.txt); aborting.")
    raise

def _get_loader(path: str, ext: str):
    """Return the appropriate LangChain loader for the file extension."""
    kind = SUPPORTED_EXTENSIONS.get(ext)
    if kind == "pdf":
        return PyPDFLoader(path)
    elif kind == "docx":
        return Docx2txtLoader(path)
    elif kind == "doc":
        return UnstructuredWordDocumentLoader(path)
    elif kind in ("pptx", "ppt"):
        return UnstructuredPowerPointLoader(path)
    elif kind == "txt":
        return TextLoader(path, encoding="utf-8")
    return None

# gather documents
documents = []
failed = []
skipped = []
for root, _, files in os.walk(RAW_DIR):
    for fn in sorted(files):
        path = os.path.join(root, fn)
        lower = fn.lower()
        ext = os.path.splitext(lower)[1]

        # Skip backup, image, and other unsupported files
        if ext in SKIP_EXTENSIONS:
            skipped.append(path)
            continue

        # Skip files with no extension (e.g. WhatsApp exports)
        if not ext:
            skipped.append(path)
            continue

        # Skip files not in supported list
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"Skipping unsupported file: {path}")
            skipped.append(path)
            continue

        try:
            loader = _get_loader(path, ext)
            if loader is None:
                skipped.append(path)
                continue

            loaded = loader.load()

            # Clean and enrich metadata
            source_name = os.path.basename(path)
            for d in loaded:
                md = getattr(d, "metadata", {}) or {}
                md["source"] = source_name
                md["source_path"] = path
                d.metadata = md

                # Skip empty documents
                content = getattr(d, "page_content", "") or ""
                if len(content.strip()) < 20:
                    continue

            documents.extend([d for d in loaded if len((getattr(d, "page_content", "") or "").strip()) >= 20])
            print(f"Loaded {fn}: {len(loaded)} pages/items")
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            failed.append({"path": path, "error": str(e)})

print(f"\n✅ Loaded total document pages/items: {len(documents)}")

# split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
for i, c in enumerate(chunks):
    md = getattr(c, "metadata", {}) or {}
    md.setdefault("source", md.get("source", "unknown"))
    md["chunk"] = i
    c.metadata = md

print(f"✅ Created {len(chunks)} text chunks")

# build embeddings and FAISS index
try:
    print("Instantiating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Computing embeddings and building FAISS index (this may take a while)...")
    db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTOR_PATH, exist_ok=True)
    db.save_local(VECTOR_PATH)
    print(f"✅ Vector database built and saved to {VECTOR_PATH}")
except Exception as e:
    print("Failed building/saving vectorstore:")
    traceback.print_exc()
    raise

# write a summary results file
out = os.path.join("archive", "rebuild_summary.json")
try:
    import json
    summary = {
        "loaded_pages": len(documents),
        "chunks": len(chunks),
        "failed": failed,
        "skipped": len(skipped),
        "vector_path": VECTOR_PATH,
        "embedding_model": EMBEDDING_MODEL,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary:", out)
except Exception as e:
    print("Could not write summary:", e)

print("Done.")
