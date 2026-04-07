"""Load raw documents from the configured RAW_DIR and split into chunks.

This module exports a `chunks` variable so it can be imported by
`scripts/load_documents.py`.
"""
import os
from vet_ai_tutor.config import RAW_DIR, CHUNK_SIZE, CHUNK_OVERLAP

documents = []

try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        UnstructuredPowerPointLoader,
        TextLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    raise RuntimeError("Required document loader/splitter packages are missing: %s" % e)

for root, _, files in os.walk(RAW_DIR):
    for file in sorted(files):
        path = os.path.join(root, file)

        try:
            loader = None
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.lower().endswith(".docx"):
                loader = Docx2txtLoader(path)
            elif file.lower().endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(path)
            elif file.lower().endswith(".txt"):
                loader = TextLoader(path)

            if loader is None:
                continue

            loaded = loader.load()
            for d in loaded:
                md = getattr(d, "metadata", {}) or {}
                # preserve original source path in metadata
                md.setdefault("source", path)
                d.metadata = md

            documents.extend(loaded)

        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")

print(f"\n✅ Loaded {len(documents)} document pages")

# Split into chunks and annotate chunk metadata
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

chunks = text_splitter.split_documents(documents)
for i, c in enumerate(chunks):
    md = getattr(c, "metadata", {}) or {}
    md["chunk"] = i
    # ensure source key exists (carried from original document loaders)
    md.setdefault("source", md.get("source", "unknown"))
    c.metadata = md

print(f"✅ Created {len(chunks)} text chunks")
