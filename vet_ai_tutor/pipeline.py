from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from vet_ai_tutor.config import (
    RAW_DIR,
    VECTOR_PATH,
    LEGACY_VECTOR_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SKIP_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
)


def _resolve_path(path: str | os.PathLike | None, base_dir: Optional[Path] = None) -> Path:
    if path is None:
        path = ""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / candidate).resolve()


def discover_documents(raw_dir: str | os.PathLike | None = None, return_stats: bool = False):
    """Load supported documents from the configured raw notes directory."""
    raw_dir_path = _resolve_path(raw_dir or RAW_DIR)
    documents: List[Any] = []
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    logger = logging.getLogger(__name__)
    try:
        from langchain_community.document_loaders import (
            PyPDFLoader,
            Docx2txtLoader,
            UnstructuredPowerPointLoader,
            UnstructuredWordDocumentLoader,
            TextLoader,
        )
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Required document loader packages are missing: {exc}") from exc

    def _binary_available(binary_name: str) -> bool:
        return shutil.which(binary_name) is not None

    if not raw_dir_path.exists():
        raise FileNotFoundError(f"Raw documents directory not found: {raw_dir_path}")

    for root, _, files in os.walk(raw_dir_path):
        for file_name in sorted(files):
            path = Path(root) / file_name
            lower_name = file_name.lower()
            ext = Path(lower_name).suffix

            if ext in SKIP_EXTENSIONS or not ext:
                skipped.append((str(path), "skip extension"))
                continue
            if ext not in SUPPORTED_EXTENSIONS:
                skipped.append((str(path), "unsupported extension"))
                continue

            try:
                loader = None
                # Prefer lightweight loaders that don't require external binaries
                if ext == ".pdf":
                    loader = PyPDFLoader(str(path))
                elif ext == ".docx":
                    # Docx2txtLoader is pure-python and safe; prefer it
                    loader = Docx2txtLoader(str(path))
                elif ext == ".doc":
                    # Use LibreOffice-backed loader for legacy Word files if available
                    if _binary_available("soffice") or _binary_available("libreoffice"):
                        loader = UnstructuredWordDocumentLoader(str(path))
                    else:
                        logger.warning("Skipping %s because required external converter (soffice/libreoffice) not found", path)
                        continue
                elif ext in {".pptx", ".pptm", ".ppt"}:
                    # Use LibreOffice-backed loader for PowerPoint files if available
                    if _binary_available("soffice") or _binary_available("libreoffice"):
                        loader = UnstructuredPowerPointLoader(str(path))
                    else:
                        logger.warning("Skipping %s because required external converter (soffice/libreoffice) not found", path)
                        continue
                elif ext == ".txt":
                    loader = TextLoader(str(path), encoding="utf-8")

                if loader is None:
                    skipped.append((str(path), "no loader available"))
                    continue

                try:
                    loaded = loader.load()
                except Exception as exc_load:
                    logger.warning("Failed to load %s with %s: %s", path, loader.__class__.__name__, exc_load)
                    failed.append((str(path), str(exc_load)))
                    continue

                for document in loaded:
                    metadata = getattr(document, "metadata", {}) or {}
                    metadata["source"] = file_name
                    metadata["source_path"] = str(path)
                    document.metadata = metadata
                    content = getattr(document, "page_content", "") or ""
                    if content.strip():
                        documents.append(document)
                    else:
                        skipped.append((str(path), "empty content"))
            except Exception as exc:  # pragma: no cover - runtime environment dependent
                logger.exception("Unexpected error loading %s: %s", path, exc)
                failed.append((str(path), str(exc)))

    if return_stats:
        return documents, {
            "skipped": skipped,
            "failed": failed,
        }
    return documents


def chunk_documents(documents: List[Any], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Split documents into retrievable chunks with stable metadata."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Required text splitter package is missing: {exc}") from exc

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        metadata = getattr(chunk, "metadata", {}) or {}
        metadata.setdefault("source", metadata.get("source", "unknown"))
        metadata["chunk"] = index
        chunk.metadata = metadata
    return chunks


def build_vectorstore(
    chunks: Optional[List[Any]] = None,
    raw_dir: str | os.PathLike | None = None,
    vector_path: str | os.PathLike | None = None,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and persist a FAISS index from the configured documents."""
    if chunks is None:
        documents = discover_documents(raw_dir=raw_dir)
        chunks = chunk_documents(documents)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Required vectorstore packages are missing: {exc}") from exc

    model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(chunks, embeddings)

    target_paths = []
    canonical_path = _resolve_path(vector_path or VECTOR_PATH)
    legacy_path = _resolve_path(LEGACY_VECTOR_PATH)

    canonical_path.mkdir(parents=True, exist_ok=True)
    legacy_path.mkdir(parents=True, exist_ok=True)

    db.save_local(str(canonical_path))
    db.save_local(str(legacy_path))
    target_paths.extend([str(canonical_path), str(legacy_path)])

    return {
        "chunks": len(chunks),
        "vector_paths": target_paths,
        "embedding_model": model_name,
    }


def build_index(raw_dir: str | os.PathLike | None = None, vector_path: str | os.PathLike | None = None):
    """Convenience wrapper for the canonical indexing workflow."""
    documents = discover_documents(raw_dir=raw_dir)
    chunks = chunk_documents(documents)
    return build_vectorstore(chunks=chunks, raw_dir=raw_dir, vector_path=vector_path)
