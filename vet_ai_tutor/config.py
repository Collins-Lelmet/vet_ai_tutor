# Central configuration for Vet AI Tutor
VECTOR_PATH = "data/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama3.2-fast"
RETRIEVER_K = 4

# Input/raw data directory (where you keep PDFs/DOCs/PPTs)
RAW_DIR = "data/raw_notes"

# Chunking defaults (tune as needed)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# File extensions to skip during ingestion (backups, images, etc.)
SKIP_EXTENSIONS = {
    ".bak", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".xlsx", ".xls",
    ".csv", ".zip", ".rar", ".mp4", ".mp3", ".wav",
}

# Supported document extensions and their loader types
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".pptx": "pptx",
    ".pptm": "pptx",
    ".ppt": "ppt",
    ".txt": "txt",
}
