import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)

RAW_DIR = "data/raw_notes"
documents = []

for root, _, files in os.walk(RAW_DIR):
    for file in files:
        path = os.path.join(root, file)

        try:
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
                documents.extend(loader.load())

            elif file.lower().endswith(".docx"):
                loader = Docx2txtLoader(path)
                documents.extend(loader.load())

            elif file.lower().endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(path)
                documents.extend(loader.load())

        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")

print(f"\n✅ Loaded {len(documents)} document pages")
