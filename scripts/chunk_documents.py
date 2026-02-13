from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_documents import documents

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"✅ Created {len(chunks)} text chunks")
