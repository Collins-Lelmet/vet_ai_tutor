from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embeddings model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Rabies is a fatal viral disease affecting the nervous system."

vector = embeddings.embed_query(text)

print("✅ Embeddings loaded successfully")
print("Vector length:", len(vector))
print("First 5 values:", vector[:5])
