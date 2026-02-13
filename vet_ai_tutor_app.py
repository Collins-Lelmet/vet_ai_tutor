import streamlit as st
from langchain_ollama import OllamaLLM
from vet_ai_tutor.core import Tutor
from vet_ai_tutor.llm_adapter import OllamaAdapter
from vet_ai_tutor.config import VECTOR_PATH

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Vet AI Tutor (Local)", page_icon="🐄", layout="centered")

st.title("🐄 Veterinary AI Tutor (Local)")
st.caption("Runs fully offline using Ollama + your local knowledge base")

# -----------------------------
# Load Embeddings & Vector Store
# -----------------------------
@st.cache_resource
def load_tutor():
    # create and return a Tutor instance (caches in Streamlit)
    base_llm = OllamaLLM(model="gemma:2b")
    adapter = OllamaAdapter(base_llm)
    tutor = Tutor(llm=adapter, vector_path=VECTOR_PATH)
    return tutor

tutor = load_tutor()
retriever = tutor.db.as_retriever(search_kwargs={"k": 3})

# maintain chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -----------------------------
# User Input
# -----------------------------
question = st.text_input("Ask a veterinary question")

if question:
    with st.spinner("Thinking..."):
        answer, docs = tutor.ask(question, k=3)
        retrieved_context = "\n".join(d.page_content for d in docs)

    # append to session history for UI
    st.session_state["chat_history"].append(("You", question))
    st.session_state["chat_history"].append(("Tutor", answer))

    st.subheader("Conversation")
    for who, text in st.session_state["chat_history"]:
        st.markdown(f"**{who}:** {text}")

    with st.expander("🔍 Retrieved Context"):
        st.text(retrieved_context)

st.markdown("---")
st.caption("Local • Private • No Internet Required")
