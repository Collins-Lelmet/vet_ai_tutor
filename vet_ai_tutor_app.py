import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)

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
    try:
        from langchain_ollama import OllamaLLM
        from vet_ai_tutor.core import Tutor
        from vet_ai_tutor.llm_adapter import OllamaAdapter
        from vet_ai_tutor.config import VECTOR_PATH, LLM_MODEL

        base_llm = OllamaLLM(model=LLM_MODEL, num_predict=512)
        adapter = OllamaAdapter(base_llm)
        tutor = Tutor(llm=adapter, vector_path=VECTOR_PATH)
        return tutor, None
    except Exception as e:
        return None, str(e)

tutor, load_error = load_tutor()

if load_error:
    st.error(f"Failed to initialize tutor: {load_error}")
    st.info("Make sure Ollama is running and the vectorstore has been built. "
            "Run: `python scripts/rebuild_vectorstore.py`")
    st.stop()

if tutor.db is None:
    st.error("Vectorstore not found or failed to load.")
    st.info("Build it first: `python scripts/rebuild_vectorstore.py`")
    st.stop()

# maintain chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Reset button
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Reset"):
        st.session_state["chat_history"] = []
        tutor.reset_history()
        st.rerun()

# -----------------------------
# User Input
# -----------------------------
question = st.chat_input("Ask a veterinary question...")

if question:
    # append user message immediately
    st.session_state["chat_history"].append(("You", question))

    with st.spinner("Thinking..."):
        try:
            answer, docs = tutor.ask(question, k=6)
            retrieved_context = "\n\n---\n\n".join(
                f"**{getattr(d, 'metadata', {}).get('source', 'unknown')}**\n{d.page_content}"
                for d in docs
            )
        except Exception as e:
            answer = f"Error: {e}"
            retrieved_context = ""
            docs = []

    st.session_state["chat_history"].append(("Tutor", answer))

# Display conversation
for who, text in st.session_state["chat_history"]:
    with st.chat_message("user" if who == "You" else "assistant"):
        st.markdown(text)

# Show retrieved context if there are docs from the last question
if st.session_state["chat_history"] and docs:
    with st.expander("🔍 Retrieved Context"):
        st.markdown(retrieved_context)

st.markdown("---")
st.caption("Local • Private • No Internet Required")
