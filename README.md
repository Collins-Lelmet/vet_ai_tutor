# Veterinary AI Tutor

Streamlit chat app with retrieval over a FAISS knowledge base. Local runs use Ollama; hosted deployments can use any OpenAI-compatible chat API.

## Run Locally

From PowerShell in the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run vet_ai_tutor_app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`. Keep the terminal open while using the app.

Local prerequisites:

- Ollama must be running and have the `llama3.2-fast` model available.
- The vectorstore must exist at `data/vectorstore`. Rebuild it from source notes with:

```powershell
python scripts\rebuild_vectorstore.py
```

The project environment can be created with `python -m venv .venv`. Activate it before installing the requirements.

## Deploy To Streamlit Community Cloud

1. Push the project code to a GitHub repository and select `vet_ai_tutor_app.py` as the app file when creating a Streamlit Community Cloud app.
2. In the app's **Settings > Secrets**, configure an OpenAI-compatible provider:

```toml
VET_AI_LLM_PROVIDER = "openai-compatible"
OPENAI_API_KEY = "your-provider-api-key"
OPENAI_MODEL = "your-provider-model-name"
OPENAI_BASE_URL = "https://api.openai.com/v1"
```

Use the base URL and model name supplied by your chosen provider. `OPENAI_BASE_URL` may be omitted when using the default OpenAI endpoint. Never commit real API keys or a local `.streamlit/secrets.toml` file.
3. The private deployment branch must contain `data/vectorstore/index.faiss` and `data/vectorstore/index.pkl`. These two files are allowed by `.gitignore`; raw source notes remain excluded. Keep the GitHub repository private because the index contains extracted note text.

The hosted model receives user questions and retrieved note excerpts. Check the provider's data handling and usage costs before enabling it. A private GitHub repository does not automatically make the deployed Streamlit app private; configure app access separately. Local Ollama remains the default when `VET_AI_LLM_PROVIDER` is not set.

## Project Paths

- Streamlit entry point: `vet_ai_tutor_app.py`
- Retrieval and prompting: `vet_ai_tutor/core.py`
- Ingestion and index rebuild: `scripts/rebuild_vectorstore.py`
- Canonical vectorstore: `data/vectorstore`
- Raw source notes: `data/raw_notes` (local-only; ignored by Git)
