"# vet_ai_tutor" 

## Quick setup

1. Create a Python virtual environment and install requirements:

```powershell
.\scripts\setup_env.ps1
```

2. Build the vectorstore (if you need to re-ingest documents):

```powershell
python scripts/chunk_documents.py
python scripts/build_vectorstore.py
```

3. Run a quick no-dependency smoke-check to verify index presence:

```powershell
.venv\Scripts\python scripts/smoke_check.py
```

4. Run the Streamlit app:

```powershell
.venv\Scripts\python -m streamlit run vet_ai_tutor_app.py
```

Canonical vectorstore location: `data/vectorstore` (scripts now consolidate any `vectorstore/` root directory into `data/vectorstore`).

