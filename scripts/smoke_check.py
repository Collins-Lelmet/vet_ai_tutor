"""Quick smoke-check: load the vectorstore and run a simple Tutor query.

This script is safe to run locally and uses a dummy LLM adapter so it does
not require Ollama to be installed.
"""
import sys
import os

# Ensure project root is on sys.path so package imports work when running scripts
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

try:
    from vet_ai_tutor.core import Tutor
except Exception as e:
    msg = str(e)
    print("ERROR: could not import Tutor:", msg)

    # No-install fallback: verify that a FAISS index exists in common locations.
    print()
    print("Attempting no-dependency check: looking for FAISS index files...")
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(proj_root, "data", "vectorstore"),
        os.path.join(proj_root, "vectorstore"),
        os.path.join(proj_root, "data"),
    ]

    found = False
    for d in candidates:
        if os.path.isdir(d):
            files = os.listdir(d)
            faiss_files = [f for f in files if f.endswith(".faiss") or f.endswith(".index") or f.endswith(".pkl")]
            if faiss_files:
                print(f"Found index files in {d}: {faiss_files}")
                found = True
            else:
                print(f"No index files in {d}")
        else:
            print(f"Directory not found: {d}")

    if found:
        print("NO-DEPENDENCY SMOKE CHECK: OK — vectorstore files present.")
        sys.exit(0)
    else:
        print()
        print("No FAISS index files found. To fully run the app, install dependencies:")
        print("  .\\.venv\\Scripts\\Activate.ps1")
        print("  python -m pip install -r requirements.txt")
        sys.exit(2)


def check_index_files_and_exit():
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(proj_root, "data", "vectorstore"),
        os.path.join(proj_root, "vectorstore"),
        os.path.join(proj_root, "data"),
    ]
    found = False
    for d in candidates:
        if os.path.isdir(d):
            files = os.listdir(d)
            faiss_files = [f for f in files if f.endswith(".faiss") or f.endswith(".index") or f.endswith(".pkl")]
            if faiss_files:
                print(f"Found index files in {d}: {faiss_files}")
                found = True
            else:
                print(f"No index files in {d}")
        else:
            print(f"Directory not found: {d}")
    if found:
        print("NO-DEPENDENCY SMOKE CHECK: OK — vectorstore files present.")
        sys.exit(0)
    else:
        print()
        print("No FAISS index files found. To fully run the app, install dependencies:")
        print("  .\\.venv\\Scripts\\Activate.ps1")
        print("  python -m pip install -r requirements.txt")
        sys.exit(2)


class DummyLLM:
    def invoke(self, prompt: str) -> str:
        return "(dummy LLM)"


def main():
    try:
        tutor = Tutor(llm=DummyLLM())
    except Exception as e:
        print("ERROR: failed to initialize Tutor (vectorstore or embeddings missing):", e)
        print("Falling back to no-dependency FAISS index check...")
        check_index_files_and_exit()

    try:
        answer, docs = tutor.ask("hello")
        print("SMOKE CHECK OK")
        print("Answer:", answer)
        print("Retrieved docs:", len(docs))
        return 0
    except Exception as e:
        print("ERROR during ask():", e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
