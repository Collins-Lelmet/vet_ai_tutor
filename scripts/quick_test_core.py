import sys
import os
# ensure repo root is on sys.path so local package imports work when running directly
sys.path.insert(0, os.getcwd())
from vet_ai_tutor.core import Tutor

class FakeRetriever:
    def get_relevant_documents(self, q):
        class Doc:
            def __init__(self):
                self.page_content = "found content"
                self.metadata = {"source": "fake", "chunk": 0}
        return [Doc()]

class FakeDB:
    def as_retriever(self, search_kwargs=None):
        return FakeRetriever()

class FakeLLM:
    def invoke(self, prompt):
        return "LLM answer"

# emulate test
try:
    t = object.__new__(Tutor)
    t.history = []
    t.llm = FakeLLM()
    t.db = FakeDB()
    t._bm25_data = None

    answer, docs = t.ask("What is it?", k=1, remember=False)

    ok = answer == "LLM answer" and isinstance(docs, list) and docs[0].page_content == "found content"
    print("TEST PASSED" if ok else "TEST FAILED")
    if not ok:
        print('answer=', answer)
        print('docs=', docs)
except Exception as e:
    print('ERROR', e)
