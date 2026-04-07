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


def test_tutor_ask_with_retriever():
    from vet_ai_tutor.core import Tutor

    # create instance without calling __init__ to avoid heavy deps
    t = object.__new__(Tutor)
    t.history = []
    t.llm = FakeLLM()
    t.db = FakeDB()
    t._bm25_data = None

    answer, docs = t.ask("What is it?", k=1, remember=False)

    assert answer == "LLM answer"
    assert isinstance(docs, list)
    assert docs[0].page_content == "found content"
