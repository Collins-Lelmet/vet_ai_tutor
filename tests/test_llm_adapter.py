from types import SimpleNamespace

from vet_ai_tutor.llm_adapter import LangChainChatAdapter


class FakeChatModel:
    def invoke(self, prompt):
        return SimpleNamespace(content="Complete answer")

    def stream(self, prompt):
        yield SimpleNamespace(content="Part one")
        yield SimpleNamespace(content="Part two")


def test_chat_adapter_returns_plain_text():
    adapter = LangChainChatAdapter(FakeChatModel())

    assert adapter.invoke("question") == "Complete answer"
    assert list(adapter.stream("question")) == ["Part one", "Part two"]


def test_chat_adapter_joins_text_content_blocks():
    message = SimpleNamespace(content=[{"type": "text", "text": "A"}, {"text": "B"}])

    assert LangChainChatAdapter._text(message) == "AB"