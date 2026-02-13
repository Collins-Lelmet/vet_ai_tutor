from vet_ai_tutor.core import Tutor
from vet_ai_tutor.llm_adapter import OllamaAdapter
from langchain_ollama import OllamaLLM


def main():
    base_llm = OllamaLLM(model="gemma:2b", temperature=0.2)
    adapter = OllamaAdapter(base_llm)
    tutor = Tutor(llm=adapter)

    history = []

    while True:
        question = input("\n🐄 Ask a veterinary question (or type 'exit' or 'reset'): ")
        if question.lower() == "exit":
            break
        if question.lower() == "reset":
            tutor.reset_history()
            history = []
            print("Conversation history cleared.")
            continue

        answer, docs = tutor.ask(question, k=5)
        history.append(("You", question))
        history.append(("Tutor", answer))

        print("\nConversation:")
        for who, text in history:
            print(f"{who}: {text}\n")


if __name__ == "__main__":
    main()
