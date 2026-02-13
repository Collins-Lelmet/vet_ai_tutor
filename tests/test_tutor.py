from tutor import ask_tutor

while True:
    q = input("\n🐄 Question (or 'exit'): ")
    if q.lower() == "exit":
        break

    answer = ask_tutor(q)
    print("\n📘 Answer:\n", answer)
