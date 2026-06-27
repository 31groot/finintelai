from src.features.qa_chain import QAChain

qa_chain = QAChain()

print("Financial RAG Chatbot Ready")
print("Type 'exit' to quit.\n")

while True:
    question = input("Ask: ").strip()

    if question.lower() in ["exit", "quit"]:
        break

    if not question:
        print("\nPlease enter a question.\n")
        continue

    try:
        answer = qa_chain.ask(question)

        print("\nANSWER:")
        print(answer)
        print()

    except Exception as e:
        print("\nERROR:")
        print(e)
        print()
