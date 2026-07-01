from src.agents.graph import run

print("Financial RAG Chatbot Ready")
print("Type 'exit' to quit.\n")

memory_companies = []

while True:
    question = input("Ask: ").strip()

    if question.lower() in ["exit", "quit"]:
        break

    if not question:
        print("\nPlease enter a question.\n")
        continue

    try:
        result = run(question, companies=memory_companies)

        print("\n" + "=" * 60)
        print(result["answer"])
        print("=" * 60 + "\n")

        companies = result.get("verification", {})
        detected = result.get("consistency")

    except Exception as e:
        print("\nERROR:")
        print(e)
        print()
