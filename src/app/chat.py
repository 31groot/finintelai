from src.agents.graph import run

print("Financial RAG Chatbot Ready")
print("Type 'exit' to quit, 'reset' to clear company memory.\n")

memory_companies = []

while True:
    question = input("Ask: ").strip()

    if question.lower() in ["exit", "quit"]:
        break

    if not question:
        print("\nPlease enter a question.\n")
        continue

    if question.lower() == "reset":
        memory_companies = []
        print("\nCompany memory cleared.\n")
        continue

    try:
        result = run(question, companies=memory_companies)

        print("\n" + "=" * 60)
        print(result.get("answer", ""))
        print("=" * 60)

        verification = result.get("verification") or {}
        grounded = verification.get("grounded")
        confidence = verification.get("confidence")

        if grounded is not None:
            status = "grounded" if grounded else "NOT grounded"
            if confidence is not None:
                status += f" ({round(confidence * 100)}%)"
            print(f"Verification: {status}")

        reason = verification.get("reason")
        if reason and not grounded:
            print(f"Reason: {reason}")

        subqueries = result.get("subqueries") or []
        if subqueries:
            print(f"Subqueries: {len(subqueries)}")

        detected_companies = result.get("companies") or []
        if detected_companies:
            memory_companies = list(detected_companies)

        if memory_companies:
            print(f"Company memory: {', '.join(memory_companies)}")

        print()

    except Exception as e:
        print("\nERROR:")
        print(f"{type(e).__name__}: {e}")
        print()