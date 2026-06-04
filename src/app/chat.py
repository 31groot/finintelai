from src.features.qa_chain import QAChain

qa_chain = QAChain()

print("\nFinancial RAG Chatbot Ready")
print("Type 'exit' to quit.\n")

while True:
    question = input("Ask: ")

    if question.lower() in ["exit", "quit"]:
        break

    try:
        answer = qa_chain.ask(question)

        print("\nANSWER:")
        print(answer)
        print()

    except Exception as e:
        print("\nERROR:")
        print(e)
        print()

        print("\nTOP RETRIEVED CHUNKS:\n")

for i, doc in enumerate(results["documents"][0]):
    print(f"\n----- CHUNK {i+1} -----")
    print(doc[:500])


        