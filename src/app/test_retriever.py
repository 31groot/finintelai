from src.retrieval.retriever import Retriever

retriever = Retriever()

query = "Infosys revenue FY24"

results = retriever.search(
    query,
    n_results=10
)

print("\nQUESTION:")
print(query)

print("\nTOP RESULTS:\n")

for i, doc in enumerate(
    results["documents"][0],
    start=1
):
    print("=" * 60)
    print(f"RESULT {i}")
    print("=" * 60)

    print(doc[:1000])

    print("\nMETADATA:")
    print(results["metadatas"][0][i - 1])

    print()