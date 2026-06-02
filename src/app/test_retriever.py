from src.retrieval.retriever import Retriever
from src.retrieval.retriever import Retriever

retriever = Retriever()

query = "attrition"

results = retriever.search(
    query,
    n_results=20
)

print("\nQUESTION:")
print(query)

print("\nTOP RESULTS:\n")

for i, doc in enumerate(results["documents"][0], start=1):

    print("=" * 60)
    print(f"RESULT {i}")
    print("=" * 60)

    print(doc[:500])
    print()