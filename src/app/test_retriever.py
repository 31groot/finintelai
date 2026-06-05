from src.retrieval.retriever import Retriever

retriever = Retriever()

query = "Mazdock order book"

results = retriever.search(
    query,
    n_results=10
)

print("\nQUESTION:")
print(query)

print("\n" + "=" * 60)
print("VECTOR RESULTS")
print("=" * 60)

vector_docs = results["vector_results"]["documents"][0]
vector_meta = results["vector_results"]["metadatas"][0]

for i, (doc, meta) in enumerate(
    zip(vector_docs, vector_meta),
    start=1
):

    print(f"\nRESULT {i}")
    print("-" * 40)

    print(doc[:500])

    print("\nMETADATA:")
    print(meta)

print("\n" + "=" * 60)
print("BM25 RESULTS")
print("=" * 60)

for i, doc in enumerate(
    results["bm25_results"],
    start=1
):

    print(f"\nRESULT {i}")
    print("-" * 40)

    print(doc[:500])