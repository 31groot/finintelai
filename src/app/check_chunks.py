
from src.retrieval.vector_store import VectorStore

store = VectorStore()

docs = store.collection.get()

print("Total chunks:", len(docs["documents"]))

for i in range(10):
    print("\n" + "="*50)
    print(f"CHUNK {i}")
    print("="*50)
    print(docs["documents"][i][:500])