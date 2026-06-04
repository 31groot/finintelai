from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


class Retriever:

    def __init__(self):

        self.embedder = EmbeddingModel()
        self.store = VectorStore()

    def search(self, query: str, n_results: int = 8):

        query_embedding = self.embedder.embed([query])

        results = self.store.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        print("\n========== DEBUG ==========")
        print("Collection count:", self.store.count())

        if results["documents"]:
            print(
                "Retrieved docs:",
                len(results["documents"][0])
            )
        else:
            print("Retrieved docs: 0")

        print("===========================\n")

        return results