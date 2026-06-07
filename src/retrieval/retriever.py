from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever
class Retriever:

    def __init__(self):

        self.embedder = EmbeddingModel()

        self.store = VectorStore()

        all_data = (
            self.store.get_all_documents()
        )

        self.documents = (
            all_data["documents"]
        )

        self.metadata = (
            all_data["metadatas"]
        )

        self.bm25 = BM25Retriever(
            self.documents
        )

    def search(
        self,
        query,
        n_results=15
    ):

        query_embedding = (
            self.embedder.embed([query])
        )

        vector_results = (
            self.store.collection.query(
                query_embeddings=
                query_embedding.tolist(),
                n_results=n_results,
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )
        )

        bm25_docs = (
            self.bm25.search(
                query,
                top_k=n_results
            )
        )

        return {
            "vector_results":
            vector_results,

            "bm25_results":
            bm25_docs
        }