from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever

RRF_K = 60

class Retriever:

    def __init__(self):
        self.embedder = EmbeddingModel()
        self.store = VectorStore()

        all_data = self.store.get_all_documents()
        self.documents = all_data["documents"]
        self.metadata = all_data["metadatas"]

        self.doc_to_metadata = dict(zip(self.documents, self.metadata))

        self.bm25 = BM25Retriever(self.documents)

    def _reciprocal_rank_fusion(
        self,
        vector_docs,
        vector_metadata,
        bm25_docs,
        n_results
    ):
        scores = {}

        for rank, doc in enumerate(vector_docs, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (rank + RRF_K)

        for rank, doc in enumerate(bm25_docs, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (rank + RRF_K)

        ranked_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        vector_meta_lookup = dict(zip(vector_docs, vector_metadata))

        fused_docs = []
        fused_metadata = []

        for doc, _ in ranked_docs[:n_results]:

            if doc in vector_meta_lookup:
                meta = vector_meta_lookup[doc]
            elif doc in self.doc_to_metadata:
                meta = self.doc_to_metadata[doc]
            else:
                meta = {}

            fused_docs.append(doc)
            fused_metadata.append(meta)

        return fused_docs, fused_metadata

    def search(self, query, n_results):

        query_embedding = self.embedder.embed([query])

        vector_results = self.store.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        if vector_results["documents"]:
            vector_docs = vector_results["documents"][0]
        else:
            vector_docs = []

        if vector_results["metadatas"]:
            vector_metadata = vector_results["metadatas"][0]
        else:
            vector_metadata = []

        bm25_docs = self.bm25.search(query, top_k=n_results)

        fused_docs, fused_metadata = self._reciprocal_rank_fusion(
            vector_docs,
            vector_metadata,
            bm25_docs,
            n_results
        )

        return {
            "documents": fused_docs,
            "metadata": fused_metadata
        }