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

    def _matches_filters(self, meta, filters):
        if not filters:
            return True

        meta = meta or {}

        companies = [c.lower() for c in filters.get("companies", [])]
        if companies:
            meta_company = str(meta.get("company", "")).lower()
            if meta_company not in companies:
                return False

        fiscal_years = [fy.upper() for fy in filters.get("fiscal_years", [])]
        if fiscal_years:
            meta_fy = str(meta.get("fiscal_year", "")).upper()
            if meta_fy not in fiscal_years:
                return False

        doc_types = [d.lower() for d in filters.get("doc_types", [])]
        if doc_types:
            meta_doc_type = str(meta.get("doc_type", "")).lower()
            if meta_doc_type not in doc_types:
                return False

        quarters = [q.upper() for q in filters.get("quarters", [])]
        if quarters:
            meta_quarter = str(meta.get("quarter", "")).upper()
            if meta_quarter not in quarters:
                return False

        return True

    def _filter_docs_and_metadata(self, docs, metadata, filters):
        if not filters:
            return docs, metadata

        filtered_docs = []
        filtered_metadata = []

        for doc, meta in zip(docs, metadata):
            if self._matches_filters(meta, filters):
                filtered_docs.append(doc)
                filtered_metadata.append(meta)

        return filtered_docs, filtered_metadata

    def _get_filtered_bm25_docs(self, query, top_k, filters):
        bm25_pool = self.bm25.search(query, top_k=max(top_k * 4, 20))

        filtered = []
        for doc in bm25_pool:
            meta = self.doc_to_metadata.get(doc, {})
            if self._matches_filters(meta, filters):
                filtered.append(doc)

        return filtered[:top_k]

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

    def search(self, query, n_results, filters=None):
        query_embedding = self.embedder.embed([query])

        vector_pool_size = max(n_results * 3, 20) if filters else n_results

        vector_results = self.store.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=vector_pool_size,
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

        vector_docs, vector_metadata = self._filter_docs_and_metadata(
            vector_docs,
            vector_metadata,
            filters
        )

        vector_docs = vector_docs[:n_results]
        vector_metadata = vector_metadata[:n_results]

        bm25_docs = self._get_filtered_bm25_docs(
            query=query,
            top_k=n_results,
            filters=filters
        )

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