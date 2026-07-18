from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever

RRF_K = 60
RRF_CANDIDATES = 50

class Retriever:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.store = VectorStore()

        all_data = self.store.get_all_documents()
        self.documents = all_data["documents"]
        self.metadata = all_data["metadatas"]

        self.records = []
        for idx, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            self.records.append(
                {
                    "doc_id": idx,
                    "text": doc,
                    "metadata": meta or {},
                }
            )

        self.bm25 = BM25Retriever([record["text"] for record in self.records])

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
            

        source_kinds = [s.lower() for s in filters.get("source_kinds", [])]
        if source_kinds:
            meta_source_kind = str(meta.get("source_kind", "")).lower()
            if meta_source_kind not in source_kinds:
                return False

        statement_type = str(filters.get("statement_type", "")).lower().strip()
        if statement_type:
            meta_statement_type = str(meta.get("statement_type", "")).lower().strip()
            if meta_statement_type != statement_type:
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
        bm25_indices = self.bm25.search(query, top_k=max(top_k * 20, 300))

        filtered_docs = []
        filtered_metadata = []

        for idx in bm25_indices:
            record = self.records[idx]
            doc = record["text"]
            meta = record["metadata"]

            if self._matches_filters(meta, filters):
                filtered_docs.append(doc)
                filtered_metadata.append(meta)

            if len(filtered_docs) >= top_k:
                break

        return filtered_docs, filtered_metadata

    def _reciprocal_rank_fusion(
        self,
        vector_docs,
        vector_metadata,
        bm25_docs,
        bm25_metadata,
        n_results,
    ):
        scores = {}

        for rank, doc in enumerate(vector_docs, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (rank + RRF_K)

        for rank, doc in enumerate(bm25_docs, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (rank + RRF_K)

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        vector_meta_lookup = {}
        for doc, meta in zip(vector_docs, vector_metadata):
            if doc not in vector_meta_lookup:
                vector_meta_lookup[doc] = meta

        bm25_meta_lookup = {}
        for doc, meta in zip(bm25_docs, bm25_metadata):
            if doc not in bm25_meta_lookup:
                bm25_meta_lookup[doc] = meta

        fused_docs = []
        fused_metadata = []

        for doc, _ in ranked_docs[:n_results]:
            if doc in vector_meta_lookup:
                meta = vector_meta_lookup[doc]
            elif doc in bm25_meta_lookup:
                meta = bm25_meta_lookup[doc]
            else:
                meta = {}

            fused_docs.append(doc)
            fused_metadata.append(meta)

        return fused_docs, fused_metadata

    def _search_once(self, query, n_results, filters=None):
        query_embedding = self.embedder.embed([query])

        if filters:
            vector_pool_size = max(n_results * 20, 300)
        else:
            vector_pool_size = n_results

        vector_results = self.store.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=vector_pool_size,
            include=["documents", "metadatas", "distances"],
        )

        vector_docs = (
            vector_results["documents"][0]
            if vector_results["documents"]
            else []
        )
        vector_metadata = (
            vector_results["metadatas"][0]
            if vector_results["metadatas"]
            else []
        )

        vector_docs, vector_metadata = self._filter_docs_and_metadata(
            vector_docs,
            vector_metadata,
            filters,
        )

        vector_docs = vector_docs[:RRF_CANDIDATES]
        vector_metadata = vector_metadata[:RRF_CANDIDATES]

        bm25_docs, bm25_metadata = self._get_filtered_bm25_docs(
            query=query,
            top_k=RRF_CANDIDATES,
            filters=filters,
        )

        fused_docs, fused_metadata = self._reciprocal_rank_fusion(
            vector_docs,
            vector_metadata,
            bm25_docs,
            bm25_metadata,
            n_results,
        )

        return {
            "documents": fused_docs,
            "metadata": fused_metadata,
        }

    def search(self, query, n_results, filters=None):
        result = self._search_once(query=query, n_results=n_results, filters=filters)

        if result["documents"]:
            return result

        if filters and filters.get("statement_type"):
            relaxed_filters = dict(filters)
            relaxed_filters.pop("statement_type", None)

            relaxed_result = self._search_once(
                query=query,
                n_results=n_results,
                filters=relaxed_filters,
            )

            if relaxed_result["documents"]:
                return relaxed_result

        return result