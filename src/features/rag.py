from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker


class RAGPipeline:

    def __init__(self):

        self.retriever = Retriever()
        self.reranker = Reranker()

    def get_context(self, query):

        # ====================
        # RETRIEVE
        # ====================

        results = self.retriever.search(
            query,
            n_results=50
        )

        # ====================
        # VECTOR RESULTS
        # ====================

        vector_docs = (
            results["vector_results"]["documents"][0]
        )

        vector_metadata = (
            results["vector_results"]["metadatas"][0]
        )

        # ====================
        # BM25 RESULTS
        # ====================

        bm25_docs = (
            results["bm25_results"]
        )

        # ====================
        # MERGE + REMOVE DUPES
        # ====================

        combined_docs = list(
            dict.fromkeys(
                vector_docs + bm25_docs
            )
        )

        # ====================
        # DYNAMIC RERANK DEPTH
        # ====================

        ranking_keywords = [
            "rank",
            "compare",
            "highest",
            "lowest",
            "top",
            "revenue",
            "profit",
            "ebit",
            "ebitda",
            "attrition"
        ]

        query_lower = query.lower()

        if any(
            keyword in query_lower
            for keyword in ranking_keywords
        ):
            top_k = 15
        else:
            top_k = 10

        # ====================
        # RERANK
        # ====================

        reranked_docs = self.reranker.rerank(
            query,
            combined_docs,
            top_k=top_k
        )

        # ====================
        # CONTEXT
        # ====================

        context = "\n\n".join(
            reranked_docs
        )

        return {
            "context": context,
            "metadata": vector_metadata
        }