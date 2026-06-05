from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.query_decomposer import QueryDecomposer


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.decomposer = QueryDecomposer()

    def get_context(self, query):

        # --------------------------------
        # Query Decomposition
        # --------------------------------

        subqueries = self.decomposer.decompose(query)

        all_docs = []
        all_metadata = []

        # --------------------------------
        # Retrieval
        # --------------------------------

        for subquery in subqueries:

            results = self.retriever.search(
                query=subquery,
                n_results=15
            )

            # Vector Search Results
            vector_docs = (
                results["vector_results"]["documents"][0]
                if results["vector_results"]["documents"]
                else []
            )

            vector_metadata = (
                results["vector_results"]["metadatas"][0]
                if results["vector_results"]["metadatas"]
                else []
            )

            # BM25 Results
            bm25_docs = results["bm25_results"]

            all_docs.extend(vector_docs)
            all_docs.extend(bm25_docs)

            all_metadata.extend(vector_metadata)

        # --------------------------------
        # Deduplicate
        # --------------------------------

        unique_docs = list(dict.fromkeys(all_docs))

        # --------------------------------
        # Dynamic Reranking Depth
        # --------------------------------

        comparison_keywords = [
            "compare",
            "comparison",
            "rank",
            "ranking",
            "highest",
            "lowest",
            "best",
            "worst",
            "versus",
            "vs"
        ]

        query_lower = query.lower()

        is_comparison = any(
            keyword in query_lower
            for keyword in comparison_keywords
        )

        if is_comparison:
            top_k = 20
        else:
            top_k = 10

        # --------------------------------
        # Reranking
        # --------------------------------

        reranked_docs = self.reranker.rerank(
            query=query,
            documents=unique_docs,
            top_k=top_k
        )

        # --------------------------------
        # Context Creation
        # --------------------------------

        context = "\n\n".join(reranked_docs)

        return {
            "context": context,
            "metadata": all_metadata
        }