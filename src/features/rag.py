from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.query_decomposer import QueryDecomposer

compare_keywords = [
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
class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.decomposer = QueryDecomposer()

    def get_context(self, query):

        subqueries = self.decomposer.decompose(query)

        query_lower = query.lower()
        is_comparison = any(
            keyword in query_lower
            for keyword in compare_keywords 
        )

        balanced_docs = []
        balanced_metadata = []

        for subquery in subqueries:

            results = self.retriever.search(
                query=subquery,
                n_results=22 if is_comparison else 15
            )

            vector_docs = results["documents"]
            vector_metadata = results["metadata"]
            
            seen_docs = set()
            docs = []
            metadata = []

            for doc, meta in zip(vector_docs, vector_metadata):
                if doc not in seen_docs:
                    seen_docs.add(doc)
                    docs.append(doc)
                    metadata.append(meta)

            if is_comparison:
                if len(subqueries) == 1:
                    rerank_top_k = 20
                elif len(subqueries) == 2:
                    rerank_top_k = 12
                elif len(subqueries) == 3:
                    rerank_top_k = 8
                else:
                    rerank_top_k = 6
            else:
                if len(subqueries) == 1:
                    rerank_top_k = 15
                elif len(subqueries) == 2:
                    rerank_top_k = 8
                elif len(subqueries) == 3:
                    rerank_top_k = 5
                else:
                    rerank_top_k = 4

            top_results = self.reranker.rerank(
                query=subquery,
                documents=docs,
                metadata=metadata,
                top_k=rerank_top_k
            )

            for doc, meta, score in top_results:
                balanced_docs.append(doc)
                balanced_metadata.append(meta)

        final_docs = []
        final_metadata = []
        seen_docs = set()

        for doc, meta in zip(balanced_docs, balanced_metadata):
            if doc not in seen_docs:
                seen_docs.add(doc)
                final_docs.append(doc)
                final_metadata.append(meta)

        context = "\n\n".join(final_docs)

        return {
            "context": context,
            "metadata": final_metadata
        }