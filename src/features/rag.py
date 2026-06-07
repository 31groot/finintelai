from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.query_decomposer import QueryDecomposer
class RAGPipeline:

    def __init__(self):

        self.retriever = Retriever()
        self.reranker = Reranker()
        self.decomposer = QueryDecomposer()

    def get_context(self, query):

        subqueries = self.decomposer.decompose(query)

        balanced_docs = []
        balanced_metadata = []


        for subquery in subqueries:

            results = self.retriever.search(
                query=subquery,
                n_results=15
            )

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


            seen_docs = set()

            docs = []
            metadata = []

            for doc, meta in zip(
                vector_docs,
                vector_metadata
            ):

                if doc in seen_docs:
                    continue

                seen_docs.add(doc)

                docs.append(doc)
                metadata.append(meta)


            if len(subqueries) == 1:
                rerank_top_k = 15

            elif len(subqueries) == 2:
                rerank_top_k = 8

            elif len(subqueries) == 3:
                rerank_top_k = 6

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

        for doc, meta in zip(
            balanced_docs,
            balanced_metadata
        ):

            if doc in seen_docs:
                continue

            seen_docs.add(doc)

            final_docs.append(doc)
            final_metadata.append(meta)

        query_lower = query.lower()

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

        is_comparison = any(
            keyword in query_lower
            for keyword in comparison_keywords
        )

        if is_comparison:

            final_docs = final_docs[:20]
            final_metadata = final_metadata[:20]

        else:

            final_docs = final_docs[:15]
            final_metadata = final_metadata[:15]

  
        context = "\n\n".join(final_docs)

        return {
            "context": context,
            "metadata": final_metadata
        }