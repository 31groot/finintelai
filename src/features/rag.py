from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker


class RAGPipeline:

    def __init__(self):

        self.retriever = Retriever()
        self.reranker = Reranker()

    def get_context(self, query):

        results = self.retriever.search(
            query,
            n_results=20
        )

        # --------------------
        # VECTOR RESULTS
        # --------------------

        vector_docs = (
            results["vector_results"]["documents"][0]
        )

        vector_metadata = (
            results["vector_results"]["metadatas"][0]
        )

        # --------------------
        # BM25 RESULTS
        # --------------------

        bm25_docs = (
            results["bm25_results"]
        )

        # --------------------
        # MERGE
        # --------------------

        combined_docs = list(
            dict.fromkeys(
                vector_docs + bm25_docs
            )
        )

        print("\n==============================")
        print("TOTAL COMBINED DOCS")
        print(len(combined_docs))
        print("==============================")

        # --------------------
        # RERANK
        # --------------------

        reranked_docs = self.reranker.rerank(
            query,
            combined_docs,
            top_k=10
        )

        

            print(f"\nDOC {i}")
            print("-" * 40)

            print(doc[:500])

        # --------------------
        # CONTEXT
        # --------------------

        context = "\n\n".join(
            reranked_docs
        )

        return {
            "context": context,
            "metadata": vector_metadata
        }