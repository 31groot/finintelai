from src.retrieval.retriever import Retriever
class RAGPipeline:

    def __init__(self):

        self.retriever = Retriever()

    def get_context(self, query):

        results = self.retriever.search(
            query,
            n_results=8
        )
        # ---------------------
        # VECTOR RESULTS
        # ---------------------

        vector_docs = (
            results["vector_results"]
            ["documents"][0]
        )

        vector_metadata = (
            results["vector_results"]
            ["metadatas"][0]
        )

        # ---------------------
        # BM25 RESULTS
        # ---------------------

        bm25_docs = (
            results["bm25_results"]
        )

        # ---------------------
        # MERGE
        # ---------------------

        combined_docs = list(
            dict.fromkeys(
                vector_docs + bm25_docs
            )
        )

        # ---------------------
        # CONTEXT
        # ---------------------

        context = "\n\n".join(
            combined_docs[:5]
        )

        return {
            "context": context,
            "metadata": vector_metadata
        }