from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()

    def get_context(self, query):

        results = self.retriever.search(query)

        docs = results["documents"][0]
        metadata = results["metadatas"][0]

        context = "\n\n".join(docs[:5])

        return {
            "context": context,
            "metadata": metadata
        }