from sentence_transformers import CrossEncoder
class Reranker:

    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        query,
        documents,
        metadata,
        top_k=25
    ):

        if not documents:
            return []

        pairs = [
            [query, doc]
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, metadata, scores),
            key=lambda x: x[2],
            reverse=True
        )

        return ranked[:top_k]