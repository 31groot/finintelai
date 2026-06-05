from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        tokenized_docs = [
            doc.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_docs
        )

    def search(
        self,
        query,
        top_k=10
    ):

        tokenized_query = (
            query.lower().split()
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, score
            in ranked[:top_k]
        ]