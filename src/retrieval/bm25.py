import re
from rank_bm25 import BM25Okapi
class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        # Prevent BM25 crash when DB is empty
        if len(documents) == 0:
            self.bm25 = None
            return

        tokenized_docs = [
            self._tokenize(doc)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_docs
        )

    def _tokenize(self, text):

        return re.findall(
            r"\w+",
            text.lower()
        )

    def search(
        self,
        query,
        top_k=10
    ):

        if self.bm25 is None:
            return []

        tokenized_query = (
            self._tokenize(query)
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