import re
from rank_bm25 import BM25Okapi
class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents

        if not documents:
            self.bm25 = None
            return

        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def search(self, query, top_k=10):
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return ranked_indices[:top_k]