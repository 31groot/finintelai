from sentence_transformers import SentenceTransformer
model = "BAAI/bge-small-en-v1.5"
class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(model)

    def embed(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )