
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:
    
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts):
        """
        Convert a list of text chunks into embeddings.
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )


if __name__ == "__main__":
    embedder = EmbeddingModel()

    sample_texts = [
        "Infosys revenue increased in FY24.",
        "Operating margin improved during the year."
    ]

    embeddings = embedder.embed(sample_texts)

    print("Embeddings generated successfully!")
    print(f"Shape: {embeddings.shape}")
    print("\nFirst embedding (first 10 dimensions):")
    print(embeddings[0][:10])