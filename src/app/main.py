from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


def main():

    pdf_path = "data/raw/infosys_fy24.pdf"

    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print(f"Characters: {len(text)}")

    print("\nChunking...")
    chunks = chunk_text(text)

    print(f"Chunks created: {len(chunks)}")

    print("\nEmbedding...")
    embedder = EmbeddingModel()

    embeddings = embedder.embed(chunks)

    print(f"Embeddings shape: {embeddings.shape}")

    print("\nSaving to ChromaDB...")

    store = VectorStore()

    metadata = [
    {"source": "infosys_fy24"}
    for _ in chunks
]

    store.add_documents(
        chunks,
        embeddings,
        metadata
    )

    print(f"Total documents in DB: {store.count()}")


if __name__ == "__main__":
    main()