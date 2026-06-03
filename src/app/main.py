from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text

from src.ingestion.table_parser import extract_tables
from src.ingestion.table_chunker import chunk_tables

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


def main():

    pdf_path = "data/raw/infosys_fy24.pdf"

    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print(f"Characters: {len(text)}")

    # ----------------------------
    # TEXT CHUNKS
    # ----------------------------

    print("\nChunking text...")
    chunks = chunk_text(text)

    print(f"Text chunks: {len(chunks)}")

    # ----------------------------
    # TABLE CHUNKS
    # ----------------------------

    print("\nExtracting tables...")

    tables = extract_tables(pdf_path)

    print(f"Tables found: {len(tables)}")

    table_chunks, table_metadata = chunk_tables(
        tables,
        "infosys_fy24"
    )

    print(f"Table chunks: {len(table_chunks)}")

    # ----------------------------
    # MERGE
    # ----------------------------

    text_metadata = [
        {
            "source": "infosys_fy24",
            "chunk_type": "text"
        }
        for _ in chunks
    ]

    all_chunks = chunks + table_chunks
    all_metadata = text_metadata + table_metadata

    print(f"\nTotal chunks: {len(all_chunks)}")

    # ----------------------------
    # EMBEDDINGS
    # ----------------------------

    print("\nEmbedding...")

    embedder = EmbeddingModel()

    embeddings = embedder.embed(all_chunks)

    print(f"Embeddings shape: {embeddings.shape}")

    # ----------------------------
    # STORE
    # ----------------------------

    print("\nSaving to ChromaDB...")

    store = VectorStore()

    store.add_documents(
        all_chunks,
        embeddings,
        all_metadata
    )

    print(f"Total documents in DB: {store.count()}")


if __name__ == "__main__":
    main()