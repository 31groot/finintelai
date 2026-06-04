from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text

from src.ingestion.table_parser import extract_tables
from src.ingestion.table_chunker import chunk_tables

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


def main():

    pdf_files = list(
        Path("data/raw").glob("*.pdf")
    )

    if not pdf_files:
        print("No PDFs found.")
        return

    all_chunks = []
    all_metadata = []

    # ==========================================
    # PROCESS ALL PDFS
    # ==========================================

    for pdf_path in pdf_files:

        source_name = pdf_path.stem

        print("\n" + "=" * 60)
        print(f"Processing: {source_name}")
        print("=" * 60)


        print("Loading PDF...")

        text = load_pdf(str(pdf_path))

        print(f"Characters: {len(text)}")


        print("\nChunking text...")

        text_chunks = chunk_text(text)

        print(f"Text chunks: {len(text_chunks)}")

        text_metadata = [
            {
                "source": source_name,
                "chunk_type": "text"
            }
            for _ in text_chunks
        ]


        print("\nExtracting tables...")

        tables = extract_tables(
            str(pdf_path)
        )

        print(f"Tables found: {len(tables)}")

        table_chunks, table_metadata = chunk_tables(
            tables,
            source_name
        )

        print(
            f"Table chunks: {len(table_chunks)}"
        )



        all_chunks.extend(text_chunks)
        all_chunks.extend(table_chunks)

        all_metadata.extend(text_metadata)
        all_metadata.extend(table_metadata)

        print(
            f"Running total chunks: {len(all_chunks)}"
        )

    print("\n" + "=" * 60)
    print("Embedding all chunks...")
    print("=" * 60)

    embedder = EmbeddingModel()

    embeddings = embedder.embed(
        all_chunks
    )

    print(
        f"Embeddings shape: {embeddings.shape}"
    )


    print("\nSaving to ChromaDB...")

    store = VectorStore()

   
    store.reset()

    store.add_documents(
        all_chunks,
        embeddings,
        all_metadata
    )

    print(
        f"Total documents in DB: {store.count()}"
    )


if __name__ == "__main__":
    main()