from pathlib import Path
from src.ingestion.pdf_loader import load_pdf_pages
from src.ingestion.chunker import chunk_text, chunk_page
from src.ingestion.table_parser import extract_tables
from src.ingestion.table_chunker import chunk_tables
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


def main():
    pdf_files = list(Path("data/raw").glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found.")
        return

    all_chunks = []
    all_metadata = []

    for pdf_path in pdf_files:
        source_name = pdf_path.stem
        print("\n" + "=" * 60)
        print(f"Processing: {source_name}")
        print("=" * 60)

        print("Loading PDF pages...")
        pages = load_pdf_pages(str(pdf_path))

        text_chunks = []
        text_metadata = []

        for page_data in pages:
            page_num = page_data["page"]
            page_text = page_data["text"]
            chunks = chunk_page(page_text)
            for chunk in chunks:
                text_chunks.append(chunk)
                text_metadata.append({
                    "source": source_name,
                    "page": page_num,
                    "chunk_type": "text"
                })

        print(f"Text chunks: {len(text_chunks)}")

        print("\nExtracting tables...")
        tables = extract_tables(str(pdf_path))
        print(f"Tables found: {len(tables)}")

        table_chunks, table_metadata = chunk_tables(tables, source_name)
        print(f"Table chunks: {len(table_chunks)}")

        all_chunks.extend(text_chunks)
        all_chunks.extend(table_chunks)
        all_metadata.extend(text_metadata)
        all_metadata.extend(table_metadata)

        print(f"Running total chunks: {len(all_chunks)}")

    print("\n" + "=" * 60)
    print("Embedding all chunks...")
    print("=" * 60)

    embedder = EmbeddingModel()
    embeddings = embedder.embed(all_chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nSaving to ChromaDB...")
    store = VectorStore()
    store.reset()
    store.add_documents(all_chunks, embeddings, all_metadata)
    print(f"Total documents in DB: {store.count()}")


if __name__ == "__main__":
    main()