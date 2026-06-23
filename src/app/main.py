from pathlib import Path
import re

from src.ingestion.pdf_loader import load_pdf_pages
from src.ingestion.chunker import chunk_page
from src.ingestion.table_parser import extract_tables
from src.ingestion.table_chunker import chunk_tables
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


company_aliases = {
    "tcs": ["tcs", "tata consultancy services", "tata_consultancy_services"],
    "infosys": ["infosys"],
    "wipro": ["wipro"],
}

doc_type_patterns = {
    "annual_report": [
        "annual_report",
        "annualreport",
        "annual-report",
        "ar",
    ],
    "earnings_call": [
        "earnings_call",
        "earningscall",
        "earnings-call",
        "transcript",
        "call",
    ],
    "investor_presentation": [
        "presentation",
        "investor_presentation",
        "investorpresentation",
        "ppt",
    ],
}


def infer_company_from_source(source_name: str):
    source_lower = source_name.lower()

    for canonical_company, aliases in company_aliases.items():
        for alias in aliases:
            if alias in source_lower:
                return canonical_company

    return None


def infer_doc_type_from_source(source_name: str):
    source_lower = source_name.lower()

    for doc_type, patterns in doc_type_patterns.items():
        for pattern in patterns:
            if pattern in source_lower:
                return doc_type

    return "annual_report"


def infer_fiscal_year_from_source(source_name: str):
    source_lower = source_name.lower()

    match = re.search(r"fy[_\s-]?(\d{2,4})", source_lower)
    if match:
        year = match.group(1)
        if len(year) == 2:
            return f"FY{year}"
        return f"FY{year[-2:]}"

    match = re.search(r"fiscal[_\s-]?(\d{4})", source_lower)
    if match:
        year = match.group(1)
        return f"FY{year[-2:]}"

    return None


def infer_quarter_from_source(source_name: str):
    source_lower = source_name.lower()
    match = re.search(r"q([1-4])", source_lower)
    if match:
        return f"Q{match.group(1)}"
    return None


def parse_source_metadata(source_name: str):
    return {
        "source": source_name,
        "company": infer_company_from_source(source_name),
        "doc_type": infer_doc_type_from_source(source_name),
        "fiscal_year": infer_fiscal_year_from_source(source_name),
        "quarter": infer_quarter_from_source(source_name),
        "report_date": None,
    }


def build_chunk_metadata(source_meta, page=None, chunk_type="text", extra=None):
    metadata = {
        "source": source_meta.get("source") or "",
        "company": source_meta.get("company") or "",
        "doc_type": source_meta.get("doc_type") or "",
        "fiscal_year": source_meta.get("fiscal_year") or "",
        "quarter": source_meta.get("quarter") or "",
        "report_date": source_meta.get("report_date") or "",
        "page": page if page is not None else -1,
        "chunk_type": chunk_type or "",
    }

    if extra:
        for key, value in extra.items():
            if value is None:
                metadata[key] = ""
            else:
                metadata[key] = value

    return metadata


def normalize_table_metadata(table_metadata_list, source_meta):
    normalized = []

    for meta in table_metadata_list:
        page = meta.get("page")
        extra = {
            k: v
            for k, v in meta.items()
            if k not in {"source", "page", "chunk_type"}
        }

        normalized.append(
            build_chunk_metadata(
                source_meta=source_meta,
                page=page,
                chunk_type="table",
                extra=extra
            )
        )

    return normalized


def main():
    pdf_files = list(Path("data/raw").rglob("*.pdf"))

    if not pdf_files:
        print("No PDFs found.")
        return

    all_chunks = []
    all_metadata = []

    for pdf_path in pdf_files:
        source_name = pdf_path.stem
        source_meta = parse_source_metadata(source_name)

        print("\n" + "=" * 60)
        print(f"Processing: {source_name}")
        print(f"Parsed source metadata: {source_meta}")

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
                text_metadata.append(
                    build_chunk_metadata(
                        source_meta=source_meta,
                        page=page_num,
                        chunk_type="text"
                    )
                )

        print(f"Text chunks: {len(text_chunks)}")

        print("\nExtracting tables...")
        tables = extract_tables(str(pdf_path))
        print(f"Tables found: {len(tables)}")

        table_chunks, table_metadata = chunk_tables(tables, source_name)
        table_metadata = normalize_table_metadata(table_metadata, source_meta)

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