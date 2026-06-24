from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_splitter(doc_type: str = "general"):
    doc_type = (doc_type or "general").lower()

    if doc_type == "quarterly_transcript":
        return RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    if doc_type == "quarterly_presentation":
        return RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    if doc_type == "annual_report":
        return RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def chunk_text(text: str, doc_type: str = "general"):
    if not text or not text.strip():
        return []

    splitter = get_splitter(doc_type)
    return splitter.split_text(text.strip())


def chunk_page(text: str, doc_type: str = "general"):
    return chunk_text(text, doc_type=doc_type)


def chunk_page_with_metadata(text: str, metadata: dict, doc_type: str = "general"):
    chunks = chunk_text(text, doc_type=doc_type)

    chunked_items = []
    for chunk in chunks:
        chunked_items.append({
            "text": chunk,
            "metadata": dict(metadata)
        })

    return chunked_items