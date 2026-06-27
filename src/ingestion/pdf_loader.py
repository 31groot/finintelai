import fitz


def load_pdf(pdf_path: str) -> str:
    text_parts = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text() or ""
            if page_text.strip():
                text_parts.append(page_text)

    return "\n".join(text_parts)


def load_pdf_pages(pdf_path: str):
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text() or ""
            pages.append(
                {
                    "page": page_num,
                    "text": page_text,
                }
            )

    return pages
