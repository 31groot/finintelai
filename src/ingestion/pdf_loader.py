import fitz

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
