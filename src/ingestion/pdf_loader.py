import fitz


def load_pdf(pdf_path: str) -> str:

    text = ""

    with fitz.open(pdf_path) as doc:

        for page in doc:
            text += page.get_text()

    return text


def load_pdf_pages(pdf_path):

    pages = []

    with fitz.open(pdf_path) as doc:

        for page_num, page in enumerate(doc, start=1):

            pages.append(
                {
                    "page": page_num,
                    "text": page.get_text()
                }
            )

    return pages