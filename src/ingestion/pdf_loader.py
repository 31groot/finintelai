import fitz  # pymupdf


def load_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF.
    Returns the complete text as a string.
    """

    text = ""

    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()

    return text


if __name__ == "__main__":

    pdf_path = "data/raw/infosys_fy24.pdf"

    text = load_pdf(pdf_path)

    print("=" * 60)
    print("PDF LOADED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nTotal Characters: {len(text)}")

    print("\nFIRST 2000 CHARACTERS:\n")

    print(text[:2000])