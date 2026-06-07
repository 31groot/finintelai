from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text

pdf_path = "data/raw/infosys_fy24.pdf"

print("Loading PDF...")

text = load_pdf(pdf_path)

print(f"Characters extracted: {len(text)}")

print("\nCreating chunks...")

chunks = chunk_text(text)

print(f"Total chunks: {len(chunks)}")

print("\nFIRST CHUNK:\n")
print(chunks[0])

print("\n" + "=" * 60)

if len(chunks) > 1:
    print("\nSECOND CHUNK:\n")
    print(chunks[1])