from src.ingestion.pdf_loader import load_pdf

text = load_pdf("data/raw/infosys_fy24.pdf")

idx = text.lower().find("attrition")

print("=" * 80)
print(text[idx-1000:idx+2000])
print("=" * 80)