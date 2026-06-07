from src.ingestion.pdf_loader import load_pdf

text = load_pdf("data/raw/infosys_fy24.pdf")

keywords = [
    "attrition",
    "employee",
    "employees",
    "headcount",
    "workforce"
]

for k in keywords:
    print(k, "=>", k.lower() in text.lower())