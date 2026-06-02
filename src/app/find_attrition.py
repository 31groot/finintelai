from src.ingestion.pdf_loader import load_pdf

text = load_pdf("data/raw/infosys_fy24.pdf")

keywords = [
    "attrition",
    "employee attrition",
    "voluntary attrition",
    "headcount",
    "workforce"
]

for keyword in keywords:
    print(f"\nSearching for: {keyword}")

    if keyword.lower() in text.lower():
        print("FOUND")
    else:
        print("NOT FOUND")