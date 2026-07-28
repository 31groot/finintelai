import json
import re
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf_pages
from src.app.main import parse_source_metadata, build_source_id

OUTPUT_DIR = Path("data/extracted")

FINANCIAL_KEYWORDS = [
    "revenue",
    "margin",
    "attrition",
    "headcount",
    "employees",
    "tcv",

    "profit",
    "ebit",
    "growth",
    "constant currency",
    "guidance",
    "segment",
    "geography",
    "operating",
    "crore",
    "million",
    "billion",
]


def score_page(text):

    if not text or len(text.strip()) < 100:
        return 0

    lower = text.lower()

    numbers = len(re.findall(r"\b\d[\d,]*\.?\d*\b", text))
    percents = len(re.findall(r"\d+\.?\d*\s*%", text))
    keywords = sum(1 for kw in FINANCIAL_KEYWORDS if kw in lower)

    return numbers + (percents * 3) + (keywords * 2)


def dump_pdf(pdf_path, min_score=0, max_pages=None):
    source_name = str(pdf_path).replace("\\", "/")
    source_meta = parse_source_metadata(source_name)
    source_id = build_source_id(source_meta)

    pages = load_pdf_pages(str(pdf_path))

    scored = []
    for page in pages:
        scored.append(
            {
                "page": page["page"],
                "text": page["text"],
                "score": score_page(page["text"]),
            }
        )

    kept = [p for p in scored if p["score"] >= min_score]

    if max_pages:
        kept = sorted(kept, key=lambda p: p["score"], reverse=True)[:max_pages]
        kept = sorted(kept, key=lambda p: p["page"])

    return {
        "source_id": source_id,
        "source_file": source_name,
        "company": source_meta.get("company"),
        "doc_type": source_meta.get("doc_type"),
        "fiscal_year": source_meta.get("fiscal_year"),
        "quarter": source_meta.get("quarter"),
        "total_pages": len(pages),
        "pages_included": len(kept),
        "pages": [{"page": p["page"], "text": p["text"]} for p in kept],
    }


def main(min_score=0, max_pages=None):
    pdf_files = sorted(Path("data/raw").rglob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data/raw")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_files:
        result = dump_pdf(pdf_path, min_score=min_score, max_pages=max_pages)

        out_path = OUTPUT_DIR / f"{result['source_id']}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        size_kb = out_path.stat().st_size / 1024
        print(
            f"{result['source_id']:<45} "
            f"{result['pages_included']:>4}/{result['total_pages']:<4} pages  "
            f"{size_kb:>8.0f} KB"
        )

    print(f"\nWritten to {OUTPUT_DIR}/")


if __name__ == "__main__":
    import sys

    min_score = 0
    max_pages = None

    for arg in sys.argv[1:]:
        if arg.startswith("--min-score="):
            min_score = int(arg.split("=")[1])
        if arg.startswith("--max-pages="):
            max_pages = int(arg.split("=")[1])

    main(min_score=min_score, max_pages=max_pages)