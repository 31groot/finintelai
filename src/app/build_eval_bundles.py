import json
from pathlib import Path

from src.app.dump_pdf_file import dump_pdf

OUTPUT_DIR = Path("data/extracted")
DOC_TYPE_SETTINGS = {
    "annual_report": {"min_score": 20, "max_pages": 105},
    "investor_presentation": {"min_score": 0, "max_pages": None},
    "earnings_call": {"min_score": 0, "max_pages": None},
}

DOC_TYPE_ORDER = ["annual_report", "investor_presentation", "earnings_call"]


def build_company_bundle(company):
    pdf_files = sorted(Path("data/raw").rglob("*.pdf"))

    documents = []

    for pdf_path in pdf_files:
        probe = dump_pdf(pdf_path, min_score=99999, max_pages=1)

        if (probe.get("company") or "").lower() != company.lower():
            continue

        doc_type = probe.get("doc_type") or "annual_report"
        settings = DOC_TYPE_SETTINGS.get(doc_type, {"min_score": 0, "max_pages": None})

        result = dump_pdf(
            pdf_path,
            min_score=settings["min_score"],
            max_pages=settings["max_pages"],
        )
        documents.append(result)

    documents.sort(
        key=lambda d: (
            DOC_TYPE_ORDER.index(d["doc_type"])
            if d["doc_type"] in DOC_TYPE_ORDER else 99,
            d.get("fiscal_year") or "",
            d.get("quarter") or "",
        )
    )

    return {
        "company": company,
        "document_count": len(documents),
        "documents": documents,
    }


def main(companies=None):
    companies = companies or ["tcs", "infosys", "wipro"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for company in companies:
        bundle = build_company_bundle(company)

        if not bundle["documents"]:
            print(f"{company}: no documents found")
            continue

        out_path = OUTPUT_DIR / f"{company}_bundle.json"
        out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))

        size_mb = out_path.stat().st_size / (1024 * 1024)

        print(f"\n{company.upper()}  ({bundle['document_count']} docs, {size_mb:.1f} MB)")
        for doc in bundle["documents"]:
            label = doc["source_id"]
            print(
                f"   {label:<45} "
                f"{doc['pages_included']:>4}/{doc['total_pages']:<4} pages"
            )

        if size_mb > 2:
            print(f"   WARNING: {size_mb:.1f} MB may be too large to upload.")
            print(f"   Re-run with a higher min_score for annual_report.")

    print(f"\nWritten to {OUTPUT_DIR}/")


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(companies=args or None)