import re
from pathlib import Path

import fitz


COMPANIES = {"infosys", "tcs", "wipro"}


def infer_metadata_from_path(pdf_path: str):
    path = Path(pdf_path)
    parts = [part.lower() for part in path.parts]
    filename = path.stem

    company = None
    for part in parts:
        if part in COMPANIES:
            company = part
            break

    doc_type = None
    if "annual_report" in parts:
        doc_type = "annual_report"
    elif "quarterly_presentation" in parts:
        doc_type = "quarterly_presentation"
    elif "quarterly_transcript" in parts:
        doc_type = "quarterly_transcript"

    fiscal_year = None
    fy_match = re.search(r"FY[\s_-]?(\d{2,4})", filename, re.IGNORECASE)
    if fy_match:
        fy = fy_match.group(1)
        fiscal_year = f"FY{fy[-2:]}".upper()

    quarter = None
    q_match = re.search(r"\bQ([1-4])\b", filename, re.IGNORECASE)
    if q_match:
        quarter = f"Q{q_match.group(1)}"

    source_kind = "general"
    if doc_type == "annual_report":
        source_kind = "annual_filing"
    elif doc_type == "quarterly_presentation":
        source_kind = "metrics_summary"
    elif doc_type == "quarterly_transcript":
        source_kind = "management_commentary"

    if company is None:
        raise ValueError(f"Could not infer company from path: {pdf_path}")

    if doc_type is None:
        raise ValueError(f"Could not infer doc_type from path: {pdf_path}")

    if fiscal_year is None:
        raise ValueError(f"Could not infer fiscal year from filename: {pdf_path}")

    source = f"{company}_{fiscal_year.lower()}"
    if quarter:
        source += f"_{quarter.lower()}"

    if doc_type == "annual_report":
        source += "_annual"
    elif doc_type == "quarterly_presentation":
        source += "_presentation"
    elif doc_type == "quarterly_transcript":
        source += "_transcript"

    return {
        "company": company,
        "doc_type": doc_type,
        "fiscal_year": fiscal_year,
        "quarter": quarter,
        "source_kind": source_kind,
        "source": source,
    }


def load_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                text += page_text + "\n\n"
    return text.strip()


def load_pdf_pages(pdf_path: str):
    pages = []
    base_metadata = infer_metadata_from_path(pdf_path)

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text().strip()
            if not page_text:
                continue

            page_metadata = dict(base_metadata)
            page_metadata["page"] = page_num

            pages.append(
                {
                    "page": page_num,
                    "text": page_text,
                    "metadata": page_metadata
                }
            )

    return pages