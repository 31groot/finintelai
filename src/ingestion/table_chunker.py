import re

ROWS_PER_CHUNK = 15

COMPANY_NAMES = {
    "tcs": "TCS Tata Consultancy Services",
    "infosys": "Infosys",
    "wipro": "Wipro",
}

DOC_TYPE_NAMES = {
    "annual_report": "annual report",
    "investor_presentation": "investor presentation",
    "earnings_call": "earnings call",
}

METRIC_HINTS = [
    "revenue", "profit", "margin", "headcount", "employee", "attrition",
    "ebit", "ebitda", "tcv", "bookings", "cash", "dividend", "segment",
    "geography", "vertical", "operating", "income", "tax", "expense",
    "americas", "europe", "india", "bfsi", "manufacturing", "retail",
]


def _describe_source(source):
    src = (source or "").lower()

    company = ""
    for key, name in COMPANY_NAMES.items():
        if key in src:
            company = name
            break

    fy_match = re.search(r"fy\s*_?(\d{2,4})", src)
    fiscal_year = ""
    if fy_match:
        digits = fy_match.group(1)
        fiscal_year = f"FY{digits[-2:]}"

    doc_type = ""
    for key, name in DOC_TYPE_NAMES.items():
        if key in src:
            doc_type = name
            break

    quarter_match = re.search(r"q([1-4])", src)
    quarter = f"Q{quarter_match.group(1)}" if quarter_match else ""

    parts = [p for p in [company, fiscal_year, quarter, doc_type] if p]
    return " ".join(parts)


def _extract_row_labels(rows):

    labels = []
    for row in rows:
        first_cell = row.split("|")[0].strip().lower()
        if not first_cell:
            continue
        for hint in METRIC_HINTS:
            if hint in first_cell:
                label = row.split("|")[0].strip()
                if label and label not in labels:
                    labels.append(label)
                break
    return labels


def _build_description(source, rows):

    src_desc = _describe_source(source)
    labels = _extract_row_labels(rows)

    pieces = []
    if src_desc:
        pieces.append(f"{src_desc} financial data table.")
    if labels:
        pieces.append("Metrics: " + ", ".join(labels[:12]) + ".")

    return " ".join(pieces)


def table_to_text(table_df):
    rows = []

    for row in table_df.values.tolist():
        values = [str(cell) for cell in row]
        if all(value == "" for value in values):
            continue
        rows.append(" | ".join(values))

    return rows


def batch_rows(rows, description=""):
    if not rows:
        return []

    if len(rows) <= 1:
        return []

    header = rows[0]
    data_rows = rows[1:]
    batches = []

    for start in range(0, len(data_rows), ROWS_PER_CHUNK):
        end = start + ROWS_PER_CHUNK
        row_group = data_rows[start:end]

        body = "\n".join([header] + row_group)
        if description:
            batch_text = f"{description}\n\n{body}"
        else:
            batch_text = body

        batches.append(batch_text)

    return batches


def chunk_tables(tables, source):
    chunks = []
    metadata = []

    for table in tables:
        rows = table_to_text(table["content"])
        if not rows:
            continue

        description = _build_description(source, rows)
        batches = batch_rows(rows, description=description)

        for batch_text in batches:
            chunks.append(batch_text)
            metadata.append(
                {
                    "source": source,
                    "page": table["page"],
                    "chunk_type": "table",
                    "detected_by": table["detected_by"],
                }
            )

    return chunks, metadata