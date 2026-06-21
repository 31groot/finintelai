import camelot

def extract_tables(pdf_path):
    extracted_tables = []

    extracted_tables += _extract_with_flavor(pdf_path, "lattice")
    extracted_tables += _extract_with_flavor(pdf_path, "stream")

    return extracted_tables


def _extract_with_flavor(pdf_path, flavor):
    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages="all",
            flavor=flavor
        )
    except Exception:
        return []

    extracted = []

    for table in tables:
        extracted.append({
            "content": table.df,
            "page": table.page,
            "detected_by": flavor
        })

    return extracted