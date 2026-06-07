import camelot

def extract_tables(pdf_path):
    tables = camelot.read_pdf(
        pdf_path,
        pages="all"
    )

    extracted_tables = []

    for table in tables:
        extracted_tables.append({
            "content": table.df,
            "page": table.page
        })

    return extracted_tables