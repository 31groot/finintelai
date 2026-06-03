def table_to_text(table_df):
    rows = []

    for _, row in table_df.iterrows():
        rows.append(" | ".join(row.astype(str)))

    return "\n".join(rows)


def chunk_tables(tables, source):
    chunks = []
    metadata = []

    for table in tables:

        content = table_to_text(table["content"])

        chunks.append(content)

        metadata.append({
            "source": source,
            "page": table["page"],
            "chunk_type": "table"
        })

    return chunks, metadata