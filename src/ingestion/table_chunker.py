rows_per_chunk = 15


def table_to_text(table_df):
    rows = []

    for _, row in table_df.iterrows():
        values = [str(cell).strip() for cell in row.tolist()]
        if all(value == "" for value in values):
            continue
        rows.append(" | ".join(values))

    return rows


def batch_rows(rows):
    if not rows:
        return []

    if len(rows) <= 1:
        return ["\n".join(rows)]

    header = rows[0]
    data_rows = rows[1:]
    batches = []

    for start in range(0, len(data_rows), rows_per_chunk):
        end = start + rows_per_chunk
        row_group = data_rows[start:end]
        batch_text = "\n".join([header] + row_group)
        batches.append(batch_text)

    return batches


def chunk_tables(tables, source):
    chunks = []
    metadata = []

    for table in tables:
        rows = table_to_text(table["content"])
        if not rows:
            continue

        batches = batch_rows(rows)

        for batch_text in batches:
            chunks.append(batch_text)
            metadata.append(
                {
                    "source": source,
                    "page": table["page"],
                    "chunk_type": "table",
                }
            )

    return chunks, metadata