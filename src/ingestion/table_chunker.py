ROWS_PER_CHUNK = 15

def table_to_text(table_df):
    rows = []

    for row in table_df.values.tolist():
        values = [str(cell) for cell in row]
        if all(value == "" for value in values):
            continue
        rows.append(" | ".join(values))

    return rows

def batch_rows(rows):
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
                    "detected_by": table["detected_by"]
                }
            )

    return chunks, metadata