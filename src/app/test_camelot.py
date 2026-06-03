import camelot

tables = camelot.read_pdf(
    "data/raw/infosys_fy24.pdf",
    pages="180-220"
)

print("Tables found:", tables.n)

for i in range(min(5, tables.n)):
    print("\n" + "="*50)
    print(f"TABLE {i+1}")
    print("="*50)

    print(tables[i].df.head(10))

print(type(tables))
print(type(tables[0]))
print(tables[0])
from src.ingestion.table_chunker import table_to_text

print("\nTABLE AS TEXT\n")
print(table_to_text(tables[0].df))