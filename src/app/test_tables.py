from src.ingestion.table_parser import extract_tables

tables = extract_tables(
    "data/raw/infosys_fy24.pdf"
)

print(f"Tables found: {len(tables)}")

for i in range(min(5, len(tables))):
    print("\n" + "="*50)
    print(f"TABLE {i+1}")
    print("="*50)
    print(tables[i]["content"][:500])