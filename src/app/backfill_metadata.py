from collections import Counter

from src.app.main import infer_basis, infer_figure_type, infer_statement_type
from src.retrieval.vector_store import VectorStore


def backfill(dry_run=True):
    store = VectorStore()
    data = store.get_all_documents()

    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    print(f"Scanning {len(ids)} chunks...")

    update_ids = []
    update_metadatas = []

    for chunk_id, doc, meta in zip(ids, documents, metadatas):
        meta = dict(meta or {})
        doc = doc or ""

        new_meta = dict(meta)
        new_meta["basis"] = infer_basis(doc)
        new_meta["figure_type"] = infer_figure_type(doc)

        chunk_statement = infer_statement_type(doc)
        if chunk_statement != "general":
            new_meta["statement_type"] = chunk_statement
        elif not meta.get("statement_type"):
            new_meta["statement_type"] = "general"

        if new_meta != meta:
            update_ids.append(chunk_id)
            update_metadatas.append(new_meta)

    print(f"Chunks needing update: {len(update_ids)}")

    print("\nResulting distribution:")
    print("  basis:      ", Counter(m["basis"] for m in update_metadatas))
    print("  figure_type:", Counter(m["figure_type"] for m in update_metadatas))
    print("  doc_type x figure_type:")
    combo = Counter(
        (m.get("doc_type", "?"), m["figure_type"]) for m in update_metadatas
    )
    for (doc_type, figure_type), count in sorted(combo.items()):
        print(f"    {doc_type:<24} {figure_type:<18} {count}")

    if dry_run:
        print("\nDry run - nothing written. Re-run with dry_run=False to apply.")
        return

    updated = store.update_metadata(update_ids, update_metadatas)
    print(f"\nUpdated {updated} chunks.")


if __name__ == "__main__":
    import sys

    backfill(dry_run="--apply" not in sys.argv)