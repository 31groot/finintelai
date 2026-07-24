import uuid
import chromadb

BATCH_SIZE = 5000
COLLECTION_NAME = "finintelai"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/chromadb")
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def source_exists(self, source_id: str):
        result = self.collection.get(
            where={"source": source_id},
            limit=1,
        )
        return len(result["ids"]) > 0

    def delete_source(self, source_id: str):
        result = self.collection.get(where={"source": source_id})
        ids = result.get("ids", [])

        if not ids:
            return 0

        for start in range(0, len(ids), BATCH_SIZE):
            self.collection.delete(ids=ids[start:start + BATCH_SIZE])

        return len(ids)

    def reset(self):
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )
        return True

    def update_metadata(self, ids, metadatas):
        if not ids:
            return 0

        for start in range(0, len(ids), BATCH_SIZE):
            end = start + BATCH_SIZE
            self.collection.update(
                ids=ids[start:end],
                metadatas=metadatas[start:end],
            )

        return len(ids)

    def add_documents(self, chunks, embeddings, metadata=None):
        if metadata is None:
            metadata = [{} for _ in chunks]

        for start in range(0, len(chunks), BATCH_SIZE):
            end = start + BATCH_SIZE

            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]
            batch_metadata = metadata[start:end]

            ids = [str(uuid.uuid4()) for _ in batch_chunks]

            self.collection.add(
                ids=ids,
                documents=batch_chunks,
                embeddings=batch_embeddings.tolist(),
                metadatas=batch_metadata,
            )

            print(f"Stored batch: {start} -> {min(end, len(chunks))}")

        print(f"Stored {len(chunks)} chunks")

    def count(self):
        return self.collection.count()

    def get_all_documents(self, batch_size=BATCH_SIZE):
        total = self.collection.count()

        all_ids = []
        all_documents = []
        all_metadatas = []

        for offset in range(0, total, batch_size):
            batch = self.collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )

            all_ids.extend(batch.get("ids", []))
            all_documents.extend(batch.get("documents", []))
            all_metadatas.extend(batch.get("metadatas", []))

        return {
            "ids": all_ids,
            "documents": all_documents,
            "metadatas": all_metadatas,
        }