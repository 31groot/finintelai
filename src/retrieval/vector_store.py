import uuid
import chromadb


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/chromadb")
        self.collection = self.client.get_or_create_collection(name="finintelai")

    def reset(self):
        try:
            self.client.delete_collection(name="finintelai")
        except ValueError:
            pass

        self.collection = self.client.get_or_create_collection(name="finintelai")
        print("Collection reset.")

    def add_documents(self, chunks, embeddings, metadata=None):
        if metadata is None:
            metadata = [{} for _ in chunks]

        batch_size = 5000

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size

            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]
            batch_metadata = metadata[start:end]

            ids = [str(uuid.uuid4()) for _ in batch_chunks]

            self.collection.add(
                ids=ids,
                documents=batch_chunks,
                embeddings=batch_embeddings.tolist(),
                metadatas=batch_metadata
            )

            print(f"Stored batch: {start} -> {min(end, len(chunks))}")

        print(f"Stored {len(chunks)} chunks")

    def count(self):
        return self.collection.count()

    def get_all_documents(self, batch_size=5000):
        total = self.collection.count()

        all_ids = []
        all_documents = []
        all_metadatas = []

        for offset in range(0, total, batch_size):
            batch = self.collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset
            )

            all_ids.extend(batch.get("ids", []))
            all_documents.extend(batch.get("documents", []))
            all_metadatas.extend(batch.get("metadatas", []))

        return {
            "ids": all_ids,
            "documents": all_documents,
            "metadatas": all_metadatas
        }