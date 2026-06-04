import uuid
import chromadb


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="data/chromadb"
        )

        self.collection = self.client.get_or_create_collection(
            name="finintelai"
        )

    def reset(self):

        try:
            self.client.delete_collection(
                name="finintelai"
            )
        except:
            pass

        self.collection = self.client.get_or_create_collection(
            name="finintelai"
        )

        print("Collection reset.")

    def add_documents(
        self,
        chunks,
        embeddings,
        metadata=None
    ):

        if metadata is None:
            metadata = [{} for _ in chunks]

        batch_size = 5000

        for start in range(
            0,
            len(chunks),
            batch_size
        ):

            end = start + batch_size

            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]
            batch_metadata = metadata[start:end]

            ids = [
                str(uuid.uuid4())
                for _ in batch_chunks
            ]

            self.collection.add(
                ids=ids,
                documents=batch_chunks,
                embeddings=batch_embeddings.tolist(),
                metadatas=batch_metadata
            )

            print(
                f"Stored batch: {start} -> {end}"
            )

        print(
            f"Stored {len(chunks)} chunks"
        )

    def count(self):
        return self.collection.count()