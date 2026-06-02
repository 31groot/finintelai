import chromadb


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="data/chromadb"
        )

        self.collection = self.client.get_or_create_collection(
            name="finintelai"
        )

    def add_documents(
        self,
        chunks,
        embeddings,
        metadata=None
    ):

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        if metadata is None:
            metadata = [{} for _ in chunks]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadata
        )

        print(f"Stored {len(chunks)} chunks")

    def count(self):
        return self.collection.count()