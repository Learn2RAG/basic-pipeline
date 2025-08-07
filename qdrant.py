from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class Qdrant:
    client = QdrantClient(host="localhost", port=6336)

    def __init__(self, collection_name, vector_size):
        self.collection_name = collection_name
        self.vector_size = vector_size

        if not Qdrant.client.collection_exists(self.collection_name):
            Qdrant.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                }
            )
