import os
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

load_dotenv()

class QdrantStore:
    def __init__(self, collection_name: str, vector_size: int):
        self.collection_name = collection_name
        self.vector_size = vector_size

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333")
        )

    def create_collection(self, recreate: bool = False):
        existing = {c.name for c in self.client.get_collections().collections}

        if recreate and self.collection_name in existing:
            self.client.delete_collection(self.collection_name)

        # refresh state after delete (important)
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            print(f"Collection '{self.collection_name}' already exists.")
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

        # verify creation to avoid mysterious 404 later
        created = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in created:
            raise RuntimeError(f"Failed to create collection: {self.collection_name}")

        print(f"Created collection '{self.collection_name}'.")

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], batch_size: int = 128):

        assert len(vectors) == len(payloads), "vectors and payloads must align"

        n = len(vectors)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            points = [
                PointStruct(
                    id=i,
                    vector = vectors[i],
                    payload = payloads[i],
                    )
                for i in range(start, end)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def search(self, query_vector: list[float], top_k: int = 3):
        # Newer qdrant-client API
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            # query_points returns an object with .points
            return res.points

        # Older qdrant-client fallback API
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
    
    

    def recreate_collection(self):
        # Delete if exists (ignore if it doesn't)
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass

        # Create fresh
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )



