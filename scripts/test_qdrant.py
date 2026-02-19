import numpy as np
from ragbench.store.qdrant_store import QdrantStore

VECTOR_SIZE = 5

store = QdrantStore(collection_name="test_collection", vector_size=VECTOR_SIZE)
store.create_collection(recreate=True)

vectors = np.random.rand(3, VECTOR_SIZE).tolist()
payloads = [
    {"text":"Kubernetes Deployment example"},
    {"text":"Helm chart configuration"},
    {"text":"Terraform state management"}
]

store.upsert(vectors, payloads)

query_vector = np.random.rand(VECTOR_SIZE).tolist()
results = store.search(query_vector, top_k=2)

for r in results:
    print("Score: ", getattr(r, "score", None))
    print("Payload: ", getattr(r, "payload", None))
    print("---------")