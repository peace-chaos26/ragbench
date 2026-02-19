from ragbench.store.qdrant_store import QdrantStore
from ragbench.embed.providers import OpenAIEmbeddingProvider

COLLECTION = "demo_k8s_helm"
EMBED_MODEL = "text-embedding-3-small"

docs = [
    "A Kubernetes Deployment manages a ReplicaSet to keep a set of Pods running.",
    "Helm templates are rendered using values from values.yaml and the Go template language.",
    "Terraform state stores information about managed infrastructure and can be local or remote."
]

query = "How do Helm charts use values.yaml?"

def main():
    embedder = OpenAIEmbeddingProvider(model=EMBED_MODEL)
    store = QdrantStore(collection_name=COLLECTION, vector_size=embedder.dim)

    store.create_collection(recreate=True)

    vectors = embedder.embed_texts(docs)
    payloads = [{"text": t, "source": "demo"} for t in docs]
    store.upsert(vectors, payloads)

    qvec = embedder.embed_query(query)
    results = store.search(qvec, top_k=3)

    print("\nQuery:", query)
    for r in results:
        print("Score:", getattr(r, "score", None))
        print("Text:", r.payload.get("text"))
        print("---")

if __name__ == "__main__":
    main()