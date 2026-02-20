from ragbench.store.qdrant_store import QdrantStore
from ragbench.embed.providers import OpenAIEmbeddingProvider
from ragbench.embed.local_provider import LocalSTEmbeddingProvider

DOCS = [
    "Helm templates are rendered using values from values.yaml and the Go template language.",
    "Helm uses Go templating; values are injected via .Values and can be overridden via --set or values files.",
    "Terraform state stores information about managed infrastructure and can be local or remote.",
    "A Kubernetes Deployment manages a ReplicaSet to keep a set of Pods running."
]

def build_collection(collection: str, vectors, payloads, dim: int):
    store = QdrantStore(collection_name=collection, vector_size=dim)
    store.recreate_collection()  # implement this if you don’t have it; else delete+create manually
    store.upsert(vectors, payloads)
    print(f"Built: {collection} (n={len(payloads)})")

def main():
    payloads = [{"text": t, "source": "demo"} for t in DOCS]

    # A) OpenAI text-embedding-3-large
    oai = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    vecs_oai = oai.embed_documents(DOCS)
    build_collection("demo_k8s_helm_te3l", vecs_oai, payloads, oai.dim)

    # B) Local bge-small
    local = LocalSTEmbeddingProvider("BAAI/bge-small-en-v1.5")
    vecs_local = local.embed_documents(DOCS)
    build_collection("demo_k8s_helm_bge_small", vecs_local, payloads, local.dim)

if __name__ == "__main__":
    main()
