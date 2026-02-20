import json
import glob
from ragbench.store.qdrant_store import QdrantStore
from ragbench.embed.providers import OpenAIEmbeddingProvider
from ragbench.embed.local_provider import LocalSTEmbeddingProvider

def load_latest_augmented():
    files = sorted(glob.glob("data/corpus_augmented_*.jsonl"))
    if not files:
        raise FileNotFoundError("No augmented corpus found. Run scripts/augment_corpus.py first.")
    return files[-1]

def load_jsonl(path):
    texts = []
    payloads = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                texts.append(row["text"])
                payloads.append({"text": row["text"], "source": row.get("source", "unknown")})
    return texts, payloads

def build_collection(collection: str, vectors, payloads, dim: int):
    store = QdrantStore(collection_name=collection, vector_size=dim)
    store.recreate_collection()
    store.upsert(vectors, payloads, batch_size=128)
    print(f"Built: {collection} (n={len(payloads)}) dim={dim}")

def main():
    path = load_latest_augmented()
    texts, payloads = load_jsonl(path)

    # OpenAI large
    oai = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    vecs_oai = oai.embed_texts(texts)
    build_collection("demo_k8s_helm_te3l", vecs_oai, payloads, oai.dim)

    # Local bge-small
    local = LocalSTEmbeddingProvider("BAAI/bge-small-en-v1.5")
    vecs_local = local.embed_documents(texts)
    build_collection("demo_k8s_helm_bge_small", vecs_local, payloads, local.dim)

    print("Indexed from:", path)

if __name__ == "__main__":
    main()