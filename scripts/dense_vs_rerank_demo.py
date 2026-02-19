import time
from ragbench.store.qdrant_store import QdrantStore
from ragbench.embed.providers import OpenAIEmbeddingProvider
from ragbench.rerank.providers import CrossEncoderReranker

COLLECTION = "demo_k8s_helm"
EMBED_MODEL = "text-embedding-3-small"
RERANK_MODEL = "BAAI/bge-reranker-base"

docs = [
    "A Kubernetes Deployment manages a ReplicaSet to keep a set of Pods running.",
    "Helm templates are rendered using values from values.yaml and the Go template language.",
    "Terraform state stores information about managed infrastructure and can be local or remote.",
    "Helm uses Go templating; values are injected via .Values and can be overridden via --set or values files."
]

query = "How do Helm charts use values.yaml?"

def now_ms():
    return int(time.perf_counter() * 1000)

def main():
    embedder = OpenAIEmbeddingProvider(model=EMBED_MODEL)
    store = QdrantStore(collection_name=COLLECTION, vector_size=embedder.dim)
    store.create_collection(recreate=True)

    t0 = now_ms()
    vectors = embedder.embed_texts(docs)
    t_embed_docs = now_ms() - t0

    store.upsert(vectors, [{"text": t, "source": "demo"} for t in docs])

    # ---- Dense retrieval ----
    t1 = now_ms()
    qvec = embedder.embed_query(query)
    t_embed_q = now_ms() - t1

    t2 = now_ms()
    dense_results = store.search(qvec, top_k=10)
    t_retrieve = now_ms() - t2

    candidates = []
    for r in dense_results:
        candidates.append((r.payload.get("text", ""), dict(r.payload), getattr(r, "score", None)))

    print("\n=== Dense top-3 ===")
    for r in dense_results[:3]:
        print("dense_score:", getattr(r, "score", None), "|", r.payload.get("text"))

    # ---- Rerank ----
    reranker = CrossEncoderReranker(model=RERANK_MODEL)
    t3 = now_ms()
    reranked = reranker.rerank(query=query, candidates=candidates, top_n=3)
    t_rerank = now_ms() - t3

    print("\n=== Reranked top-3 ===")
    for it in reranked:
        print("rerank_score:", it.rerank_score, "| dense_score:", it.base_score, "|", it.text)

    print("\n=== Timings (ms) ===")
    print("embed_docs:", t_embed_docs)
    print("embed_query:", t_embed_q)
    print("retrieve:", t_retrieve)
    print("rerank:", t_rerank)

    # ---- Save structured run record ----
    import json, os
    from datetime import datetime
    from ragbench.utils.run_schema import RunRecord, Timing, RetrievedChunk

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    record = RunRecord(
        query=query,
        dense_top_k=10,
        rerank_top_n=3,
        embedding_model=f"openai:{EMBED_MODEL}",
        reranker_model=RERANK_MODEL,
        timings=Timing(
            embed_query_ms=t_embed_q,
            retrieve_ms=t_retrieve,
            rerank_ms=t_rerank,
            total_ms=(t_embed_q + t_retrieve + t_rerank),
        ),
        dense_results=[
            RetrievedChunk(text=c[0], payload=c[1], score=c[2])
            for c in candidates
        ],
        reranked_results=[
            RetrievedChunk(text=it.text, payload=it.payload, score=it.rerank_score)
            for it in reranked
        ],
    )

    with open(f"results/run_{ts}.json", "w") as f:
        f.write(record.model_dump_json(indent=2))

    print(f"\nSaved: results/run_{ts}.json")

if __name__ == "__main__":
    main()
