import json
import os
from statistics import mean
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from ragbench.bench.schema import BenchItem
from ragbench.pipeline.simple_rag import run_rag

BENCH_PATH = "benchmarks/sample.jsonl"

COLLECTIONS = [
    {
        "name": "demo_k8s_helm_te3l",
        "label": "text-embedding-3-large",
        "embed_kind": "openai",
        "embed_model": "text-embedding-3-large",
    },
    {
        "name": "demo_k8s_helm_bge_small",
        "label": "bge-small-en-v1.5 (local)",
        "embed_kind": "local",
        "embed_model": "BAAI/bge-small-en-v1.5",
    },
]

K_LIST = [1, 3, 5, 10]

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def recall_at_k(retrieved_chunks, gold_substring: str, k: int) -> int:
    if not gold_substring:
        return 0
    
    if isinstance(gold_substring, str):
        gold_substring = [gold_substring]

    topk = retrieved_chunks[:k]
    
    for chunk in topk:
        chunk_lower = (chunk or "").lower()
        for sub in gold_substring:
            if sub.lower() in chunk_lower:
                return 1
    return 0

def main():
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []

    for c in COLLECTIONS:
        label = c["label"]
        collection = c["name"]

        # Only compute recall on answerable items with gold_doc_contains
        rec = {k: [] for k in K_LIST}
        latencies = []
        rerank_ms = []

        for raw in load_jsonl(BENCH_PATH):
            item = BenchItem(**raw)
            gold_sub = raw.get("gold_doc_contains") or raw.get("must_contain")

            # skip unanswerables for recall@k
            if item.gold_answer is None or not gold_sub:
                continue

            retr = run_rag(
                question=item.question,
                collection=collection,
                dense_top_k=50,
                use_rerank=True,
                rerank_top_n=3,
                embed_kind=c["embed_kind"],
                embed_model=c["embed_model"]
            )

            chunks = retr["context_chunks"]
            for k in K_LIST:
                rec[k].append(recall_at_k(chunks, gold_sub, k))

            latencies.append(retr["timings_ms"]["total_retrieval"])
            rerank_ms.append(retr["timings_ms"]["rerank"])

        row = {
            "embedding": label,
            "collection": collection,
            "avg_retrieval_ms": mean(latencies) if latencies else 0,
            "avg_rerank_ms": mean(rerank_ms) if rerank_ms else 0,
        }
        for k in K_LIST:
            row[f"recall@{k}"] = mean(rec[k]) if rec[k] else 0

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = f"results/embedding_comparison_{ts}.csv"
    df.to_csv(csv_path, index=False)

    # Plot recall@k curves
    plt.figure()
    for _, r in df.iterrows():
        xs = K_LIST
        ys = [r[f"recall@{k}"] for k in K_LIST]
        plt.plot(xs, ys, marker="o", label=r["embedding"])
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title("Embedding Retrieval Recall@k")
    plt.xticks(K_LIST)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/recall_at_k_{ts}.png")
    plt.close()

    # Plot latency
    plt.figure()
    plt.bar(df["embedding"], df["avg_retrieval_ms"])
    plt.title("Avg Retrieval Latency (Embedding-only)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"results/embedding_latency_{ts}.png")
    plt.close()

    print("\nSaved:")
    print(csv_path)
    print("results/recall_at_k_*.png")
    print("results/embedding_latency_*.png")
    print("\nSummary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()