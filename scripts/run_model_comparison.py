import json
import os
from statistics import mean
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from ragbench.bench.schema import BenchItem
from ragbench.pipeline.simple_rag import run_rag
from ragbench.generation.answer import generate_answer
from ragbench.eval.judge import judge_faithfulness
from ragbench.eval.pricing import estimate_cost

COLLECTION = "demo_k8s_helm"

GENERATOR_MODELS = [
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-5.1",
]

JUDGE_MODEL = "gpt-4.1-mini"

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_path = "benchmarks/sample.jsonl"

    summary_rows = []

    for model in GENERATOR_MODELS:
        faithful_flags = []
        total_costs = []
        total_tokens = []
        retrieval_latencies = []

        print(f"\nRunning model: {model}")

        for raw in load_jsonl(bench_path):
            item = BenchItem(**raw)

            retr = run_rag(
                question=item.question,
                collection=COLLECTION,
                dense_top_k=10,
                use_rerank=True,
                rerank_top_n=3,
            )

            answer, ans_usage = generate_answer(
                question=item.question,
                context_chunks=retr["context_chunks"],
                model=model,
            )

            verdict, judge_usage = judge_faithfulness(
                question=item.question,
                answer=answer,
                context_chunks=retr["context_chunks"],
                model=JUDGE_MODEL,
            )

            faithful = bool(verdict.get("faithful", False))
            faithful_flags.append(1 if faithful else 0)

            # token accounting
            ans_prompt = ans_usage.get("prompt_tokens", 0)
            ans_completion = ans_usage.get("completion_tokens", 0)

            cost = estimate_cost(model, ans_prompt, ans_completion)
            total_costs.append(cost)

            total_tokens.append(ans_usage.get("total_tokens", 0))
            retrieval_latencies.append(retr["timings_ms"]["total_retrieval"])

        summary_rows.append({
            "model": model,
            "faithfulness_rate": mean(faithful_flags),
            "avg_cost_usd": mean(total_costs),
            "avg_tokens": mean(total_tokens),
            "avg_retrieval_ms": mean(retrieval_latencies),
        })

    df = pd.DataFrame(summary_rows)
    csv_path = f"results/model_comparison_{ts}.csv"
    df.to_csv(csv_path, index=False)

    # ---- Plot Faithfulness vs Cost ----
    plt.figure()
    plt.scatter(df["avg_cost_usd"], df["faithfulness_rate"])
    for i, row in df.iterrows():
        plt.text(row["avg_cost_usd"], row["faithfulness_rate"], row["model"])
    plt.xlabel("Avg Cost per Query (USD)")
    plt.ylabel("Faithfulness Rate")
    plt.title("Faithfulness vs Cost")
    plt.tight_layout()
    plt.savefig(f"results/faithfulness_vs_cost_{ts}.png")

    print("\nSaved:")
    print(csv_path)
    print("faithfulness_vs_cost_*.png")

if __name__ == "__main__":
    main()
