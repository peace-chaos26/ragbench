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
from ragbench.eval.refusal import is_refusal

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
        answer_success_flags = []
        false_refusal_flags = []
        hallucination_flags = []
        correct_refusal_flags = []
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

            # --- threshold guardrail ---
            tau_dense = 0.3
            tau_rerank = 0.2

            should_answer = True

            if retr["top_dense_score"] < tau_dense:
                should_answer = False

            if retr["top_rerank_score"] is not None and retr["top_rerank_score"] < tau_rerank:
                should_answer = False

            if not should_answer:
                answer = "I don't know based on the provided context."
                ans_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            else:
                answer, ans_usage = generate_answer(
                    question=item.question,
                    context_chunks=retr["context_chunks"],
                    model=model,
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

            refused = is_refusal(answer)
            is_answerable = item.gold_answer is not None

            if is_answerable:
                answer_success = (not refused) and faithful
                false_refusal = refused

                answer_success_flags.append(1 if answer_success else 0)
                false_refusal_flags.append(1 if false_refusal else 0)
            else:
                hallucination = not refused
                correct_refusal = refused

                hallucination_flags.append(1 if hallucination else 0)
                correct_refusal_flags.append(1 if correct_refusal else 0)

            # cost
            ans_prompt = ans_usage.get("prompt_tokens", 0)
            ans_completion = ans_usage.get("completion_tokens", 0)
            cost = estimate_cost(model, ans_prompt, ans_completion)

            total_costs.append(cost)
            total_tokens.append(ans_usage.get("total_tokens", 0))
            retrieval_latencies.append(retr["timings_ms"]["total_retrieval"])

        summary_rows.append({
            "model": model,
            "faithfulness_rate": mean(faithful_flags) if faithful_flags else 0,
            "answer_success_rate": mean(answer_success_flags) if answer_success_flags else 0,
            "false_refusal_rate": mean(false_refusal_flags) if false_refusal_flags else 0,
            "hallucination_rate": mean(hallucination_flags) if hallucination_flags else 0,
            "correct_refusal_rate": mean(correct_refusal_flags) if correct_refusal_flags else 0,
            "avg_cost_usd": mean(total_costs),
            "avg_tokens": mean(total_tokens),
            "avg_retrieval_ms": mean(retrieval_latencies),
        })

    df = pd.DataFrame(summary_rows)
    csv_path = f"results/model_comparison_{ts}.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== Model Comparison Summary ===")
    print(df[[
        "model",
        "faithfulness_rate",
        "answer_success_rate",
        "hallucination_rate",
        "false_refusal_rate",
        "avg_cost_usd",
        "avg_retrieval_ms"
    ]].to_string(index=False))

    # ---- Plot 1: Faithfulness vs Cost ----
    plt.figure()
    plt.scatter(df["avg_cost_usd"], df["faithfulness_rate"])
    for i, row in df.iterrows():
        plt.text(row["avg_cost_usd"], row["faithfulness_rate"], row["model"])
    plt.xlabel("Avg Cost per Query (USD)")
    plt.ylabel("Faithfulness Rate")
    plt.title("Faithfulness vs Cost")
    plt.tight_layout()
    plt.savefig(f"results/faithfulness_vs_cost_{ts}.png")
    plt.close()

    # ---- Plot 2: Hallucination vs Coverage ----
    plt.figure()
    plt.scatter(df["hallucination_rate"], df["answer_success_rate"])
    for i, row in df.iterrows():
        plt.text(row["hallucination_rate"], row["answer_success_rate"], row["model"])
    plt.xlabel("Hallucination Rate")
    plt.ylabel("Answer Success Rate")
    plt.title("Safety–Utility Frontier")
    plt.tight_layout()
    plt.savefig(f"results/safety_vs_utility_{ts}.png")

    print("\nSaved:")
    print(csv_path)
    print("faithfulness_vs_cost_*.png")
    print("safety_vs_utility_*.png")

if __name__ == "__main__":
    main()