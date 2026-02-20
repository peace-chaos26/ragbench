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
from ragbench.eval.refusal import is_refusal

COLLECTION = "demo_k8s_helm"
GEN_MODEL = "gpt-4.1-mini"
JUDGE_MODEL = "gpt-4.1-mini"

TAU_DENSE_LIST = [0.2, 0.3, 0.4]
TAU_RERANK_LIST = [0.1, 0.2, 0.3]

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_path = "benchmarks/sample.jsonl"

    rows = []

    for tau_dense in TAU_DENSE_LIST:
        for tau_rerank in TAU_RERANK_LIST:
            answer_success_flags = []
            hallucination_flags = []
            false_refusal_flags = []
            faithful_flags = []

            for raw in load_jsonl(bench_path):
                item = BenchItem(**raw)

                retr = run_rag(
                    question=item.question,
                    collection=COLLECTION,
                    dense_top_k=10,
                    use_rerank=True,
                    rerank_top_n=3,
                )

                top_dense = retr.get("top_dense_score", 0) or 0
                top_rerank = retr.get("top_rerank_score", 0) or 0

                should_answer = (top_dense >= tau_dense) and (top_rerank >= tau_rerank)

                if not should_answer:
                    answer = "I don't know based on the provided context."
                    ans_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    verdict = {"faithful": True, "confidence": 1.0, "rationale": "Refusal."}
                else:
                    answer, ans_usage = generate_answer(
                        question=item.question,
                        context_chunks=retr["context_chunks"],
                        model=GEN_MODEL,
                    )
                    verdict, _ = judge_faithfulness(
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
                    hallucination_flags.append(1 if hallucination else 0)

            rows.append({
                "tau_dense": tau_dense,
                "tau_rerank": tau_rerank,
                "faithfulness_rate": mean(faithful_flags) if faithful_flags else 0,
                "answer_success_rate": mean(answer_success_flags) if answer_success_flags else 0,
                "false_refusal_rate": mean(false_refusal_flags) if false_refusal_flags else 0,
                "hallucination_rate": mean(hallucination_flags) if hallucination_flags else 0,
            })

    df = pd.DataFrame(rows)
    csv_path = f"results/threshold_sweep_{ts}.csv"
    df.to_csv(csv_path, index=False)

    # --- Line plots by tau_dense ---
    plt.figure()
    for td in sorted(df["tau_dense"].unique()):
        sub = df[df["tau_dense"] == td].sort_values("tau_rerank")
        plt.plot(sub["tau_rerank"], sub["answer_success_rate"], marker="o", label=f"tau_dense={td}")
    plt.xlabel("tau_rerank")
    plt.ylabel("answer_success_rate")
    plt.title("Coverage vs Rerank Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/coverage_vs_tau_rerank_{ts}.png")
    plt.close()

    plt.figure()
    for td in sorted(df["tau_dense"].unique()):
        sub = df[df["tau_dense"] == td].sort_values("tau_rerank")
        plt.plot(sub["tau_rerank"], sub["hallucination_rate"], marker="o", label=f"tau_dense={td}")
    plt.xlabel("tau_rerank")
    plt.ylabel("hallucination_rate")
    plt.title("Hallucination vs Rerank Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/hallucination_vs_tau_rerank_{ts}.png")
    plt.close()


    # Plot: coverage (answer_success) vs hallucination
    plt.figure()
    plt.scatter(df["hallucination_rate"], df["answer_success_rate"])
    for i, r in df.iterrows():
        plt.text(r["hallucination_rate"], r["answer_success_rate"], f"d{r['tau_dense']}/r{r['tau_rerank']}", fontsize=8)
    plt.xlabel("Hallucination Rate")
    plt.ylabel("Answer Success Rate")
    plt.title("Coverage vs Hallucination (Threshold Sweep)")
    plt.tight_layout()
    plt.savefig(f"results/coverage_vs_hallucination_{ts}.png")
    plt.close()

    print("Saved:", csv_path)
    print("Saved: results/coverage_vs_hallucination_*.png")

if __name__ == "__main__":
    main()
