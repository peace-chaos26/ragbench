import json
import os
from statistics import mean
from datetime import datetime

from ragbench.bench.schema import BenchItem
from ragbench.pipeline.simple_rag import run_rag
from ragbench.generation.answer import generate_answer
from ragbench.eval.judge import judge_faithfulness
from ragbench.eval.refusal import is_refusal

COLLECTION = "demo_k8s_helm"  # uses your existing indexed demo collection
ANSWER_MODEL = "gpt-4.1-mini"
JUDGE_MODEL = "gpt-4.1-mini"

def load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    bench_path = "benchmarks/sample.jsonl"
    out_jsonl = f"results/bench_runs_{ts}.jsonl"
    out_csv = f"results/bench_summary_{ts}.csv"

    rows = []
    faithful_flags = []
    total_latency = []

    answer_success_flags = []
    false_refusal_flags = []
    hallucination_flags = []
    correct_refusal_flags = []

    with open(out_jsonl, "w") as fout:
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
                model=ANSWER_MODEL,
            )

            verdict, judge_usage = judge_faithfulness(
                question=item.question,
                answer=answer,
                context_chunks=retr["context_chunks"],
                model=JUDGE_MODEL,
            )

            total_ms = retr["timings_ms"]["total_retrieval"]  # generation/judge latency not included yet
            faithful = bool(verdict.get("faithful", False))

            is_answerable = item.gold_answer is not None
            refused = is_refusal(answer)

            if is_answerable:
                false_refusal = refused
                answer_success = (not refused) and faithful
            else:
                hallucination = not refused
                correct_refusal = refused

            faithful_flags.append(1 if faithful else 0)
            total_latency.append(total_ms)

            if is_answerable:
                answer_success_flags.append(1 if answer_success else 0)
                false_refusal_flags.append(1 if false_refusal else 0)
            else:
                hallucination_flags.append(1 if hallucination else 0)
                correct_refusal_flags.append(1 if correct_refusal else 0)

            record = {
                "id": item.id,
                "question": item.question,
                "gold_answer": item.gold_answer,
                "answer": answer,
                "retrieval": retr,
                "judge": verdict,
                "usage": {
                    "answer": ans_usage,
                    "judge": judge_usage,
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            rows.append((item.id, faithful, verdict.get("confidence", ""), verdict.get("rationale", ""), total_ms))

    answer_success_rate = mean(answer_success_flags) if answer_success_flags else 0
    false_refusal_rate = mean(false_refusal_flags) if false_refusal_flags else 0
    hallucination_rate = mean(hallucination_flags) if hallucination_flags else 0
    correct_refusal_rate = mean(correct_refusal_flags) if correct_refusal_flags else 0

    print(f"Answer Success Rate: {answer_success_rate:.3f}")
    print(f"False Refusal Rate: {false_refusal_rate:.3f}")
    print(f"Hallucination Rate: {hallucination_rate:.3f}")
    print(f"Correct Refusal Rate: {correct_refusal_rate:.3f}")

    # Write summary CSV
    with open(out_csv, "w") as f:
        f.write("id,faithful,confidence,rationale,retrieval_ms\n")
        for r in rows:
            id_, faithful, conf, rat, ms_ = r
            rat = str(rat).replace('"', '""')
            f.write(f'{id_},{int(faithful)},{conf},"{rat}",{ms_}\n')

        f.write("\n")
        f.write(f"overall_faithfulness_rate,{mean(faithful_flags):.3f}\n")
        f.write(f"avg_retrieval_ms,{mean(total_latency):.1f}\n")

    print(f"Saved runs: {out_jsonl}")
    print(f"Saved summary: {out_csv}")
    print(f"Faithfulness rate: {mean(faithful_flags):.3f}")
    print(f"Avg retrieval ms: {mean(total_latency):.1f}")

if __name__ == "__main__":
    main()

