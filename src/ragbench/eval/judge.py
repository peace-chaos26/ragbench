import os
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_SYSTEM = """You are a strict RAG faithfulness judge.
Decide whether the ANSWER is fully supported by the CONTEXT.
If any factual claim is not supported by CONTEXT, faithful=false.
Be conservative. If unclear, faithful=false.

Return ONLY valid JSON:
faithful (boolean),
confidence (0-1),
rationale (string, <= 25 words).
"""

def judge_faithfulness(
    question: str,
    answer: str,
    context_chunks: List[str],
    model: str = "gpt-4.1-mini",
):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = "\n\n".join([f"[chunk_{i}] {c}" for i, c in enumerate(context_chunks)])

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},  # 🔥 critical
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps({
                    "question": question,
                    "context": context,
                    "answer": answer
                })
            },
        ],
    )

    verdict = json.loads(resp.choices[0].message.content)
    usage = resp.usage.model_dump() if resp.usage else {}
    return verdict, usage

