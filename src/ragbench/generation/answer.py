import os
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM = """You answer ONLY using the provided CONTEXT.
If the context is insufficient, say: "I don't know based on the provided context."
Cite sources as [chunk_i] where i is the index of the chunk in CONTEXT.
Keep it concise.
"""

def generate_answer(
    question: str,
    context_chunks: List[str],
    model: str = "gpt-4.1-mini",
) -> Tuple[str, Dict[str, Any]]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = "\n\n".join([f"[chunk_{i}] {c}" for i, c in enumerate(context_chunks)])

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({"question": question, "context": context})},
        ],
    )

    answer = resp.choices[0].message.content
    usage = resp.usage.model_dump() if resp.usage else {}
    return answer, usage