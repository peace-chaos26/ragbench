MODEL_PRICING = {
    # USD per 1M tokens (update if pricing changes)
    "gpt-4.1-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4o": {
        "input": 5.00,
        "output": 15.00,
    },
    "gpt-5.1": {
        "input": 10.00,
        "output": 30.00,
    },
}

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int):
    pricing = MODEL_PRICING[model]
    cost = (
        prompt_tokens / 1_000_000 * pricing["input"]
        + completion_tokens / 1_000_000 * pricing["output"]
    )
    return cost

