def is_refusal(answer: str) -> bool:
    a = (answer or "").strip().lower()
    patterns = [
        "i don't know based on the provided context",
        "i do not know based on the provided context",
        "can't answer based on the provided context",
        "cannot answer based on the provided context",
        "insufficient context",
    ]
    return any(p in a for p in patterns)