from pydantic import BaseModel
from typing import List

class BenchItem(BaseModel):
    id: str
    question: str
    gold_answer: str | None = None
    must_contain: List[str] = []