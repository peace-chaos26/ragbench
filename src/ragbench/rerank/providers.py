from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RerankedItem:
    text: str
    payload: dict
    base_score: float | None
    rerank_score: float

class CrossEncoderReranker:
    """
    Cross-encoder reranker: scores (query, chunk) pairs.
    """
    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.model_name = model
        self.model = CrossEncoder(model)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, dict, float | None]],
        top_n: int = 3,
    ) -> List[RerankedItem]:
        """
        candidates: list of (text, payload, base_score)
        """
        pairs = [(query, c[0]) for c in candidates]
        scores = self.model.predict(pairs).tolist()

        items = [
            RerankedItem(
                text=candidates[i][0],
                payload=candidates[i][1],
                base_score=candidates[i][2],
                rerank_score=float(scores[i]),
            )
            for i in range(len(candidates))
        ]
        items.sort(key=lambda x: x.rerank_score, reverse=True)
        return items[:top_n]