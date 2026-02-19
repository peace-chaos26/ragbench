from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class RetrievedChunk(BaseModel):
    text: str
    payload: Dict[str, Any]
    score: Optional[float] = None

class Timing(BaseModel):
    embed_query_ms: int
    retrieve_ms: int
    rerank_ms: int = 0
    total_ms: int

class RunRecord(BaseModel):
    query: str
    dense_top_k: int
    rerank_top_n: int
    embedding_model: str
    reranker_model: Optional[str] = None
    timings: Timing
    dense_results: List[RetrievedChunk]
    reranked_results: List[RetrievedChunk] = []