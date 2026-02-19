from __future__ import annotations
from typing import List, Protocol
import os

from dotenv import load_dotenv

load_dotenv()

class EmbeddingProvider(Protocol):
    name: str
    dim: int
    def embed_texts(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


class OpenAIEmbeddingProvider:
    """
    Uses OpenAI embeddings. Good baseline: strong quality, stable.
    """
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.name = f"openai:{model}"
        # dims: 1536 for text-embedding-3-small, 3072 for -3-large
        self.dim = 1536 if model.endswith("3-small") else 3072

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class SentenceTransformerProvider:
    """
    Open-source embeddings. Great for reproducibility + cost-free runs.
    """
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.model = SentenceTransformer(model)
        self.name = f"st:{model}"
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True).tolist()
        return vecs

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]