from typing import List
from sentence_transformers import SentenceTransformer

class LocalSTEmbeddingProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()