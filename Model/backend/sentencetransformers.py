import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class SentenceTransformerBackend(BaseEmbedder):
    def __init__(self, embedding_model: Union[str, SentenceTransformer]):
        super().__init__()

        if isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            raise ValueError("Please select a correct SentenceTransformers model: \n")

    def embed(self,
              documents: List[str],
              verbose: bool = False) -> np.ndarray:
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings
