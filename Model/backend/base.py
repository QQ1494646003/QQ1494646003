import numpy as np
from typing import List


class BaseEmbedder:
    def __init__(self,
                 embedding_model=None,
                 word_embedding_model=None):
        self.embedding_model = embedding_model
        self.word_embedding_model = word_embedding_model

    def embed(self,
              documents: List[str],
              verbose: bool = False) -> np.ndarray:
        pass
