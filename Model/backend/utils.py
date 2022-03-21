from .base import BaseEmbedder
from .sentencetransformers import SentenceTransformerBackend


def select_backend(embedding_model) -> BaseEmbedder:
    """ `all-MiniLM-L6-v2` for English and `paraphrase-multilingual-MiniLM-L12-v2` for all other languages

    Returns:
        model: Either a Sentence-Transformer or Flair model
    """
    

    if isinstance(embedding_model, BaseEmbedder):
        # print('hitbaseemb')
        return embedding_model

    if isinstance(embedding_model, str):
        # print('stransformer')
        return SentenceTransformerBackend(embedding_model)
    # print('hit')
    return SentenceTransformerBackend("all-MiniLM-L6-v2")
