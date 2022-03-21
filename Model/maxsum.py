import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


def max_sum_similarity(doc_embedding: np.ndarray,
                       word_embeddings: np.ndarray,
                       words: List[str],
                       top_n: int,
                       nr_candidates: int) -> List[Tuple[str, float]]:

    if nr_candidates < top_n:
        raise Exception("Make sure that the number of candidates exceeds the number "
                        "of keywords to return.")

    distances = cosine_similarity(doc_embedding, word_embeddings)
    distances_words = cosine_similarity(word_embeddings, word_embeddings)

    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    candidates = distances_words[np.ix_(words_idx, words_idx)]

    min_sim = 100_000
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [(words_vals[idx], round(float(distances[0][idx]), 4)) for idx in candidate]
