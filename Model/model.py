import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from .mmr import mmr
from .maxsum import max_sum_similarity
from .backend.utils import select_backend


class CoreBERT:
    def __init__(self,
                 model="all-MiniLM-L6-v2"):
        self.model = select_backend(model)

    def extract_keywords(self,
                         docs: Union[str, List[str]],
                         candidates: List[str] = None,
                         keyphrase_ngram_range: Tuple[int, int] = (1, 1),
                         stop_words: Union[str, List[str]] = 'english',
                         top_n: int = 5,
                         min_df: int = 1,
                         use_maxsum: bool = False,
                         use_mmr: bool = False,
                         diversity: float = 0.5,
                         nr_candidates: int = 20,
                         vectorizer: CountVectorizer = None,
                         seed_keywords: List[str] = None) -> Union[List[Tuple[str, float]],
                                                                    List[List[Tuple[str, float]]]]:
        if isinstance(docs, str):
            keywords = self._extract_keywords_single_doc(doc=docs,
                                                         candidates=candidates,
                                                         keyphrase_ngram_range=keyphrase_ngram_range,
                                                         stop_words=stop_words,
                                                         top_n=top_n,
                                                         use_maxsum=use_maxsum,
                                                         use_mmr=use_mmr,
                                                         diversity=diversity,
                                                         nr_candidates=nr_candidates,
                                                         vectorizer=vectorizer,
                                                         seed_keywords=seed_keywords)

            return keywords

    def _extract_keywords_single_doc(self,
                                     doc: str,
                                     candidates: List[str] = None,
                                     keyphrase_ngram_range: Tuple[int, int] = (1, 1),
                                     stop_words: Union[str, List[str]] = 'english',
                                     top_n: int = 5,
                                     use_maxsum: bool = False,
                                     use_mmr: bool = False,
                                     diversity: float = 0.5,
                                     nr_candidates: int = 20,
                                     vectorizer: CountVectorizer = None,
                                     seed_keywords: List[str] = None) -> List[Tuple[str, float]]:
        """ Extract keywords/keyphrases for a single document

        Arguments:
            doc: The document for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases
            stop_words: Stopwords to remove from the document
            top_n: Return the top n keywords/keyphrases
            use_mmr: Whether to use Max Sum Similarity
            use_mmr: Whether to use MMR
            diversity: The diversity of results between 0 and 1 if use_mmr is True
            nr_candidates: The number of candidates to consider if use_maxsum is set to True
            vectorizer: Pass in your own CountVectorizer from scikit-learn
            seed_keywords: Seed keywords that may guide the extraction of keywords by
                           steering the similarities towards the seeded keywords

        Returns:
            keywords: the top n keywords for a document with their respective distances
                      to the input document
        """
        try:
            if candidates is None:
                if vectorizer:
                    count = vectorizer.fit([doc])
                else:
                    count = CountVectorizer(ngram_range=keyphrase_ngram_range, stop_words=stop_words).fit([doc])
                candidates = count.get_feature_names()

            doc_embedding = self.model.embed([doc])
            candidate_embeddings = self.model.embed(candidates)

            if seed_keywords is not None:
                seed_embeddings = self.model.embed([" ".join(seed_keywords)])
                doc_embedding = np.average([doc_embedding, seed_embeddings], axis=0, weights=[3, 1])

            if use_mmr:
                keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n, diversity)
            elif use_maxsum:
                keywords = max_sum_similarity(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates)
            else:
                distances = cosine_similarity(doc_embedding, candidate_embeddings)
                keywords = [(candidates[index], round(float(distances[0][index]), 4))
                            for index in distances.argsort()[0][-top_n:]][::-1]

            return keywords
        except ValueError:
            return []
