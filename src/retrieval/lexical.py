import math
from spacy import Language
from spacy.lang.en.stop_words import STOP_WORDS
from typing import Iterable, Sequence, Counter
import heapq
import spacy

DEFAULT_STOP_WORDS: frozenset[str] = frozenset(STOP_WORDS)


nlp: Language = spacy.load("en_core_web_sm",
                        disable=[
                            "parser",
                            "ner",
                            "lemmatizer",
                            "tagger",
                            "attribute_ruler"
                        ])


def tokenize(text: str, stopword: Iterable | None = DEFAULT_STOP_WORDS) -> list[str]:
    stop_set = set(stopword) if stopword is not None else set()

    doc = nlp(text.lower())

    return [token.text
            for token in doc
            if not token.is_punct
            and not token.is_space
            and token.text
            not in stop_set]


class LexicalRetrieval:
    """
    Using BM25 for lexical retrieval
    k1: TF Saturation
    b: length normalization
    postings: term details
    doc_len: length of each document
    n_docs: number of documents in scopus
    """
    def __init__(
            self,
            b: float,
            k1: float,
            stopwords: Iterable[str] | None = DEFAULT_STOP_WORDS
    ):
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")


        self.b = b
        self.k1 = k1
        self.stopwords = stopwords

        self.postings: dict[str, dict[int, int]] = {} # saving detail {term: [document_id, frequency]}
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.n_docs = 0

    @property
    def n_docs(self) -> int:
        return self.n_docs

    @property
    def vocab_size(self) -> int:
        return len(self.postings)


    def fit(self, corpus: Sequence[str], batch_size = 512):
        if not corpus:
            raise ValueError("Must be at least one document")

        postings: dict[str, dict[int, int]] = {}
        doc_len: list[int] = []

        docs = nlp.pipe(texts=(text.lower() for text in corpus), batch_size=512)

        for doc_id, doc in enumerate(docs):
            tokens = [
                token.text for token in doc
                if not token.is_punct and not token.is_space and token.text not in self.stopwords
            ]

            doc_len.append(len(tokens))

            for term, tf in Counter(tokens).items():
                postings.setdefault(term, {})[doc_id] = tf

        self.postings = postings
        self.doc_len = doc_len
        self.n_docs = len(corpus)
        self.avgdl = sum(doc_len) / self.n_docs


    def idf(self, df: int):
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))


    def _length_norm(self, doc_len: int) -> float:
        if self.avgdl == 0.0:
            return 1.0
        return 1.0 - self.b + self.b * doc_len / self.avgdl



    def score(self, query: str) -> list[float]:

        self.validate_fitted()

        scores = [0.0] * self.n_docs

        for term in dict.fromkeys(tokenize(query, self.stopwords)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(len(posting))

            for doc_id, tf in posting.items():

                norm = self._length_norm(self.doc_len[doc_id])
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)

        return scores


    def validate_fitted(self) -> None:
        if self.n_docs == 0:
            raise RuntimeError("call fit() before scoring")


    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:

        if k <= 0:
            raise ValueError("k must be positive")

        self.validate_fitted()

        scores = self.score(query)

        best = heapq.nlargest(k, range(self.n_docs), key=lambda i: (scores[i], -i))
        return [(doc_id, scores[doc_id]) for doc_id in best]


