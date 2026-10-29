"""Contract routing: which indexed contract does the question refer to?

Benchmark questions name their contract ("Consider the <agreement> between <A> and <B>; <question>"). The router
compares that title against a "title card" for every contract (file name + opening text, where agreements state
their name and parties) using character 3-5-gram TF-IDF cosine. Character n-grams tolerate punctuation, casing and
abbreviation differences ("Inc." / "Inc", "Co-Branding" / "Co-branding").

The best similarity is also an abstention signal: a low value means no indexed contract resembles the named one.
Inputs at inference time are the question text and the corpus only."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.corpus import Corpus

_TITLE = re.compile(r"^\s*consider\s+(?:the\s+)?(?P<title>[^;]{3,500});\s*(?P<rest>.+)$", re.I | re.S)


@dataclass(frozen=True)
class ParsedQuery:
    raw: str
    title: Optional[str]  # the contract the user named, if any
    question: str  # the question without the title clause


def parse_query(query: str) -> ParsedQuery:
    m = _TITLE.match(query)
    if not m:
        return ParsedQuery(query, None, query.strip())
    return ParsedQuery(query, m.group("title").strip(), m.group("rest").strip())


class ContractRouter:
    def __init__(self, corpus: Corpus, head_chars: int = 2000):
        self.doc_ids = corpus.doc_ids()
        cards = [Corpus.title_of(d) + "\n" + corpus.text(d)[:head_chars] for d in self.doc_ids]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(cards)

    def similarities(self, title: str) -> np.ndarray:
        q = self.vec.transform([title])
        return np.asarray((self.matrix @ q.T).toarray()).ravel()

    def route(self, title: str, top: int) -> List[Tuple[str, float]]:
        sims = self.similarities(title)
        order = np.argsort(-sims, kind="stable")[:top]
        return [(self.doc_ids[i], float(sims[i])) for i in order]
