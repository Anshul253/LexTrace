"""First-stage retrievers. Each scores ALL chunks of the index for a query string and returns a numpy vector,
so routing, fusion and abstention can be implemented once, independent of the method.

  BM25Retriever   - Okapi BM25 (rank-bm25). Also reports `idf_mass`: the summed idf of the query's indexed terms,
                    used to normalise the top score into a query-length-independent confidence.
  TfidfRetriever  - scikit-learn TF-IDF (same tokeniser), cosine similarity.
  DenseRetriever  - sentence-transformers bi-encoder, cosine similarity; chunk embeddings are cached on disk
                    keyed by model name + chunk content hash, so re-runs do not recompute them."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..chunking import Chunk
from ..config import DenseConfig, RetrievalConfig, TokenizerConfig
from ..paths import EMBED_CACHE_DIR
from .text import tokenize


class Retriever:
    name = "base"

    def scores(self, query: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], tok: TokenizerConfig, k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi

        self.tok = tok
        corpus = [tokenize(c.text, tok) or ["∅"] for c in chunks]
        self.bm25 = BM25Okapi(corpus, k1=k1, b=b)

    def query_terms(self, query: str) -> List[str]:
        return [t for t in dict.fromkeys(tokenize(query, self.tok)) if t in self.bm25.idf]

    def scores(self, query: str) -> np.ndarray:
        terms = self.query_terms(query)
        if not terms:
            return np.zeros(self.bm25.corpus_size)
        return np.asarray(self.bm25.get_scores(terms), dtype=float)

    def idf_mass(self, query: str) -> float:
        return float(sum(max(self.bm25.idf[t], 0.0) for t in self.query_terms(query)))


class TfidfRetriever(Retriever):
    name = "tfidf"

    def __init__(self, chunks: Sequence[Chunk], tok: TokenizerConfig, ngram_max: int = 1, sublinear_tf: bool = True):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tok = tok
        self.vec = TfidfVectorizer(analyzer=lambda s: self._analyze(s, ngram_max), sublinear_tf=sublinear_tf, norm="l2")
        self.matrix = self.vec.fit_transform([c.text for c in chunks])

    def _analyze(self, text: str, ngram_max: int) -> List[str]:
        toks = tokenize(text, self.tok)
        grams = list(toks)
        for n in range(2, ngram_max + 1):
            grams.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
        return grams

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        return np.asarray((self.matrix @ q.T).toarray()).ravel()


_MODELS: Dict[str, object] = {}


def load_sentence_model(name: str):
    """Load (once per process) a sentence-transformers model. Raises a RuntimeError with recovery steps."""
    if name not in _MODELS:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("dense retrieval needs sentence-transformers: pip install -r requirements.txt") from e
        try:
            _MODELS[name] = SentenceTransformer(name, device="cpu")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"could not load embedding model {name!r}: {e}. It is downloaded from huggingface.co on "
                               "first use; check network access or pre-download it, or use a non-dense config.") from e
    return _MODELS[name]


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], cfg: DenseConfig, cache_dir: Optional[Path] = EMBED_CACHE_DIR):
        self.cfg = cfg
        self.model = load_sentence_model(cfg.model)
        self.emb = self._chunk_embeddings(chunks, cache_dir)

    def _chunk_embeddings(self, chunks: Sequence[Chunk], cache_dir: Optional[Path]) -> np.ndarray:
        h = hashlib.sha256(self.cfg.model.encode())
        for c in chunks:
            h.update(f"{c.doc_id}:{c.start}:{c.end}\n".encode())
        path = cache_dir / f"{self.cfg.model.replace('/', '__')}_{h.hexdigest()[:16]}.npy" if cache_dir else None
        if path is not None and path.exists():
            emb = np.load(path)
            if emb.shape[0] == len(chunks):
                return emb
        emb = self.model.encode([c.text for c in chunks], batch_size=self.cfg.batch_size, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([self.cfg.query_prefix + query], normalize_embeddings=True, show_progress_bar=False,
                              convert_to_numpy=True)[0]
        return self.emb @ q


def build_retriever(method: str, chunks: Sequence[Chunk], cfg: RetrievalConfig) -> Retriever:
    if method == "bm25":
        return BM25Retriever(chunks, cfg.tokenizer, cfg.bm25_k1, cfg.bm25_b)
    if method == "tfidf":
        return TfidfRetriever(chunks, cfg.tokenizer, cfg.tfidf_ngram_max, cfg.tfidf_sublinear_tf)
    if method == "dense":
        return DenseRetriever(chunks, cfg.dense)
    raise ValueError(f"unknown retrieval method {method!r}")


# ------------------------------------------------------------------ fusion


def minmax(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Min-max normalise over the masked entries (others set to 0)."""
    out = np.zeros_like(x, dtype=float)
    idx = np.flatnonzero(mask) if mask is not None else np.arange(len(x))
    if len(idx) == 0:
        return out
    v = x[idx]
    lo, hi = float(v.min()), float(v.max())
    out[idx] = (v - lo) / (hi - lo) if hi > lo else (v > 0).astype(float)
    return out


def rrf(score_lists: Sequence[np.ndarray], weights: Sequence[float], k: int, mask: np.ndarray) -> np.ndarray:
    """Weighted reciprocal-rank fusion over the masked chunks: sum_i w_i / (k + rank_i)."""
    fused = np.zeros(len(mask), dtype=float)
    idx = np.flatnonzero(mask)
    for s, w in zip(score_lists, weights):
        order = idx[np.argsort(-s[idx], kind="stable")]
        ranks = np.empty(len(order))
        ranks[:] = np.arange(1, len(order) + 1)
        fused[order] += w / (k + ranks)
    return fused
