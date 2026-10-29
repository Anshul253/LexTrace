"""The search engine: chunk -> (route) -> retrieve -> (fuse) -> (rerank) -> post-process -> abstain.

`Engine.search(query)` is the ONLY inference entry point and receives nothing but the question string; gold spans
never reach this module. Both interfaces (CLI, web app) and the evaluator call it."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .abstention import decide
from .chunking import Chunk, make_chunks
from .config import PipelineConfig
from .data.corpus import Corpus
from .retrieval.retrievers import BM25Retriever, DenseRetriever, Retriever, build_retriever, minmax, rrf
from .retrieval.router import ContractRouter, ParsedQuery, parse_query
from .retrieval.text import expand_query, tokenize

MAX_QUERY_CHARS = 2000


class InvalidQueryError(ValueError):
    """The question is empty, too long or has no searchable content."""


@dataclass
class Passage:
    rank: int
    doc_id: str
    start: int
    end: int
    text: str
    score: float
    method: str
    chunk_id: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SearchResponse:
    query: str
    contract_named: Optional[str]
    question: str
    passages: List[Passage]  # what the user sees: top-k, empty when abstained
    candidates: List[Passage]  # full ranked list before the gate (evaluation "ranking view")
    abstained: bool
    abstention_reason: Optional[str]
    failed_gate: Optional[str]
    signals: Dict[str, float]
    routed_contracts: List[Tuple[str, float]]
    config_name: str
    latency_ms: float
    retrieval_text: str = ""
    method: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["passages"] = [p.to_dict() for p in self.passages]
        d["candidates"] = [p.to_dict() for p in self.candidates]
        return d


def validate_query(query: object) -> str:
    if not isinstance(query, str):
        raise InvalidQueryError("the question must be text")
    q = query.strip()
    if not q:
        raise InvalidQueryError("the question is empty")
    if len(q) > MAX_QUERY_CHARS:
        raise InvalidQueryError(f"the question is longer than {MAX_QUERY_CHARS} characters")
    if not any(ch.isalnum() for ch in q):
        raise InvalidQueryError("the question contains no words to search for")
    return q


class Engine:
    def __init__(self, cfg: PipelineConfig, corpus: Corpus):
        self.cfg = cfg
        self.corpus = corpus
        t0 = time.perf_counter()
        self.chunks: List[Chunk] = [c for d in corpus.doc_ids() for c in make_chunks(d, corpus.text(d), cfg.chunking)]
        if not self.chunks:
            raise ValueError("indexing produced no chunks")
        self.chunk_doc = np.array([c.doc_id for c in self.chunks])
        r = cfg.retrieval
        needed = set(r.hybrid.components) if r.method == "hybrid" else {r.method}
        gate_signals = {g.signal for g in cfg.abstention.gates}  # computed even when the gate is off (evaluation)
        if gate_signals & {"bm25_norm", "query_coverage"}:
            needed.add("bm25")
        if "dense_top_cosine" in gate_signals:
            needed.add("dense")
        self.retrievers: Dict[str, Retriever] = {m: build_retriever(m, self.chunks, r) for m in sorted(needed)}
        self.router: Optional[ContractRouter] = None
        if cfg.routing.mode != "none" or "route_similarity" in gate_signals:
            self.router = ContractRouter(corpus, cfg.routing.head_chars)
        self._reranker = None
        self.build_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------ helpers

    @property
    def method_label(self) -> str:
        r = self.cfg.retrieval
        label = (f"hybrid({'+'.join(r.hybrid.components)},{r.hybrid.fusion})" if r.method == "hybrid" else r.method)
        if self.cfg.routing.mode != "none":
            label = f"route:{self.cfg.routing.mode} > {label}"
        if self.cfg.rerank.enabled:
            label += " > rerank"
        return label

    def _first_stage(self, text: str, mask: np.ndarray) -> np.ndarray:
        r = self.cfg.retrieval
        if r.method != "hybrid":
            return self.retrievers[r.method].scores(text)
        lists = [self.retrievers[m].scores(text) for m in r.hybrid.components]
        if r.hybrid.fusion == "rrf":
            return rrf(lists, r.hybrid.weights, r.hybrid.rrf_k, mask)
        return sum(w * minmax(s, mask) for s, w in zip(lists, r.hybrid.weights))

    def _rerank(self, question: str, idx: List[int], first: np.ndarray) -> List[Tuple[int, float]]:
        rc = self.cfg.rerank
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(rc.model, device="cpu")
        head, tail = idx[: rc.top_n], idx[rc.top_n:]
        ce = np.asarray(self._reranker.predict([(question, self.chunks[i].text) for i in head], show_progress_bar=False),
                        dtype=float)
        if rc.weight >= 1.0:
            # pure cross-encoder ranking: keep the raw logits as scores so the rank-1 score stays an informative,
            # query-dependent confidence signal (min-max normalising would make it a constant)
            combined = ce
        else:
            fs = np.array([first[i] for i in head], dtype=float)
            norm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.ones_like(x)  # noqa: E731
            combined = rc.weight * norm(ce) + (1 - rc.weight) * norm(fs)
        reranked = [(head[j], float(combined[j])) for j in np.argsort(-combined, kind="stable")]
        # the tail keeps its first-stage order strictly below every reranked item
        floor = float(combined.min()) - 1.0
        return reranked + [(i, floor - n * 1e-6) for n, i in enumerate(tail)]

    def _postprocess(self, ranked: List[Tuple[int, float]], limit: int) -> List[Passage]:
        pp = self.cfg.postprocess
        kept: List[Passage] = []
        spans: Dict[str, List[Tuple[int, int]]] = {}
        label = self.method_label
        for i, score in ranked:
            c = self.chunks[i]
            s, e = c.start, c.end
            if pp.expand_chars:
                n = len(self.corpus.text(c.doc_id))
                s, e = max(0, s - pp.expand_chars), min(n, e + pp.expand_chars)
            if pp.dedupe_overlaps and any(s < b and a < e for a, b in spans.get(c.doc_id, [])):
                continue
            spans.setdefault(c.doc_id, []).append((s, e))
            kept.append(Passage(len(kept) + 1, c.doc_id, s, e, self.corpus.text(c.doc_id)[s:e], float(score), label,
                                c.chunk_id))
            if len(kept) >= limit:
                break
        return kept

    def _signals(self, parsed: ParsedQuery, text: str, mask: np.ndarray, cands: List[Passage],
                 route_sim: float) -> Dict[str, float]:
        sig: Dict[str, float] = {"route_similarity": route_sim, "top_score": cands[0].score if cands else 0.0}
        bm = self.retrievers.get("bm25")
        if isinstance(bm, BM25Retriever):
            mass = bm.idf_mass(text)
            s = bm.scores(text)
            top = float(s[mask].max()) if mask.any() else 0.0
            sig["bm25_norm"] = top / mass if mass > 0 else 0.0
            if cands and mass > 0:
                present = set(tokenize(cands[0].text, bm.tok))
                sig["query_coverage"] = sum(max(bm.bm25.idf[t], 0.0) for t in bm.query_terms(text) if t in present) / mass
            else:
                sig["query_coverage"] = 0.0
        dn = self.retrievers.get("dense")
        if isinstance(dn, DenseRetriever):
            s = dn.scores(parsed.question)
            sig["dense_top_cosine"] = float(s[mask].max()) if mask.any() else 0.0
        return sig

    # ------------------------------------------------------------------ entry point

    def search(self, query: str, k: Optional[int] = None, apply_gate: bool = True) -> SearchResponse:
        """apply_gate=False returns candidates and signals without deciding (the evaluator decides offline)."""
        t0 = time.perf_counter()
        q = validate_query(query)
        cfg = self.cfg
        k = k or cfg.evaluation.primary_k
        parsed = parse_query(q)
        mask = np.ones(len(self.chunks), dtype=bool)
        routed: List[Tuple[str, float]] = []
        route_sim = 1.0
        if parsed.title and self.router is not None:
            routed = self.router.route(parsed.title, max(cfg.routing.top_docs, 3))
            route_sim = routed[0][1]
            if cfg.routing.mode == "hard":
                mask = np.isin(self.chunk_doc, [d for d, _ in routed[: cfg.routing.top_docs]])
        use_question = parsed.title is not None and cfg.routing.mode != "none" and cfg.routing.query_part == "question"
        text = parsed.question if use_question else q
        if cfg.retrieval.expansion == "lexicon":
            text = expand_query(text)
        scores = self._first_stage(text, mask)
        if cfg.routing.mode == "soft" and parsed.title and self.router is not None:
            doc_sim = dict(zip(self.router.doc_ids, self.router.similarities(parsed.title)))
            scores = minmax(scores, mask) + cfg.routing.soft_weight * np.array([doc_sim[d] for d in self.chunk_doc])
        idx = np.flatnonzero(mask)
        if cfg.retrieval.method in ("bm25", "tfidf"):
            idx = idx[scores[idx] > 0]  # lexical: a chunk sharing no term with the question is not a candidate
        order = idx[np.argsort(-scores[idx], kind="stable")][: cfg.retrieval.candidates * 3]
        ranked = [(int(i), float(scores[i])) for i in order]
        if cfg.rerank.enabled and ranked:
            ranked = self._rerank(parsed.question, [i for i, _ in ranked], scores)
        cands = self._postprocess(ranked, cfg.retrieval.candidates)
        signals = self._signals(parsed, text, mask, cands, route_sim)
        dec = decide(signals, cfg.abstention.gates, len(cands), cfg.abstention.enabled and apply_gate)
        return SearchResponse(
            query=q, contract_named=parsed.title, question=parsed.question,
            passages=[] if dec.abstained else cands[:k], candidates=cands, abstained=dec.abstained,
            abstention_reason=dec.reason, failed_gate=dec.failed_gate, signals=signals, routed_contracts=routed,
            config_name=cfg.name, latency_ms=(time.perf_counter() - t0) * 1000, retrieval_text=text,
            method=self.method_label)
