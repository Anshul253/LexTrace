"""Rule-based proposer: a pre-registered catalogue of concrete pipeline changes, each tagged with the failure
category it is meant to fix. Given the current failure counts (train), the proposer picks the most frequent failure
category that still has untried proposals and returns the first untried one for it. Deterministic and auditable;
no LLM is involved."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Proposal:
    pid: str
    targets: tuple
    patch: Dict
    rationale: str


def _gates(*signals: str) -> Dict:
    return {"abstention": {"gates": [{"signal": s, "threshold": None} for s in signals]}}


CATALOGUE: List[Proposal] = [
    # --- wrong document: the passage comes from a different contract than the one asked about
    Proposal("route_hard", ("wrong_document",),
             {"routing": {"mode": "hard", "top_docs": 1, "query_part": "question"}},
             "Questions name their contract; restrict search to the best-matching contract and rank clauses with the "
             "question text only, so party names stop pulling cover pages of other contracts."),
    Proposal("route_soft", ("wrong_document",), {"routing": {"mode": "soft", "soft_weight": 0.5, "query_part": "question"}},
             "Softer alternative to hard routing: boost chunks of contracts similar to the named title."),
    Proposal("route_hard_top2", ("wrong_document",), {"routing": {"mode": "hard", "top_docs": 2}},
             "If the router's first choice is sometimes wrong, searching the top two contracts may recover it."),
    # --- wrong section: right contract, wrong clause (vocabulary mismatch between question and clause)
    Proposal("dense_minilm", ("wrong_section",), {"retrieval": {"method": "dense"}},
             "Lexical overlap fails when the clause words the concept differently; a bi-encoder matches meaning."),
    Proposal("hybrid_bm25_dense", ("wrong_section",),
             {"retrieval": {"method": "hybrid", "hybrid": {"components": ["bm25", "dense"], "fusion": "rrf",
                                                           "weights": [0.5, 0.5]}}},
             "Fuse lexical and semantic rankings (RRF) so each covers the other's misses."),
    Proposal("expansion_lexicon", ("wrong_section",), {"retrieval": {"expansion": "lexicon"}},
             "Expand concept names in questions with how contracts usually phrase them (e.g. expiration date -> term)."),
    Proposal("rerank_cross_encoder", ("wrong_section", "partial_span"), {"rerank": {"enabled": True, "top_n": 32}},
             "A cross-encoder reads question and passage together and can reorder near-misses."),
    Proposal("tfidf", ("wrong_section",), {"retrieval": {"method": "tfidf"}},
             "TF-IDF with cosine normalisation penalises long chunks differently from BM25."),
    Proposal("stem_light", ("wrong_section",), {"retrieval": {"tokenizer": {"stemming": "light"}}},
             "Conflate inflections (license/licensed/licenses) so they match."),
    Proposal("dense_bge_small", ("wrong_section",),
             {"retrieval": {"dense": {"model": "BAAI/bge-small-en-v1.5",
                                      "query_prefix": "Represent this sentence for searching relevant passages: "}}},
             "A retrieval-tuned embedding model may separate clauses better than a general sentence model."),
    Proposal("hybrid_weight_lexical", ("wrong_section",), {"retrieval": {"hybrid": {"weights": [0.7, 0.3]}}},
             "Shift fusion weight towards the lexical ranking when exact legal terms matter."),
    # --- partial span: the right clause is found but only partly covered
    Proposal("chunks_paragraph_800", ("partial_span", "wrong_section"),
             {"chunking": {"strategy": "paragraph", "size": 800}},
             "Align chunks with paragraphs so a clause is not split across fixed windows."),
    Proposal("chunks_clause_800", ("partial_span", "wrong_section"), {"chunking": {"strategy": "clause", "size": 800}},
             "Split at section headings so each chunk is one numbered clause."),
    Proposal("chunks_fixed_1000", ("partial_span",), {"chunking": {"size": 1000}},
             "Bigger chunks cover more of long clauses (at a precision cost)."),
    Proposal("expand_100", ("partial_span",), {"postprocess": {"expand_chars": 100}},
             "Widen each passage slightly to capture clause text cut at chunk edges."),
    # --- abstention: unanswerable queries answered / answerable ones refused
    Proposal("gate_bm25_norm", ("false_answer", "false_abstention"), _gates("bm25_norm"),
             "Raw scores depend on query length; normalise the top BM25 score by the question's idf mass."),
    Proposal("gate_route_and_norm", ("false_answer",), _gates("route_similarity", "bm25_norm"),
             "Two failure modes need two signals: an unknown contract (route similarity) and an absent clause "
             "(weak evidence inside the right contract)."),
    Proposal("gate_route_and_dense", ("false_answer",), _gates("route_similarity", "dense_top_cosine"),
             "Semantic evidence strength may separate absent clauses better than lexical overlap."),
    Proposal("gate_route_and_coverage", ("false_answer",), _gates("route_similarity", "query_coverage"),
             "Share of the question's distinctive terms present in the best passage."),
]


def failure_counts(summary: Dict) -> Dict[str, int]:
    c = {k: v for k, v in summary["failure_counts"].items() if k != "hit"}
    c["false_answer"] = summary["unanswerable_outcomes"]["false_answer"]
    return c


def propose(summary: Dict, tried: Sequence[str], catalogue: Sequence[Proposal] = CATALOGUE) -> Optional[Dict]:
    counts = failure_counts(summary)
    for category, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if n == 0:
            continue
        for p in catalogue:
            if category in p.targets and p.pid not in tried:
                return {"proposal": p, "category": category, "count": n}
    for p in catalogue:  # every targeted category exhausted: try what is left in catalogue order
        if p.pid not in tried:
            return {"proposal": p, "category": "untargeted", "count": 0}
    return None
