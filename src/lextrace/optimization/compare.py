"""Train-only side-by-side comparison of retrieval methods with all other settings held fixed.

This complements the loop: the loop changes one thing at a time along a path, while this table answers
"BM25 vs TF-IDF vs dense vs hybrid" directly under the same chunking, routing and gate."""
from __future__ import annotations

from typing import Callable, Dict, List

from ..config import PipelineConfig, apply_patch
from ..evaluation.evaluator import evaluate
from ..pipeline import Engine
from .loop import _train_data, metrics_snapshot, reset_thresholds

VARIANTS = {
    "bm25": {"retrieval": {"method": "bm25"}, "rerank": {"enabled": False}},
    "tfidf": {"retrieval": {"method": "tfidf"}, "rerank": {"enabled": False}},
    "dense": {"retrieval": {"method": "dense"}, "rerank": {"enabled": False}},
    "hybrid_bm25_dense_rrf": {"retrieval": {"method": "hybrid", "hybrid": {"components": ["bm25", "dense"],
                                                                            "fusion": "rrf", "weights": [0.5, 0.5]}},
                              "rerank": {"enabled": False}},
    "hybrid_bm25_dense_weighted": {"retrieval": {"method": "hybrid", "hybrid": {"components": ["bm25", "dense"],
                                                                                 "fusion": "weighted",
                                                                                 "weights": [0.5, 0.5]}},
                                   "rerank": {"enabled": False}},
    "hybrid_bm25_tfidf_dense_rrf": {"retrieval": {"method": "hybrid",
                                                  "hybrid": {"components": ["bm25", "tfidf", "dense"], "fusion": "rrf",
                                                             "weights": [1.0, 1.0, 1.0]}},
                                    "rerank": {"enabled": False}},
    "hybrid_bm25_dense_rrf+rerank": {"retrieval": {"method": "hybrid", "hybrid": {"components": ["bm25", "dense"],
                                                                                   "fusion": "rrf", "weights": [0.5, 0.5]}},
                                     "rerank": {"enabled": True}},
}


def compare_methods(base: PipelineConfig, log: Callable[[str], None] = print) -> List[Dict]:
    corpus, train_q, train_u = _train_data()
    rows = []
    for name, patch in VARIANTS.items():
        cfg = reset_thresholds(apply_patch(base, patch, name=f"compare_{name}"))
        res = evaluate(cfg, Engine(cfg, corpus), train_q, train_u, "train")
        snap = metrics_snapshot(res["summary"], cfg.evaluation.primary_k)
        snap["ranking"] = res["summary"]["ranking"]
        log(f"{name:32s} R@4={snap['recall@4']:.3f} P@4={snap['precision@4']:.3f} MRR={snap['mrr']:.3f} "
            f"CA={snap['correct_abstentions']}/{snap['n_unanswerable']} p95={snap['p95_latency_ms']:.0f}ms")
        rows.append({"variant": name, "patch": patch, "metrics": snap})
    return rows
