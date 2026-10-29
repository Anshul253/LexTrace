"""Error-analysis records: for every failing train query, the evidence needed to diagnose it by hand."""
from __future__ import annotations

from collections import Counter
from typing import Dict

from ..data.corpus import Corpus


def _clip(text: str, n: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " ..."


def error_records(result: Dict, corpus: Corpus, k: int = None) -> Dict:
    cfg = result["config"]
    k = k or cfg.evaluation.primary_k
    failures = []
    for r in result["rows"]["answerable"]:
        if r["diagnosis"] == "hit":
            continue
        gold = r["gold"]
        failures.append({
            "query_id": r["query_id"], "query": r["query"], "diagnosis": r["diagnosis"],
            "recall_at_k": r["ranking"][str(k)]["recall"], "first_relevant_rank": r["first_relevant_rank"],
            "abstained": r["abstained"], "abstention_reason": r.get("abstention_reason"),
            "signals": r["signals"], "routed": r["routed"],
            "gold": [{**g, "text": _clip(corpus.text(g["doc_id"])[g["start"]:g["end"]])} for g in gold],
            "top_passages": [{"rank": p["rank"], "doc_id": p["doc_id"], "start": p["start"], "end": p["end"],
                              "score": p["score"], "text": _clip(p["text"], 200)} for p in r["top"][:k]],
        })
    for r in result["rows"]["unanswerable"]:
        if r["abstained"]:
            continue
        failures.append({
            "query_id": r["query_id"], "query": r["query"], "diagnosis": "false_answer", "category": r["category"],
            "signals": r["signals"], "routed": r["routed"],
            "top_passages": [{"rank": p["rank"], "doc_id": p["doc_id"], "start": p["start"], "end": p["end"],
                              "score": p["score"], "text": _clip(p["text"], 200)} for p in r["top"][:3]],
        })
    return {"config": cfg.name, "split": result["split"], "k": k,
            "counts": dict(Counter(f["diagnosis"] for f in failures)), "failures": failures}
