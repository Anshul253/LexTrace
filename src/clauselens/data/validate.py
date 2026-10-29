"""Dataset validation: every gold snippet must satisfy `document[start:end] == answer`."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from .benchmark import BenchmarkQuery
from .corpus import Corpus


def validate_offsets(queries: Sequence[BenchmarkQuery], corpus: Corpus) -> Dict:
    failures: List[Dict] = []
    overlapping: List[str] = []
    n_spans = 0
    for q in queries:
        spans_by_doc: Dict[str, list] = {}
        for g in q.gold:
            n_spans += 1
            if g.doc_id not in corpus:
                failures.append({"query_id": q.query_id, "doc_id": g.doc_id, "reason": "document missing"})
                continue
            text = corpus.text(g.doc_id)
            if g.end > len(text):
                failures.append({"query_id": q.query_id, "doc_id": g.doc_id, "reason": "span beyond document end"})
            elif text[g.start:g.end] != g.answer:
                failures.append({"query_id": q.query_id, "doc_id": g.doc_id, "reason": "text mismatch",
                                 "expected": g.answer[:80], "found": text[g.start:g.end][:80]})
            spans_by_doc.setdefault(g.doc_id, []).append((g.start, g.end))
        for spans in spans_by_doc.values():
            spans.sort()
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                overlapping.append(q.query_id)
    splits = Counter(q.split for q in queries)
    dup = [k for k, v in Counter(q.query.strip() for q in queries).items() if v > 1]
    return {
        "n_queries": len(queries),
        "n_gold_spans": n_spans,
        "n_documents": len(corpus),
        "corpus_chars": corpus.total_chars(),
        "split_counts": dict(sorted(splits.items())),
        "offset_failures": failures,
        "queries_with_overlapping_gold": overlapping,
        "duplicate_query_strings": len(dup),
        "ok": not failures,
    }
