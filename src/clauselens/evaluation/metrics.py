"""Evaluation metrics - pure functions over spans, independently unit-tested.

Character-level precision / recall (LegalBench-RAG, Pipitone & Alami 2024; the authors' `QAResult` code):
    G   = gold spans of the query, R_k = the top-k returned passages, both as (doc_id, start, end)
    overlap(R_k, G) = number of characters that lie in a returned passage AND in a gold span of the same document
    precision@k = overlap / total characters returned (text from wrong documents counts in the denominator)
    recall@k    = overlap / total gold characters
    nothing returned -> precision = recall = 0 (the authors' convention)
The authors' code sums pairwise overlaps, which double-counts overlapping returned passages. We take the union of
returned characters per document, which equals the authors' value whenever returned passages do not overlap (the
pipeline de-duplicates overlaps by default; `authors_precision_recall` is a direct port used in a parity test).
Aggregation: macro mean over queries.

Additional metrics
    span_hit@k  - 1 if any of the top-k passages overlaps a gold span, else 0
    doc_hit@k   - 1 if any of the top-k passages is from a gold document
    MRR         - mean over queries of 1 / rank of the first passage overlapping gold (0 if none in the list)"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

Span = Tuple[str, int, int]  # (doc_id, start, end)


def merge_intervals(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[List[int]] = []
    for s, e in sorted(spans):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def _by_doc(spans: Iterable[Span]) -> Dict[str, List[Tuple[int, int]]]:
    d: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for doc, s, e in spans:
        d[doc].append((s, e))
    return {k: merge_intervals(v) for k, v in d.items()}


def _intersection(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> int:
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def char_precision_recall(retrieved: Sequence[Span], gold: Sequence[Span]) -> Dict[str, float]:
    r, g = _by_doc(retrieved), _by_doc(gold)
    n_ret = sum(e - s for v in r.values() for s, e in v)
    n_gold = sum(e - s for v in g.values() for s, e in v)
    overlap = sum(_intersection(v, g.get(doc, [])) for doc, v in r.items())
    return {"precision": overlap / n_ret if n_ret else 0.0, "recall": overlap / n_gold if n_gold else 0.0,
            "overlap_chars": overlap, "retrieved_chars": n_ret, "gold_chars": n_gold}


def authors_precision_recall(retrieved: Sequence[Span], gold: Sequence[Span]) -> Tuple[float, float]:
    """Direct port of LegalBench-RAG's QAResult.precision / .recall (pairwise sums, no union)."""
    def common(a: Span, b: Span) -> int:
        if a[0] != b[0]:
            return 0
        lo, hi = max(a[1], b[1]), min(a[2], b[2])
        return hi - lo if hi > lo else 0

    tot_r = sum(e - s for _, s, e in retrieved)
    rel_r = sum(common(r, g) for r in retrieved for g in gold)
    tot_g = sum(e - s for _, s, e in gold)
    rel_g = sum(common(r, g) for g in gold for r in retrieved)
    return (rel_r / tot_r if tot_r else 0.0), (rel_g / tot_g if tot_g else 0.0)


def overlaps_gold(span: Span, gold: Sequence[Span]) -> bool:
    return any(span[0] == g[0] and span[1] < g[2] and g[1] < span[2] for g in gold)


def span_hit(retrieved: Sequence[Span], gold: Sequence[Span]) -> float:
    return float(any(overlaps_gold(r, gold) for r in retrieved))


def doc_hit(retrieved: Sequence[Span], gold: Sequence[Span]) -> float:
    docs = {g[0] for g in gold}
    return float(any(r[0] in docs for r in retrieved))


def first_relevant_rank(retrieved: Sequence[Span], gold: Sequence[Span]) -> int:
    """1-based rank of the first passage overlapping gold; 0 if none."""
    for i, r in enumerate(retrieved, 1):
        if overlaps_gold(r, gold):
            return i
    return 0


def reciprocal_rank(retrieved: Sequence[Span], gold: Sequence[Span]) -> float:
    r = first_relevant_rank(retrieved, gold)
    return 1.0 / r if r else 0.0


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))
    return float(s[idx])


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(xs: Sequence[float], b: int = 2000, seed: int = 0) -> Tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return (means[int(0.025 * b)], means[int(0.975 * b) - 1])


def paired_bootstrap_gain(new: Sequence[float], old: Sequence[float], b: int = 2000, seed: int = 0) -> float:
    """P(mean(new - old) > 0) under a paired bootstrap over queries."""
    diffs = [a - c for a, c in zip(new, old)]
    if not diffs:
        return float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    wins = sum(sum(diffs[rng.randrange(n)] for _ in range(n)) > 0 for _ in range(b))
    return wins / b


def abstention_metrics(answerable_abstained: Sequence[bool], unanswerable_abstained: Sequence[bool]) -> Dict:
    """Decision-level abstention rates.

    false_abstention_rate   = answerable queries returning nothing / answerable queries
    correct_abstention_rate = unanswerable queries returning nothing / unanswerable queries
    false_answer_rate       = 1 - correct_abstention_rate (unanswerable queries given passages)"""
    na, nu = len(answerable_abstained), len(unanswerable_abstained)
    fa, ca = sum(answerable_abstained), sum(unanswerable_abstained)
    return {
        "n_answerable": na, "n_unanswerable": nu,
        "false_abstentions": fa, "correct_abstentions": ca, "false_answers": nu - ca,
        "false_abstention_rate": fa / na if na else float("nan"),
        "correct_abstention_rate": ca / nu if nu else float("nan"),
        "false_answer_rate": (nu - ca) / nu if nu else float("nan"),
        "correct_abstention_ci95": wilson_interval(ca, nu),
        "false_abstention_ci95": wilson_interval(fa, na),
    }


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score of a random answerable > score of a random unanswerable); ties count 1/2."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
