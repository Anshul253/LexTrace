"""Abstention: decide whether any retrieved passage is relevant enough to show.

The pipeline computes a dict of confidence SIGNALS per query; the configured GATES each compare one signal with a
threshold, and the system abstains if ANY gate fails (or nothing was retrieved). An abstention means "no passage met
the configured evidence threshold" - never "the contract has no such clause".

Calibration (train only): thresholds are chosen so that at most `max_false_abstention` of the TRAIN answerable
queries are refused; among threshold combinations meeting that budget, the one abstaining on the most TRAIN
unanswerable queries wins (ties -> the combination using more of the budget -> higher thresholds). With a single
gate this reduces to the largest threshold within budget, i.e. it does not depend on the unanswerable labels.
Rationale for the tie-break: there are few train unanswerable queries, so ties are common; the budget is the
declared acceptable cost, and a higher threshold is more cautious on unseen unanswerable questions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

SIGNALS = {
    "top_score": "score of the rank-1 passage in the final ranking (scale depends on the method)",
    "bm25_norm": "rank-1 BM25 score within the searched contracts / summed idf of the question's terms",
    "query_coverage": "share of the question's idf mass whose terms occur in the rank-1 passage",
    "dense_top_cosine": "highest bi-encoder cosine between the question and any searched passage",
    "route_similarity": "similarity between the contract named in the question and the best indexed contract "
                        "(1.0 when the question names no contract)",
}


@dataclass
class Decision:
    abstained: bool
    reason: Optional[str]
    failed_gate: Optional[str] = None


def decide(signals: Dict[str, float], gates, n_candidates: int, enabled: bool = True) -> Decision:
    if n_candidates == 0:
        return Decision(True, "No indexed passage matched the question at all.", "no_candidates")
    if not enabled:
        return Decision(False, None)
    for g in gates:
        if g.threshold is None:
            raise ValueError(f"gate {g.signal!r} has no threshold; calibrate on train first (`clauselens evaluate --split train`)")
        value = signals.get(g.signal, 0.0)
        if value < g.threshold:
            if g.signal == "route_similarity":
                msg = (f"The contract named in the question does not closely match any indexed contract "
                       f"(similarity {value:.3f} < {g.threshold:.3f}).")
            else:
                msg = f"The best evidence scored {value:.3f} on '{g.signal}', below the threshold {g.threshold:.3f}."
            return Decision(True, msg + " This does not mean the contract lacks such a clause.", g.signal)
    return Decision(False, None)


def _abstains(row: Dict[str, float], ths: Sequence[float], names: Sequence[str]) -> bool:
    if row.get("_n_candidates", 1) == 0:
        return True
    return any(row.get(n, 0.0) < t for n, t in zip(names, ths))


def calibrate(answerable: List[Dict[str, float]], unanswerable: List[Dict[str, float]], signal_names: Sequence[str],
              max_false_abstention: float) -> Dict[str, object]:
    """Return {"thresholds": [...], "false_abstention": r, "correct_abstention": r} chosen on the given (train) rows."""
    if not answerable:
        raise ValueError("calibration needs answerable queries")
    n_a, n_u = len(answerable), len(unanswerable)
    budget = math.floor(max_false_abstention * n_a + 1e-9)
    names = list(signal_names)

    def candidates(name: str) -> List[float]:
        vals = sorted({float(r.get(name, 0.0)) for r in answerable})
        return [-math.inf] + vals  # threshold v abstains on rows with signal < v

    def score(ths):
        fa = sum(_abstains(r, ths, names) for r in answerable)
        ca = sum(_abstains(r, ths, names) for r in unanswerable)
        return fa, ca

    best = None
    if len(names) == 1:
        grid = [[t] for t in candidates(names[0])]
    elif len(names) == 2:
        grid = [[a, b] for a in candidates(names[0]) for b in candidates(names[1])]
    else:
        raise ValueError("calibration supports at most two gates")
    for ths in grid:
        fa, ca = score(ths)
        if fa > budget:
            continue
        key = (ca, fa, ths)  # most correct abstentions; ties -> spend the declared budget (more cautious on unseen)
        if best is None or key > best[0]:
            best = (key, ths, fa, ca)
    assert best is not None  # -inf thresholds always satisfy the budget
    _, ths, fa, ca = best
    ths = [t if math.isfinite(t) else -1e9 for t in ths]  # "never abstain on this gate", YAML-serialisable
    return {"thresholds": ths, "false_abstention": fa / n_a, "correct_abstention": (ca / n_u) if n_u else float("nan"),
            "n_answerable": n_a, "n_unanswerable": n_u}
