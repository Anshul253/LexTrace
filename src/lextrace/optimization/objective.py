"""The improvement loop's declared objective, guardrails and acceptance rule (fixed before any tuning ran).

PRIMARY  train recall@4, ranking view (macro mean over the train answerable queries). A reviewer reads a handful of
         passages; recall@4 measures how much of the gold clause text is among them. Missing the clause is the
         costly error in contract review, so recall leads; precision is a guardrail.
GUARDRAILS (candidate vs current accepted parent, both on train, thresholds re-calibrated for each at the same
         false-abstention budget):
  G1 precision@4  >= 0.90 x parent          (no large dilution of the evidence a reader must scan)
  G2 MRR          >= parent - 0.02          (the first relevant passage must not move down)
  G3 correct abstentions >= parent - 2      (on train unanswerable queries; at most 2 queries lost)
  G4 false-abstention rate <= budget + 0.02 (calibration targets the budget; this catches ties/discreteness)
  G5 p95 query latency <= 1500 ms           (interactive use on a CPU laptop)
ACCEPT  if  (A) retrieval gain: delta recall@4 >= +0.010 and paired-bootstrap P(gain) >= 0.90 and G1-G5 pass
        or  (B) abstention gain: |delta recall@4| < 0.005, correct abstentions >= parent + 2 and G1, G2, G4, G5 pass.
        Otherwise REJECT. Equal outcomes are rejected (keep the simpler parent)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from ..evaluation.metrics import paired_bootstrap_gain


@dataclass(frozen=True)
class Objective:
    primary_k: int = 4
    min_gain: float = 0.010
    min_p_gain: float = 0.90
    neutral_band: float = 0.005
    precision_ratio: float = 0.90
    mrr_drop: float = 0.02
    abstention_loss: int = 2
    abstention_gain: int = 2
    far_slack: float = 0.02
    p95_latency_ms: float = 1500.0

    def to_dict(self) -> Dict:
        return asdict(self)


def per_query_recall(rows: Sequence[Dict], k: int) -> List[float]:
    return [r["ranking"][str(k)]["recall"] for r in sorted(rows, key=lambda r: r["query_id"])]


def judge(parent: Dict, cand: Dict, budget: float, obj: Objective = Objective()) -> Dict:
    """parent / cand: evaluator results (train). Returns decision, guardrails and explanation."""
    k = str(obj.primary_k)
    ps, cs = parent["summary"], cand["summary"]
    r_old, r_new = ps["ranking"][k]["recall"], cs["ranking"][k]["recall"]
    delta = r_new - r_old
    p_gain = paired_bootstrap_gain(per_query_recall(cand["rows"]["answerable"], obj.primary_k),
                                   per_query_recall(parent["rows"]["answerable"], obj.primary_k))
    ca_old, ca_new = ps["abstention"]["correct_abstentions"], cs["abstention"]["correct_abstentions"]
    g = {
        "G1_precision": cs["ranking"][k]["precision"] >= obj.precision_ratio * ps["ranking"][k]["precision"],
        "G2_mrr": cs["mrr"] >= ps["mrr"] - obj.mrr_drop,
        "G3_correct_abstention": ca_new >= ca_old - obj.abstention_loss,
        "G4_false_abstention": cs["abstention"]["false_abstention_rate"] <= budget + obj.far_slack,
        "G5_latency": cs["latency_ms"]["p95"] <= obj.p95_latency_ms,
    }
    retrieval_gain = delta >= obj.min_gain and p_gain >= obj.min_p_gain and all(g.values())
    abst_gain = (abs(delta) < obj.neutral_band and ca_new >= ca_old + obj.abstention_gain
                 and all(v for name, v in g.items() if name != "G3_correct_abstention"))
    failed = [n for n, v in g.items() if not v]
    if retrieval_gain:
        decision, why = "accepted", f"recall@{k} {r_old:.3f} -> {r_new:.3f} (+{delta:.3f}, P(gain)={p_gain:.2f}); all guardrails pass"
    elif abst_gain:
        decision, why = "accepted", (f"retrieval unchanged (delta {delta:+.3f}); correct abstentions {ca_old} -> {ca_new}; "
                                     "guardrails pass")
    else:
        reasons = []
        if delta < obj.min_gain and not (abs(delta) < obj.neutral_band and ca_new >= ca_old + obj.abstention_gain):
            reasons.append(f"recall@{k} change {delta:+.3f} below +{obj.min_gain:.3f}"
                           + (f" and abstention gain {ca_new - ca_old:+d} below +{obj.abstention_gain}"
                              if abs(delta) < obj.neutral_band else ""))
        elif delta >= obj.min_gain and p_gain < obj.min_p_gain:
            reasons.append(f"gain +{delta:.3f} not reliable (P(gain)={p_gain:.2f} < {obj.min_p_gain})")
        if failed:
            reasons.append("guardrail(s) failed: " + ", ".join(failed))
        decision, why = "rejected", "; ".join(reasons) or "no improvement"
    return {"decision": decision, "explanation": why, "delta_primary": delta, "p_gain": p_gain, "guardrails": g,
            "correct_abstentions": [ca_old, ca_new]}
