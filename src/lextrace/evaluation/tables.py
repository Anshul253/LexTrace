"""Markdown tables generated from saved results (so REPORT.md numbers are copied, not typed)."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..paths import EXPERIMENTS_DIR, TEST_LEDGER
from .evaluator import load_result


def _f(x: Optional[float], nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def retrieval_table(cols: List[tuple]) -> str:
    """cols: [(label, result_dict_or_None)]"""
    present = [(l, r) for l, r in cols if r]
    if not present:
        return "_pending: no results saved_\n"
    ks = list(present[0][1]["summary"]["ranking"])
    head = "| k | " + " | ".join(f"{l} R / P" for l, _ in present) + " |\n|---|" + "---|" * len(present) + "\n"
    body = ""
    for k in ks:
        body += f"| {k} | " + " | ".join(f"{_f(r['summary']['ranking'][k]['recall'])} / "
                                          f"{_f(r['summary']['ranking'][k]['precision'])}" for _, r in present) + " |\n"
    body += "| MRR | " + " | ".join(_f(r["summary"]["mrr"]) for _, r in present) + " |\n"
    body += "| span-hit@4 | " + " | ".join(_f(r["summary"]["ranking"]["4"]["span_hit"]) for _, r in present) + " |\n"
    body += "| e2e R@4 | " + " | ".join(_f(r["summary"]["end_to_end"]["4"]["recall"]) for _, r in present) + " |\n"
    body += "| n answerable | " + " | ".join(str(r["summary"]["n_answerable"]) for _, r in present) + " |\n"
    return head + body


def abstention_table(cols: List[tuple]) -> str:
    present = [(l, r) for l, r in cols if r]
    if not present:
        return "_pending_\n"
    head = "| | " + " | ".join(l for l, _ in present) + " |\n|---|" + "---|" * len(present) + "\n"
    a = lambda r: r["summary"]["abstention"]  # noqa: E731
    rows = [
        ("false abstentions / answerable", lambda r: f"{a(r)['false_abstentions']}/{a(r)['n_answerable']}"),
        ("correct abstentions / unanswerable", lambda r: f"{a(r)['correct_abstentions']}/{a(r)['n_unanswerable']}"),
        ("false answers / unanswerable", lambda r: f"{a(r)['false_answers']}/{a(r)['n_unanswerable']}"),
        ("correct-abstention 95% CI", lambda r: "[{:.2f}, {:.2f}]".format(*a(r)["correct_abstention_ci95"])),
        ("gates (thresholds fitted on train)", lambda r: "; ".join(
            f"{g['signal']} >= {g['threshold']:.4g}" for g in r["config"]["abstention"]["gates"])),
        ("p95 latency (ms)", lambda r: _f(r["summary"]["latency_ms"]["p95"], 1)),
    ]
    body = "".join(f"| {name} | " + " | ".join(fn(r) for _, r in present) + " |\n" for name, fn in rows)
    cats = sorted({c for _, r in present for c in r["summary"]["abstention_by_category"]})
    for c in cats:
        body += f"| correct abstention: {c} | " + " | ".join(
            "{correct}/{n}".format(**r["summary"]["abstention_by_category"].get(c, {"correct": 0, "n": 0}))
            for _, r in present) + " |\n"
    return head + body


def iteration_table(path=EXPERIMENTS_DIR / "experiment_log.jsonl") -> str:
    if not path.exists():
        return "_pending: no experiment log_\n"
    log = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = ("| # | change | R@4 | dR@4 | P(gain) | P@4 | MRR | correct abst. | FA rate | decision | explanation |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|\n")
    skipped = [e for e in log if e["kind"] == "skipped"]
    for e in log:
        if e["kind"] == "skipped":
            continue
        m = e.get("metrics_after") or e["metrics"]
        out += (f"| {e['iteration']} | {e.get('proposal_id', 'baseline')} | {_f(m['recall@4'])} | "
                f"{_f(e.get('delta_primary'), 3) if 'delta_primary' in e else ''} | "
                f"{_f(e.get('p_gain'), 2) if 'p_gain' in e else ''} | {_f(m['precision@4'])} | {_f(m['mrr'])} | "
                f"{m['correct_abstentions']}/{m['n_unanswerable']} | {_f(m['false_abstention_rate'])} | "
                f"**{e['decision']}** | {e.get('explanation', '')} |\n")
    if skipped:
        out += "\nSkipped without evaluation (no effect on the current parent): " + "; ".join(
            f"{e['proposal_id']} ({e['explanation']})" for e in skipped) + "\n"
    return out


def comparison_table(path=EXPERIMENTS_DIR / "method_comparison.json") -> str:
    if not path.exists():
        return "_pending: run `lextrace compare`_\n"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = "| method (train) | R@1 | R@4 | R@16 | P@4 | MRR | correct abst. | p95 ms |\n|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        m = r["metrics"]
        out += (f"| {r['variant']} | {_f(m['recall@1'])} | {_f(m['recall@4'])} | {_f(m['recall@16'])} | "
                f"{_f(m['precision@4'])} | {_f(m['mrr'])} | {m['correct_abstentions']}/{m['n_unanswerable']} | "
                f"{_f(m['p95_latency_ms'], 0)} |\n")
    return out


def render_all() -> str:
    b_tr, f_tr = load_result("baseline", "train"), load_result("final", "train")
    b_te, f_te = load_result("baseline", "test"), load_result("final", "test")
    cols = [("train baseline", b_tr), ("train final", f_tr), ("test baseline", b_te), ("test final", f_te)]
    n_test = (sum(1 for l in TEST_LEDGER.read_text(encoding="utf-8").splitlines() if l.strip())
              if TEST_LEDGER.exists() else 0)
    return ("## Retrieval (ranking view)\n\n" + retrieval_table(cols) + "\n## Abstention\n\n" + abstention_table(cols)
            + "\n## Method comparison (train)\n\n" + comparison_table() + "\n## Iterations\n\n" + iteration_table()
            + f"\nTest ledger entries: {n_test}\n")
