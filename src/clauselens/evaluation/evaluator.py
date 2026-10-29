"""Run a configuration over a split and score it.

Every query goes through `Engine.search(query_string)`. Gold spans are read only AFTER the response exists, to score
it. The abstention decision is re-applied from the returned signals with `abstention.decide` (the same function the
engine uses), which lets train evaluation calibrate thresholds first and then apply them.

Two views of retrieval quality on answerable queries:
  ranking view    - the ranked candidates before the abstention gate (measures retrieval itself)
  end-to-end view - what a user sees: an abstained query returns nothing and scores 0"""
from __future__ import annotations

import copy
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..abstention import calibrate, decide
from ..config import PipelineConfig
from ..data.benchmark import BenchmarkQuery
from ..data.unanswerable import UnanswerableQuery
from ..paths import RESULTS_DIR, TEST_LEDGER
from ..pipeline import Engine
from . import metrics as M


class SplitGuardError(RuntimeError):
    """Raised when held-out data is about to be used outside a declared final evaluation."""


def _spans(passages) -> List[M.Span]:
    return [(p.doc_id, p.start, p.end) for p in passages]


def diagnose(row: Dict, k: int) -> str:
    """Failure category of an answerable query at cutoff k (end-to-end view)."""
    if row["abstained"]:
        return "false_abstention"
    m = row["ranking"][str(k)]
    if m["recall"] >= 0.9:
        return "hit"
    if m["recall"] > 0:
        return "partial_span"
    if m["doc_hit"]:
        return "wrong_section"
    return "wrong_document"


def run_queries(engine: Engine, answerable: Sequence[BenchmarkQuery], unanswerable: Sequence[UnanswerableQuery],
                ks: Sequence[int]) -> Dict[str, List[Dict]]:
    kmax = max(ks)
    a_rows, u_rows = [], []
    for q in answerable:
        resp = engine.search(q.query, k=kmax, apply_gate=False)
        gold = [(g.doc_id, g.start, g.end) for g in q.gold]  # scoring only, after the search
        cands = _spans(resp.candidates)
        row = {"query_id": q.query_id, "query": q.query, "split": q.split, "latency_ms": resp.latency_ms,
               "signals": resp.signals, "n_candidates": len(resp.candidates),
               "gold": [{"doc_id": d, "start": s, "end": e} for d, s, e in gold],
               "top": [p.to_dict() for p in resp.candidates[:kmax]],
               "first_relevant_rank": M.first_relevant_rank(cands, gold),
               "routed": resp.routed_contracts[:3], "ranking": {}}
        for k in ks:
            pr = M.char_precision_recall(cands[:k], gold)
            row["ranking"][str(k)] = {"precision": pr["precision"], "recall": pr["recall"],
                                      "span_hit": M.span_hit(cands[:k], gold), "doc_hit": M.doc_hit(cands[:k], gold)}
        row["rr"] = M.reciprocal_rank(cands, gold)
        a_rows.append(row)
    for q in unanswerable:
        resp = engine.search(q.query, k=kmax, apply_gate=False)
        u_rows.append({"query_id": q.query_id, "query": q.query, "split": q.split, "category": q.category,
                       "latency_ms": resp.latency_ms, "signals": resp.signals, "n_candidates": len(resp.candidates),
                       "top": [p.to_dict() for p in resp.candidates[:3]], "routed": resp.routed_contracts[:3]})
    return {"answerable": a_rows, "unanswerable": u_rows}


def apply_gates(rows: Dict[str, List[Dict]], cfg: PipelineConfig) -> None:
    for r in rows["answerable"] + rows["unanswerable"]:
        d = decide(r["signals"], cfg.abstention.gates, r["n_candidates"], cfg.abstention.enabled)
        r["abstained"], r["abstention_reason"], r["failed_gate"] = d.abstained, d.reason, d.failed_gate


def calibrate_config(rows: Dict[str, List[Dict]], cfg: PipelineConfig) -> Dict:
    """Fit thresholds for gates whose threshold is None. MUST only be called with train rows."""
    if any(r["split"] != "train" for r in rows["answerable"] + rows["unanswerable"]):
        raise SplitGuardError("thresholds may only be calibrated on train queries")
    names = [g.signal for g in cfg.abstention.gates]
    sig = lambda r: dict(r["signals"], _n_candidates=r["n_candidates"])  # noqa: E731
    res = calibrate([sig(r) for r in rows["answerable"]], [sig(r) for r in rows["unanswerable"]], names,
                    cfg.abstention.max_false_abstention)
    for g, t in zip(cfg.abstention.gates, res["thresholds"]):
        g.threshold = float(t)
    return res


def summarize(rows: Dict[str, List[Dict]], cfg: PipelineConfig) -> Dict:
    a, u = rows["answerable"], rows["unanswerable"]
    ks = cfg.evaluation.ks
    out: Dict = {"n_answerable": len(a), "n_unanswerable": len(u), "ranking": {}, "end_to_end": {}}
    for k in ks:
        key = str(k)
        out["ranking"][key] = {m: M.mean([r["ranking"][key][m] for r in a]) for m in ("precision", "recall", "span_hit",
                                                                                     "doc_hit")}
        out["end_to_end"][key] = {m: M.mean([0.0 if r["abstained"] else r["ranking"][key][m] for r in a])
                                  for m in ("precision", "recall")}
    pk = str(cfg.evaluation.primary_k)
    out["primary_k"] = cfg.evaluation.primary_k
    out["recall_ci95"] = M.bootstrap_ci([r["ranking"][pk]["recall"] for r in a])
    out["mrr"] = M.mean([r["rr"] for r in a])
    out["abstention"] = M.abstention_metrics([r["abstained"] for r in a], [r["abstained"] for r in u])
    out["abstention_by_category"] = {
        c: {"n": sum(1 for r in u if r["category"] == c), "correct": sum(1 for r in u if r["category"] == c and r["abstained"])}
        for c in sorted({r["category"] for r in u})}
    gate_signals = [g.signal for g in cfg.abstention.gates]
    out["signal_auroc"] = {s: M.auroc([r["signals"].get(s, 0.0) for r in a], [r["signals"].get(s, 0.0) for r in u])
                           for s in sorted(set(gate_signals) | {"top_score"})}
    lat = [r["latency_ms"] for r in a + u]
    out["latency_ms"] = {"mean": M.mean(lat), "p50": M.percentile(lat, 50), "p95": M.percentile(lat, 95)}
    out["failure_counts"] = dict(Counter(diagnose(r, cfg.evaluation.primary_k) for r in a))
    out["unanswerable_outcomes"] = {"correct_abstention": sum(r["abstained"] for r in u),
                                    "false_answer": sum(not r["abstained"] for r in u)}
    return out


def evaluate(cfg: PipelineConfig, engine: Engine, answerable: Sequence[BenchmarkQuery],
             unanswerable: Sequence[UnanswerableQuery], split: str, allow_test: bool = False,
             calibrate_missing: bool = True) -> Dict:
    """Evaluate one split. Returns {"config", "summary", "rows", "calibration"}; cfg thresholds may be filled in."""
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    if any(q.split != split for q in answerable) or any(q.split != split for q in unanswerable):
        raise SplitGuardError(f"queries from another split were passed to a {split} evaluation")
    if split == "test":
        if not allow_test:
            raise SplitGuardError("test evaluation is reserved for final reporting; pass allow_test=True explicitly")
        if cfg.needs_calibration():
            raise SplitGuardError("this configuration has uncalibrated thresholds; calibrate on train first")
    cfg = copy.deepcopy(cfg)
    t0 = time.perf_counter()
    rows = run_queries(engine, answerable, unanswerable, cfg.evaluation.ks)
    calibration = None
    if cfg.needs_calibration():
        if not calibrate_missing:
            raise SplitGuardError("uncalibrated thresholds")
        calibration = calibrate_config(rows, cfg)
    apply_gates(rows, cfg)
    for r in rows["answerable"]:
        r["diagnosis"] = diagnose(r, cfg.evaluation.primary_k)
    for r in rows["unanswerable"]:
        r["diagnosis"] = "correct_abstention" if r["abstained"] else "false_answer"
    summary = summarize(rows, cfg)
    summary["eval_seconds"] = time.perf_counter() - t0
    summary["index_build_seconds"] = engine.build_seconds
    summary["n_chunks"] = len(engine.chunks)
    return {"config": cfg, "summary": summary, "rows": rows, "calibration": calibration, "split": split}


def save_result(result: Dict, tag: str, out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg: PipelineConfig = result["config"]
    path = out_dir / f"{tag}_{result['split']}.json"
    payload = {"config_name": cfg.name, "config_fingerprint": cfg.fingerprint(), "config": cfg.to_dict(),
               "split": result["split"], "calibration": result["calibration"], "summary": result["summary"],
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    with open(out_dir / f"{tag}_{result['split']}_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in result["rows"]["answerable"] + result["rows"]["unanswerable"]:
            f.write(json.dumps(r, default=float) + "\n")
    return path


def record_test_run(cfg: PipelineConfig, summary: Dict, note: str, ledger: Path = TEST_LEDGER) -> int:
    """Append to the test ledger; returns how many test evaluations the ledger now holds."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config_name": cfg.name,
             "config_fingerprint": cfg.fingerprint(), "note": note,
             "recall_at_primary_k": summary["ranking"][str(summary["primary_k"])]["recall"]}
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return sum(1 for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip())


def load_result(tag: str, split: str, out_dir: Path = RESULTS_DIR) -> Optional[Dict]:
    p = out_dir / f"{tag}_{split}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
