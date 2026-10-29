"""Self-improvement loop (train only).

    evaluate parent on train -> diagnose failures -> propose a change (rule catalogue) -> evaluate the candidate on
    train -> accept / reject with the declared rule (objective.py) -> log -> repeat

Leakage protection: this module only ever requests the TRAIN split (`_train_data`), the evaluator refuses mixed
splits, thresholds are calibrated on train rows only, and nothing here reads test results or the test ledger.
Every candidate's thresholds are reset and re-calibrated on train at the same false-abstention budget, so a change
in retrieval and its effect on abstention are judged together."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from ..config import PipelineConfig, apply_patch, save_config
from ..evaluation.evaluator import evaluate, save_result
from ..paths import CONFIG_DIR, EXPERIMENTS_DIR, RESULTS_DIR
from ..pipeline import Engine
from .objective import Objective, judge
from .proposer import CATALOGUE, Proposal, inapplicable, propose

LOG_FILE = EXPERIMENTS_DIR / "experiment_log.jsonl"
CANDIDATE_DIR = EXPERIMENTS_DIR / "configs"


def _train_data():
    from ..workspace import benchmark_split, corpus, unanswerable_split

    return corpus(), benchmark_split("train"), unanswerable_split("train")


def reset_thresholds(cfg: PipelineConfig) -> PipelineConfig:
    for g in cfg.abstention.gates:
        g.threshold = None
    return cfg


RANK_BASED = "rank-1 score is fusion/normalisation-scaled and nearly constant across queries"


def stabilize_gates(parent: PipelineConfig, cand: PipelineConfig) -> Optional[str]:
    """`top_score` is only a meaningful confidence for single-retriever pipelines without reranking with weight < 1:
    RRF and min-max fusion put the rank-1 score near a constant. When a candidate moves to such a pipeline, swap a
    `top_score` gate for the absolute-scale signal of a retriever it still uses (dense cosine, else normalised BM25).
    Returns a note for the log, or None when nothing changed."""
    r = cand.retrieval
    fused = r.method == "hybrid" or (cand.rerank.enabled and cand.rerank.weight < 1.0)
    if not fused:
        return None
    comps = set(r.hybrid.components) if r.method == "hybrid" else {r.method}
    repl = "dense_top_cosine" if "dense" in comps else "bm25_norm"
    changed = False
    for g in cand.abstention.gates:
        if g.signal == "top_score":
            g.signal, g.threshold, changed = repl, None, True
    return f"gate top_score -> {repl} ({RANK_BASED})" if changed else None


def blocked_only_by_abstention(verdict: Dict, obj: Objective) -> bool:
    g = verdict["guardrails"]
    return (verdict["delta_primary"] >= obj.min_gain and verdict["p_gain"] >= obj.min_p_gain
            and not g["G3_correct_abstention"] and all(v for n, v in g.items() if n != "G3_correct_abstention"))


def metrics_snapshot(summary: Dict, k: int) -> Dict:
    r = summary["ranking"]
    return {
        f"recall@{k}": r[str(k)]["recall"], f"precision@{k}": r[str(k)]["precision"],
        "recall@1": r["1"]["recall"], "recall@16": r["16"]["recall"], "recall@64": r["64"]["recall"],
        "mrr": summary["mrr"], "doc_hit@1": r["1"]["doc_hit"],
        "e2e_recall@%d" % k: summary["end_to_end"][str(k)]["recall"],
        "false_abstention_rate": summary["abstention"]["false_abstention_rate"],
        "correct_abstentions": summary["abstention"]["correct_abstentions"],
        "n_unanswerable": summary["abstention"]["n_unanswerable"],
        "p95_latency_ms": summary["latency_ms"]["p95"], "n_answerable": summary["n_answerable"],
        "failure_counts": summary["failure_counts"],
    }


class ImprovementLoop:
    def __init__(self, start: PipelineConfig, log_file: Path = LOG_FILE, objective: Objective = Objective(),
                 catalogue: List[Proposal] = CATALOGUE, log: Callable[[str], None] = print,
                 data_loader: Callable = _train_data, results_dir: Path = RESULTS_DIR,
                 candidate_dir: Path = CANDIDATE_DIR):
        self.objective = objective
        self.catalogue = catalogue
        self.log_file = log_file
        self.say = log
        self.results_dir, self.candidate_dir = results_dir, candidate_dir
        self.corpus, self.train_q, self.train_u = data_loader()
        assert all(q.split == "train" for q in self.train_q) and all(q.split == "train" for q in self.train_u)
        self.parent_cfg = reset_thresholds(start)
        self.budget = start.abstention.max_false_abstention

    def _evaluate(self, cfg: PipelineConfig) -> Dict:
        engine = Engine(cfg, self.corpus)
        res = evaluate(cfg, engine, self.train_q, self.train_u, "train")
        return res

    def _write(self, entry: Dict) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=float) + "\n")

    def run(self, iterations: int, fresh: bool = True, final_path: Optional[Path] = CONFIG_DIR / "final.yaml") -> Dict:
        if fresh and self.log_file.exists():
            self.log_file.unlink()
        k = self.objective.primary_k
        now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731
        self.say(f"[baseline] evaluating {self.parent_cfg.name} on train ({len(self.train_q)} answerable, "
                 f"{len(self.train_u)} unanswerable)")
        parent = self._evaluate(self.parent_cfg)
        save_result(parent, f"loop_iter00_{self.parent_cfg.name}", self.results_dir)
        self._write({"iteration": 0, "timestamp": now(), "kind": "baseline", "config_name": self.parent_cfg.name,
                     "config_fingerprint": parent["config"].fingerprint(), "config": parent["config"].to_dict(),
                     "objective": self.objective.to_dict(), "metrics": metrics_snapshot(parent["summary"], k),
                     "calibration": parent["calibration"], "decision": "baseline"})
        self.say(f"  recall@{k}={parent['summary']['ranking'][str(k)]['recall']:.3f} "
                 f"failures={parent['summary']['failure_counts']}")
        tried: List[str] = []
        accepted: List[str] = []
        deferred: Set[str] = set()  # rejected only by G3 despite a retrieval gain
        retried: Set[str] = set()
        it = 0
        while it < iterations:
            prop = propose(parent["summary"], tried, self.catalogue)
            if prop is None:
                self.say("catalogue exhausted")
                break
            p: Proposal = prop["proposal"]
            tried.append(p.pid)
            cand_cfg = reset_thresholds(apply_patch(parent["config"], p.patch, name=f"iter{it + 1:02d}_{p.pid}"))
            why_noop = inapplicable(parent["config"], p.patch)
            if why_noop is None and (cand_cfg.fingerprint(include_thresholds=False)
                                     == parent["config"].fingerprint(include_thresholds=False)):
                why_noop = "identical to the current parent"
            if why_noop:
                # not evaluated and not counted as an iteration, but logged so the skip is visible
                self.say(f"[skip] {p.pid}: {why_noop}")
                self._write({"iteration": None, "timestamp": now(), "kind": "skipped", "proposal_id": p.pid,
                             "change": p.patch, "decision": "skipped", "explanation": why_noop})
                continue
            it += 1
            gate_note = stabilize_gates(parent["config"], cand_cfg)
            retry = p.pid in retried
            self.say(f"[iter {it}] {p.pid}{' (retry)' if retry else ''} (targets {prop['category']}, "
                     f"{prop['count']} train failures){' | ' + gate_note if gate_note else ''}")
            cand = self._evaluate(cand_cfg)
            verdict = judge(parent, cand, self.budget, self.objective)
            save_result(cand, f"loop_iter{it:02d}_{p.pid}", self.results_dir)
            save_config(cand["config"], self.candidate_dir / f"iter{it:02d}_{p.pid}.yaml",
                        header=f"Loop iteration {it}: {p.pid} -> {verdict['decision']}")
            source = (f"rule catalogue (optimization/proposer.py); most frequent untreated train failure "
                      f"category '{prop['category']}' ({prop['count']} queries)")
            if retry:
                source = ("retry rule: rejected earlier only by the abstention guardrail (G3) with a retrieval gain; "
                          "re-proposed once after the abstention gate changed. " + source)
            entry = {
                "iteration": it, "timestamp": now(), "kind": "candidate", "proposal_id": p.pid,
                "proposal_source": source, "retry": retry, "gate_adjustment": gate_note,
                "change": p.patch, "reason": p.rationale,
                "parent_config": parent["config"].name, "parent_fingerprint": parent["config"].fingerprint(),
                "candidate_config": cand["config"].name, "candidate_fingerprint": cand["config"].fingerprint(),
                "metrics_before": metrics_snapshot(parent["summary"], k),
                "metrics_after": metrics_snapshot(cand["summary"], k),
                "calibration": cand["calibration"],
                "guardrails": verdict["guardrails"], "delta_primary": verdict["delta_primary"],
                "p_gain": verdict["p_gain"], "decision": verdict["decision"], "explanation": verdict["explanation"],
            }
            self._write(entry)
            self.say(f"  -> {verdict['decision']}: {verdict['explanation']}")
            if verdict["decision"] == "accepted":
                accepted.append(p.pid)
                parent = cand
                if "abstention" in p.patch and deferred:
                    # the gate changed: give retrieval gains that only the old gate blocked one more chance
                    for pid in deferred:
                        tried.remove(pid)
                        retried.add(pid)
                    self.say(f"  re-queued after gate change: {sorted(deferred)}")
                    deferred.clear()
            elif blocked_only_by_abstention(verdict, self.objective) and p.pid not in retried:
                deferred.add(p.pid)
        best = parent["config"]
        best.name = "final"
        best.description = ("Selected by the train-only improvement loop; accepted changes: "
                            + (", ".join(accepted) if accepted else "none"))
        if final_path is not None:
            save_config(best, final_path, header="Produced by `lextrace improve` (train data only). Thresholds were "
                                                 "calibrated on train; do not edit by hand.")
        self.say(f"best configuration -> configs/{final_path.name if final_path else None}: accepted {accepted}")
        return {"accepted": accepted, "tried": tried, "best": best, "best_summary": parent["summary"]}
