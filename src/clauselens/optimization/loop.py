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
from typing import Callable, Dict, List, Optional

from ..config import PipelineConfig, apply_patch, save_config
from ..evaluation.evaluator import evaluate, save_result
from ..paths import CONFIG_DIR, EXPERIMENTS_DIR
from ..pipeline import Engine
from .objective import Objective, judge
from .proposer import CATALOGUE, Proposal, propose

LOG_FILE = EXPERIMENTS_DIR / "experiment_log.jsonl"
CANDIDATE_DIR = EXPERIMENTS_DIR / "configs"


def _train_data():
    from ..workspace import benchmark_split, corpus, unanswerable_split

    return corpus(), benchmark_split("train"), unanswerable_split("train")


def reset_thresholds(cfg: PipelineConfig) -> PipelineConfig:
    for g in cfg.abstention.gates:
        g.threshold = None
    return cfg


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
                 data_loader: Callable = _train_data):
        self.objective = objective
        self.catalogue = catalogue
        self.log_file = log_file
        self.say = log
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
        save_result(parent, f"loop_iter00_{self.parent_cfg.name}")
        self._write({"iteration": 0, "timestamp": now(), "kind": "baseline", "config_name": self.parent_cfg.name,
                     "config_fingerprint": parent["config"].fingerprint(), "config": parent["config"].to_dict(),
                     "objective": self.objective.to_dict(), "metrics": metrics_snapshot(parent["summary"], k),
                     "calibration": parent["calibration"], "decision": "baseline"})
        self.say(f"  recall@{k}={parent['summary']['ranking'][str(k)]['recall']:.3f} "
                 f"failures={parent['summary']['failure_counts']}")
        tried: List[str] = []
        accepted: List[str] = []
        it = 0
        while it < iterations:
            prop = propose(parent["summary"], tried, self.catalogue)
            if prop is None:
                self.say("catalogue exhausted")
                break
            p: Proposal = prop["proposal"]
            tried.append(p.pid)
            cand_cfg = reset_thresholds(apply_patch(parent["config"], p.patch, name=f"iter{it + 1:02d}_{p.pid}"))
            if cand_cfg.fingerprint(include_thresholds=False) == parent["config"].fingerprint(include_thresholds=False):
                self.say(f"[skip] {p.pid}: no-op relative to the current parent")
                continue
            it += 1
            self.say(f"[iter {it}] {p.pid} (targets {prop['category']}, {prop['count']} train failures)")
            cand = self._evaluate(cand_cfg)
            verdict = judge(parent, cand, self.budget, self.objective)
            save_result(cand, f"loop_iter{it:02d}_{p.pid}")
            save_config(cand["config"], CANDIDATE_DIR / f"iter{it:02d}_{p.pid}.yaml",
                        header=f"Loop iteration {it}: {p.pid} -> {verdict['decision']}")
            entry = {
                "iteration": it, "timestamp": now(), "kind": "candidate", "proposal_id": p.pid,
                "proposal_source": f"rule catalogue (optimization/proposer.py); most frequent untreated train failure "
                                   f"category '{prop['category']}' ({prop['count']} queries)",
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
        best = parent["config"]
        best.name = "final"
        best.description = ("Selected by the train-only improvement loop; accepted changes: "
                            + (", ".join(accepted) if accepted else "none"))
        if final_path is not None:
            save_config(best, final_path, header="Produced by `clauselens improve` (train data only). Thresholds were "
                                                 "calibrated on train; do not edit by hand.")
        self.say(f"best configuration -> {final_path}: accepted {accepted}")
        return {"accepted": accepted, "tried": tried, "best": best, "best_summary": parent["summary"]}
