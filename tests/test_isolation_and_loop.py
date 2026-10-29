import json
from dataclasses import replace

import pytest

from lextrace.data.benchmark import assign_split
from lextrace.data.unanswerable import UnanswerableQuery
from lextrace.evaluation.evaluator import SplitGuardError, calibrate_config, evaluate, run_queries
from lextrace.optimization.loop import ImprovementLoop
from lextrace.optimization.objective import Objective, judge
from lextrace.optimization.proposer import Proposal, propose
from lextrace.pipeline import Engine

from .conftest import fixture_queries, small_config


def _unans(split="train"):
    texts = [f"Consider the Licence between Nobody Ltd and Noone {i} Inc.; What is the streaming window?" for i in range(40)]
    picked = [t for t in texts if assign_split(t) == split][:2]
    return [UnanswerableQuery(f"u{i}", t, "contract_not_in_corpus", None, "fixture", ("x",), "corpus")
            for i, t in enumerate(picked)]


def loop_cfg(**over):
    base = dict(evaluation={"ks": [1, 4, 16, 64], "primary_k": 4},
                abstention={"gates": [{"signal": "top_score", "threshold": None}], "max_false_abstention": 0.34})
    base.update(over)
    return small_config(**base)


def test_test_split_requires_explicit_permission(corpus):
    cfg = small_config()
    test_q = [replace(q, split="test") for q in fixture_queries()]
    with pytest.raises(SplitGuardError):
        evaluate(cfg, Engine(cfg, corpus), test_q, [], "test")
    res = evaluate(cfg, Engine(cfg, corpus), test_q, [], "test", allow_test=True)
    assert res["summary"]["n_answerable"] == 3


def test_mixed_splits_are_refused(corpus):
    cfg = small_config()
    qs = fixture_queries()
    qs[0] = replace(qs[0], split="test")
    with pytest.raises(SplitGuardError):
        evaluate(cfg, Engine(cfg, corpus), qs, [], "train")


def test_test_evaluation_never_calibrates(corpus):
    cfg = loop_cfg()
    test_q = [replace(q, split="test") for q in fixture_queries()]
    with pytest.raises(SplitGuardError, match="uncalibrated"):
        evaluate(cfg, Engine(cfg, corpus), test_q, [], "test", allow_test=True)


def test_calibration_refuses_test_rows(corpus):
    cfg = loop_cfg()
    rows = run_queries(Engine(cfg, corpus), [replace(q, split="test") for q in fixture_queries()], [], [1, 4])
    with pytest.raises(SplitGuardError):
        calibrate_config(rows, cfg)


def test_evaluator_scores_fixture_queries(corpus):
    cfg = loop_cfg(routing={"mode": "hard", "query_part": "question", "head_chars": 300})
    res = evaluate(cfg, Engine(cfg, corpus), fixture_queries(), _unans(), "train")
    s = res["summary"]
    assert s["ranking"]["1"]["recall"] == pytest.approx(1.0)
    assert res["calibration"]["thresholds"][0] is not None
    assert s["n_unanswerable"] == len(_unans())
    assert all("diagnosis" in r for r in res["rows"]["answerable"])


def _fake(recalls, ca, prec=0.1, mrr=0.5, far=0.05, p95=10.0):
    rows = [{"query_id": f"q{i}", "ranking": {"4": {"recall": r}}} for i, r in enumerate(recalls)]
    return {"summary": {"ranking": {"4": {"recall": sum(recalls) / len(recalls), "precision": prec}}, "mrr": mrr,
                        "abstention": {"correct_abstentions": ca, "false_abstention_rate": far},
                        "latency_ms": {"p95": p95}},
            "rows": {"answerable": rows}}


def test_acceptance_rule():
    parent = _fake([0.2] * 20, ca=10)
    assert judge(parent, _fake([0.4] * 20, ca=10), 0.1)["decision"] == "accepted"
    r = judge(parent, _fake([0.4] * 20, ca=5), 0.1)  # abstention guardrail G3
    assert r["decision"] == "rejected" and not r["guardrails"]["G3_correct_abstention"]
    r = judge(parent, _fake([0.4] * 20, ca=10, prec=0.05), 0.1)  # precision guardrail G1
    assert r["decision"] == "rejected" and "G1_precision" in r["explanation"]
    assert judge(parent, _fake([0.2] * 20, ca=13), 0.1)["decision"] == "accepted"  # abstention-only gain
    assert judge(parent, _fake([0.2] * 20, ca=11), 0.1)["decision"] == "rejected"  # +1 is not enough
    assert judge(parent, _fake([0.2] * 20, ca=10), 0.1)["decision"] == "rejected"  # ties are rejected
    noisy = _fake([0.2] * 19 + [0.6], ca=10)  # +0.02 driven by one query
    assert judge(parent, noisy, 0.1)["decision"] == "rejected"


def test_proposer_targets_most_frequent_failure():
    summary = {"failure_counts": {"hit": 5, "wrong_section": 9, "partial_span": 2},
               "unanswerable_outcomes": {"false_answer": 3}, "misses_top1_in_header": {"share": 0.1, "count": 1}}
    p = propose(summary, [])
    assert "wrong_section" in p["proposal"].targets and p["category"] == "wrong_section"
    summary["misses_top1_in_header"] = {"share": 0.8, "count": 9}
    assert propose(summary, [])["proposal"].pid == "route_hard"
    assert propose(summary, ["route_hard"])["proposal"].pid == "route_soft"


def test_loop_runs_logs_and_rejects(tmp_path, corpus):
    cat = [Proposal("route", ("wrong_section", "wrong_document", "partial_span", "header_dominance"),
                    {"routing": {"mode": "hard", "query_part": "question", "head_chars": 300}}, "fixture: route"),
           Proposal("noop", ("wrong_section",), {"retrieval": {"method": "bm25"}}, "fixture: no-op"),
           Proposal("tfidf", ("wrong_section", "partial_span"), {"retrieval": {"method": "tfidf"}}, "fixture: tfidf")]
    log = tmp_path / "log.jsonl"
    loop = ImprovementLoop(loop_cfg(), log_file=log, catalogue=cat, log=lambda s: None,
                           data_loader=lambda: (corpus, fixture_queries(), _unans()))
    res = loop.run(iterations=5, final_path=tmp_path / "final.yaml")
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["kind"] == "baseline"
    cands = [e for e in entries if e["kind"] == "candidate"]
    assert [e["proposal_id"] for e in cands] == ["route", "tfidf"] or [e["proposal_id"] for e in cands][0] == "route"
    for e in cands:
        for key in ("iteration", "timestamp", "change", "reason", "proposal_source", "metrics_before",
                    "metrics_after", "guardrails", "decision", "explanation"):
            assert key in e
        assert e["decision"] in ("accepted", "rejected")
    assert (tmp_path / "final.yaml").exists() and res["best"].name == "final"


def test_loop_refuses_test_data(tmp_path, corpus):
    with pytest.raises(AssertionError):
        ImprovementLoop(loop_cfg(), log_file=tmp_path / "l.jsonl", log=lambda s: None,
                        data_loader=lambda: (corpus, [replace(q, split="test") for q in fixture_queries()], []))
