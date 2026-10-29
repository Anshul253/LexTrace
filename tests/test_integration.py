"""Dataset-backed checks. Run with `pytest -m integration` after `python -m lextrace prepare`."""
import pytest

from lextrace.paths import BENCHMARK_FILE

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not BENCHMARK_FILE.exists(), reason="dataset not downloaded")]


def test_mini_selection_and_split_counts():
    from lextrace.workspace import all_benchmark_queries

    qs = all_benchmark_queries()
    assert len(qs) == 194
    assert sum(q.split == "train" for q in qs) == 140 and sum(q.split == "test" for q in qs) == 54


def test_all_gold_offsets_match_and_corpus_is_referenced_docs_only():
    from lextrace.data.validate import validate_offsets
    from lextrace.workspace import all_benchmark_queries, corpus

    rep = validate_offsets(all_benchmark_queries(), corpus())
    assert rep["ok"] and rep["n_gold_spans"] == 272 and rep["n_documents"] == 29


def test_unanswerable_absence_checks_pass():
    from lextrace.data.unanswerable import absence_report, load_unanswerable
    from lextrace.workspace import corpus

    assert all(r["ok"] for r in absence_report(load_unanswerable(), corpus()))


def test_bm25_engine_on_real_corpus_maps_offsets():
    from lextrace.config import load_config
    from lextrace.pipeline import Engine
    from lextrace.workspace import benchmark_split, corpus

    cfg = load_config("baseline")
    eng = Engine(cfg, corpus())
    q = benchmark_split("train")[0]
    resp = eng.search(q.query, k=8, apply_gate=False)
    for p in resp.candidates:
        assert corpus().text(p.doc_id)[p.start:p.end] == p.text
