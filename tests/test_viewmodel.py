from lextrace.pipeline import Engine
from lextrace.ui import viewmodel as vm

from .conftest import small_config


def test_answered_view_has_exact_offsets(corpus):
    view = vm.run_search(Engine(small_config(), corpus), "royalty of ten percent", 3)
    assert view.status == "answered" and view.cards
    for c in view.cards:
        assert corpus.text(c.doc_id)[c.start:c.end] == c.text
        assert c.before == corpus.text(c.doc_id)[max(0, c.start - vm.CONTEXT_CHARS):c.start]


def test_abstained_view_keeps_weak_candidates_separate(corpus):
    eng = Engine(small_config(abstention={"gates": [{"signal": "top_score", "threshold": 1e9}]}), corpus)
    view = vm.run_search(eng, "governing law", 3)
    assert view.status == "abstained" and view.cards == [] and view.weak_cards
    assert "does not mean" in view.message


def test_invalid_and_error_views(corpus):
    eng = Engine(small_config(), corpus)
    assert vm.run_search(eng, "  ", 3).status == "invalid"

    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("boom")

    assert vm.run_search(Broken(), "x", 3).status == "error"


def test_highlight_escapes_html(corpus):
    card = vm.EvidenceCard(1, "d", "d", 0, 5, 1.0, "bm25", "<b>x</b>", "<script>", "&")
    out = vm.highlight_html(card)
    assert "<script>" not in out and "&lt;b&gt;" in out and '<mark class="cl-hit">' in out


def test_missing_files_give_empty_views(tmp_path):
    assert vm.load_experiment_log(tmp_path / "none.jsonl") == []
    assert vm.test_run_count(tmp_path / "none.jsonl") is None
