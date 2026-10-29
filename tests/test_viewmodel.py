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


def test_agreement_type_from_cuad_file_names():
    assert vm.agreement_type("cuad/ALCOSTORESINC_12_14_2005-EX-10.26-AGENCY AGREEMENT.txt") == "Agency Agreement"
    assert vm.agreement_type("cuad/PcquoteComInc_19990721_S-1A_EX-10.11_6377149_EX-10.11_Co-Branding Agreement1.txt") \
        == "Co-Branding Agreement"
    assert vm.agreement_type("cuad/WESTERN COPPER - NON-COMPETITION AGREEMENT.txt") == "Non-Competition Agreement"


def test_compose_question_and_titles_use_train_only():
    from dataclasses import replace

    from .conftest import fixture_queries

    assert vm.compose_question("X Agreement", " Is there a cap? ") == "Consider the X Agreement; Is there a cap?"
    assert vm.compose_question(None, "Is there a cap?") == "Is there a cap?"
    qs = fixture_queries()
    titles = vm.contract_titles(qs + [replace(qs[0], query="Consider the SECRET; q", split="test")])
    assert titles["fx/alpha_beta_sponsorship.txt"].startswith("Sponsorship Agreement")
    assert "SECRET" not in titles.values()


def test_catalog_and_document_search(corpus):
    rows = vm.contract_catalog(corpus, {})
    assert len(rows) == 3 and all(r["characters"] > 0 for r in rows)
    text = corpus.text("fx/gamma_delta_distribution.txt")
    hits = vm.find_in_document(text, "governing LAW")
    assert hits and text[hits[0][0]:hits[0][1]].lower() == "governing law"
    assert vm.find_in_document(text, "   ") == []
    out = vm.document_html("a<b>c", [(1, 4)])
    assert out == 'a<mark class="cl-hit" id="cl-1">&lt;b&gt;</mark>c'


def test_missing_files_give_empty_views(tmp_path):
    assert vm.load_experiment_log(tmp_path / "none.jsonl") == []
    assert vm.test_run_count(tmp_path / "none.jsonl") is None
