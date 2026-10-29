import pytest

from clauselens.pipeline import Engine, InvalidQueryError
from clauselens.retrieval.router import ContractRouter, parse_query
from clauselens.retrieval.text import light_stem, tokenize
from clauselens.config import TokenizerConfig

from .conftest import DOCS, small_config

A, B = "fx/alpha_beta_sponsorship.txt", "fx/gamma_delta_distribution.txt"


@pytest.mark.parametrize("method", ["bm25", "tfidf"])
def test_lexical_ranking_puts_the_answer_first(corpus, method):
    eng = Engine(small_config(retrieval={"method": method}), corpus)
    resp = eng.search("Can Delta sublicense the films to a third-party platform?", k=3)
    top = resp.passages[0]
    assert top.doc_id == B and "sublicense" in top.text
    assert [p.rank for p in resp.passages] == [1, 2, 3][: len(resp.passages)]
    assert all(a.score >= b.score for a, b in zip(resp.candidates, resp.candidates[1:]))


def test_every_passage_maps_to_original_offsets(corpus):
    eng = Engine(small_config(postprocess={"expand_chars": 15}), corpus)
    resp = eng.search("governing law", k=5)
    assert resp.passages
    for p in resp.candidates:
        assert corpus.text(p.doc_id)[p.start:p.end] == p.text
        assert p.method and p.chunk_id.startswith(p.doc_id)


def test_dedupe_removes_overlapping_candidates(corpus):
    eng = Engine(small_config(chunking={"strategy": "fixed", "size": 80, "overlap": 40}), corpus)
    resp = eng.search("governing law New York", k=10)
    by_doc = {}
    for p in resp.candidates:
        for s, e in by_doc.get(p.doc_id, []):
            assert p.end <= s or e <= p.start
        by_doc.setdefault(p.doc_id, []).append((p.start, p.end))


def test_routing_restricts_to_named_contract(corpus):
    cfg = small_config(routing={"mode": "hard", "top_docs": 1, "query_part": "question", "head_chars": 200})
    eng = Engine(cfg, corpus)
    resp = eng.search("Consider the Distribution Agreement between Gamma Films LLC and Delta Streaming Corp.; "
                      "What is the governing law?", k=3)
    assert resp.contract_named.startswith("Distribution Agreement")
    assert resp.routed_contracts[0][0] == B
    assert {p.doc_id for p in resp.candidates} == {B}
    assert "England" in resp.passages[0].text
    assert resp.retrieval_text == "What is the governing law?"


def test_parse_query_variants():
    p = parse_query("Consider the X Agreement between A and B; Is there a cap?")
    assert p.title == "X Agreement between A and B" and p.question == "Is there a cap?"
    p = parse_query("Is there a cap on liability?")
    assert p.title is None and p.question == "Is there a cap on liability?"


def test_router_similarity_is_low_for_unknown_contract(corpus):
    r = ContractRouter(corpus, 300)
    known = r.route("Sponsorship Agreement between Alpha Media Ltd. and Beta Drinks Inc.", 1)[0]
    unknown = r.route("Music License between Quokka Records and Wombat Games", 1)[0]
    assert known[0] == A and known[1] > unknown[1]


@pytest.mark.parametrize("bad", ["", "   ", "?!?", "x" * 2001, None, 42])
def test_invalid_queries_raise(corpus, bad):
    eng = Engine(small_config(), corpus)
    with pytest.raises(InvalidQueryError):
        eng.search(bad)


def test_abstains_when_nothing_matches(corpus):
    eng = Engine(small_config(), corpus)
    resp = eng.search("zzzz qqqq")
    assert resp.abstained and resp.passages == [] and resp.failed_gate == "no_candidates"


def test_abstains_below_threshold_and_answers_above(corpus):
    high = Engine(small_config(abstention={"gates": [{"signal": "top_score", "threshold": 1e6}]}), corpus)
    resp = high.search("What is the governing law?")
    assert resp.abstained and resp.passages == [] and resp.candidates  # candidates kept for inspection
    assert "does not mean" in resp.abstention_reason
    low = Engine(small_config(abstention={"gates": [{"signal": "top_score", "threshold": 0.0}]}), corpus)
    assert not low.search("What is the governing law?").abstained


def test_bm25_norm_and_coverage_signals(corpus):
    cfg = small_config(abstention={"gates": [{"signal": "bm25_norm", "threshold": 0.0},
                                             {"signal": "query_coverage", "threshold": 0.0}]})
    s = Engine(cfg, corpus).search("royalty net receipts").signals
    assert 0 < s["bm25_norm"] and 0 < s["query_coverage"] <= 1.0


def test_uncalibrated_gate_raises_but_evaluator_mode_does_not(corpus):
    eng = Engine(small_config(abstention={"gates": [{"signal": "top_score", "threshold": None}]}), corpus)
    with pytest.raises(ValueError):
        eng.search("governing law")
    assert eng.search("governing law", apply_gate=False).candidates


def test_tokenizer_options():
    assert tokenize("The Licensee's licenses", TokenizerConfig()) == ["the", "licensee", "licenses"]
    assert "the" not in tokenize("The licenses", TokenizerConfig(stopwords=True))
    assert light_stem("licenses") == light_stem("licensed") == light_stem("license")
