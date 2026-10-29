import hashlib
import json
import random

import pytest

from lextrace.data.benchmark import (DatasetError, assign_split, load_tests, mini, split_queries, to_queries,
                                       validate_tests)
from lextrace.data.corpus import Corpus, read_contract
from lextrace.data.validate import validate_offsets

from .conftest import DOCS


def _tests(n, n_docs=4):
    return [{"query": f"q{i}", "snippets": [{"file_path": f"doc{i % n_docs}.txt", "span": [0, 1], "answer": "x"}]}
            for i in range(n)]


def _authors_mini(tests, n=194):  # verbatim logic of the benchmark authors' selection, for parity
    if len(tests) <= n:
        return tests

    def key(t):
        random.seed(t["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def test_mini_matches_authors_function_and_is_pinned():
    tests = _tests(10)
    assert mini(tests, 5) == _authors_mini(tests, 5)
    assert [t["query"] for t in mini(tests, 5)] == ["q3", "q7", "q1", "q5", "q9"]  # pinned output


def test_mini_returns_everything_when_small():
    tests = _tests(3)
    assert mini(tests, 194) is tests


def test_mini_groups_by_first_snippet_file_and_is_stable():
    tests = _tests(40, n_docs=5)
    sel = mini(tests, 16)
    assert len(sel) == 16
    # all queries of a document share one key, so a document's queries stay contiguous in input order
    docs = [t["snippets"][0]["file_path"] for t in sel]
    assert docs == sorted(docs, key=lambda d: docs.index(d))
    assert sel == mini(list(tests), 16)


def test_assign_split_is_the_prescribed_hash():
    for q in ["What is the governing law?", "Is there a non-compete clause?", "a", "d"]:
        h = int(hashlib.sha256(q.strip().encode("utf-8")).hexdigest(), 16)
        assert assign_split(q) == ("test" if h % 10 < 3 else "train")
    assert assign_split("d") == "test" and assign_split("a") == "train"  # pinned values
    assert assign_split("  What is the governing law?  ") == assign_split("What is the governing law?")


def test_assign_split_ratio_is_roughly_30_percent():
    n_test = sum(assign_split(f"question number {i}") == "test" for i in range(5000))
    assert 0.27 < n_test / 5000 < 0.33


def test_to_queries_ids_and_splits():
    qs = to_queries(_tests(5))
    assert [q.query_id for q in qs] == [f"cuad-mini-{i:03d}" for i in range(5)]
    assert all(q.split == assign_split(q.query) for q in qs)
    assert set(split_queries(qs, "train")) | set(split_queries(qs, "test")) == set(qs)
    with pytest.raises(ValueError):
        split_queries(qs, "dev")


@pytest.mark.parametrize("bad", [
    [],
    [{"query": "", "snippets": [{"file_path": "a", "span": [0, 1], "answer": "x"}]}],
    [{"query": "q", "snippets": []}],
    [{"query": "q", "snippets": [{"file_path": "a", "span": [5, 2], "answer": "x"}]}],
    [{"query": "q", "snippets": [{"file_path": "a", "span": [0], "answer": "x"}]}],
    [{"query": "q", "snippets": [{"span": [0, 1], "answer": "x"}]}],
    [{"query": "q", "snippets": [{"file_path": "a", "span": [0, 1]}]}],
])
def test_malformed_records_are_rejected(bad):
    with pytest.raises(DatasetError):
        validate_tests(bad)


def test_load_tests_errors(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_tests(tmp_path / "missing.json")
    p = tmp_path / "b.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="valid JSON"):
        load_tests(p)
    p.write_text(json.dumps({"other": []}), encoding="utf-8")
    with pytest.raises(DatasetError, match="tests"):
        load_tests(p)


def test_read_contract_uses_text_mode_newlines(tmp_path):
    p = tmp_path / "c.txt"
    p.write_bytes("Line one\r\nCafé two\r\n".encode("utf-8"))
    text = read_contract(p)
    assert text == "Line one\nCafé two\n"  # universal newlines, UTF-8 decoding, as the benchmark assumes


def test_offset_validation_detects_mismatch(corpus):
    doc = "fx/gamma_delta_distribution.txt"
    phrase = "royalty of ten percent"
    s = DOCS[doc].index(phrase)
    good = {"query": "q", "snippets": [{"file_path": doc, "span": [s, s + len(phrase)], "answer": phrase}]}
    bad = {"query": "r", "snippets": [{"file_path": doc, "span": [s + 1, s + len(phrase)], "answer": phrase}]}
    missing = {"query": "s", "snippets": [{"file_path": "fx/none.txt", "span": [0, 3], "answer": "abc"}]}
    rep = validate_offsets(to_queries([good, bad, missing]), corpus)
    assert rep["n_gold_spans"] == 3
    reasons = sorted(f["reason"] for f in rep["offset_failures"])
    assert reasons == ["document missing", "text mismatch"]
    assert not rep["ok"]


def test_corpus_missing_document(tmp_path):
    from lextrace.data.corpus import MissingDocumentError

    with pytest.raises(MissingDocumentError):
        Corpus.load(["cuad/nope.txt"], tmp_path)
