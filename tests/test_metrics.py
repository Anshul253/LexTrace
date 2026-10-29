import math
import random

import pytest

from lextrace.evaluation import metrics as M

G = [("d1", 100, 200)]


def test_exact_match():
    r = M.char_precision_recall([("d1", 100, 200)], G)
    assert r["precision"] == 1.0 and r["recall"] == 1.0


def test_partial_overlap_known_values():
    r = M.char_precision_recall([("d1", 150, 250)], G)  # 50 of 100 returned chars relevant; 50 of 100 gold found
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(0.5)


def test_wrong_document_counts_against_precision():
    r = M.char_precision_recall([("d1", 100, 200), ("d2", 100, 200)], G)
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == 1.0
    r = M.char_precision_recall([("d2", 100, 200)], G)
    assert r["precision"] == 0.0 and r["recall"] == 0.0


def test_empty_retrieval_scores_zero():
    assert M.char_precision_recall([], G)["precision"] == 0.0
    assert M.char_precision_recall([], G)["recall"] == 0.0


def test_multiple_gold_spans_pooled():
    gold = [("d1", 0, 100), ("d1", 300, 400)]
    r = M.char_precision_recall([("d1", 0, 100)], gold)
    assert r["recall"] == pytest.approx(0.5) and r["precision"] == 1.0


def test_overlapping_returns_not_double_counted():
    ret = [("d1", 100, 200), ("d1", 150, 200)]
    ours = M.char_precision_recall(ret, G)
    p_auth, r_auth = M.authors_precision_recall(ret, G)
    assert ours["recall"] == 1.0 and ours["precision"] == 1.0
    assert r_auth == pytest.approx(1.5)  # the pairwise formula exceeds 1 on overlapping returns


def test_parity_with_authors_formula_on_disjoint_returns():
    rng = random.Random(0)
    for _ in range(200):
        gold = [("d1", 100, 300), ("d1", 500, 520), ("d2", 0, 50)]
        ret, pos = [], 0
        for _ in range(rng.randint(0, 6)):
            doc = rng.choice(["d1", "d2", "d3"])
            s = pos + rng.randint(0, 80)
            e = s + rng.randint(1, 120)
            ret.append((doc, s, e))
            pos = e  # disjoint, increasing
        ours = M.char_precision_recall(ret, gold)
        p, r = M.authors_precision_recall(ret, gold)
        assert ours["precision"] == pytest.approx(p) and ours["recall"] == pytest.approx(r)


def test_rank_metrics():
    ret = [("d1", 0, 50), ("d2", 100, 200), ("d1", 120, 140)]
    assert M.first_relevant_rank(ret, G) == 3
    assert M.reciprocal_rank(ret, G) == pytest.approx(1 / 3)
    assert M.span_hit(ret[:2], G) == 0.0 and M.span_hit(ret, G) == 1.0
    assert M.doc_hit(ret[:1], G) == 1.0
    assert M.reciprocal_rank([], G) == 0.0
    # touching but not overlapping is not a hit
    assert M.span_hit([("d1", 200, 250)], G) == 0.0


def test_merge_intervals():
    assert M.merge_intervals([(5, 10), (0, 3), (2, 6), (12, 12)]) == [(0, 10)]


def test_abstention_metrics_denominators():
    m = M.abstention_metrics([True, False, False, False], [True, True, False])
    assert m["false_abstention_rate"] == pytest.approx(0.25)
    assert m["correct_abstention_rate"] == pytest.approx(2 / 3)
    assert m["false_answer_rate"] == pytest.approx(1 / 3)
    assert m["n_answerable"] == 4 and m["n_unanswerable"] == 3
    empty = M.abstention_metrics([], [])
    assert math.isnan(empty["false_abstention_rate"])


def test_auroc_and_intervals():
    assert M.auroc([3, 4], [1, 2]) == 1.0
    assert M.auroc([1], [1]) == 0.5
    lo, hi = M.wilson_interval(5, 10)
    assert lo < 0.5 < hi
    assert M.paired_bootstrap_gain([1, 1, 1], [0, 0, 0]) == 1.0
    assert M.percentile([1, 2, 3, 4], 50) == 2
