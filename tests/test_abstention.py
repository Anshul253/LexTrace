import pytest

from lextrace.abstention import calibrate, decide
from lextrace.config import Gate


def rows(vals, name="s"):
    return [{name: v} for v in vals]


def test_single_gate_takes_largest_threshold_within_budget():
    ans = rows([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    res = calibrate(ans, rows([0.05, 0.15, 0.95]), ["s"], max_false_abstention=0.2)
    # at most 2 of 10 answerable may be refused -> threshold 0.3 (abstain on 0.1, 0.2)
    assert res["thresholds"] == [0.3]
    assert res["false_abstention"] == pytest.approx(0.2)
    assert res["correct_abstention"] == pytest.approx(2 / 3)


def test_zero_budget_never_refuses_answerable():
    ans = rows([0.5, 0.6])
    res = calibrate(ans, rows([0.1]), ["s"], 0.0)
    assert res["false_abstention"] == 0.0 and res["thresholds"][0] <= 0.5


def test_two_gates_use_both_signals():
    ans = [{"a": 0.9, "b": 0.9}] * 9 + [{"a": 0.9, "b": 0.1}]
    un = [{"a": 0.1, "b": 0.9}, {"a": 0.9, "b": 0.05}]
    res = calibrate(ans, un, ["a", "b"], 0.1)
    assert res["correct_abstention"] == 1.0 and res["false_abstention"] <= 0.1


def test_no_candidates_always_abstains():
    d = decide({"s": 99.0}, [Gate("s", 0.0)], n_candidates=0)
    assert d.abstained and d.failed_gate == "no_candidates"


def test_disabled_gate_never_abstains_with_candidates():
    assert not decide({"s": -5.0}, [Gate("s", 0.0)], 3, enabled=False).abstained


def test_route_gate_message():
    d = decide({"route_similarity": 0.1}, [Gate("route_similarity", 0.5)], 3)
    assert d.abstained and "does not closely match" in d.reason


def test_calibration_requires_answerable():
    with pytest.raises(ValueError):
        calibrate([], rows([0.1]), ["s"], 0.1)
