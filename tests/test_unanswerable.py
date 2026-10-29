import json
import shutil

import pytest

from clauselens.data.benchmark import DatasetError, assign_split
from clauselens.data.unanswerable import CATEGORIES, file_sha256, load_unanswerable, split_counts
from clauselens.paths import UNANSWERABLE_FILE, UNANSWERABLE_LOCK


def test_frozen_set_matches_lock_and_has_at_least_30():
    qs = load_unanswerable()
    assert len(qs) >= 30
    assert UNANSWERABLE_LOCK.read_text(encoding="utf-8").split()[0] == file_sha256()
    assert len({q.query_id for q in qs}) == len(qs) and len({q.query for q in qs}) == len(qs)
    assert {q.category for q in qs} <= set(CATEGORIES)
    assert all(q.rationale and q.patterns for q in qs)


def test_split_uses_the_prescribed_function():
    qs = load_unanswerable()
    c = split_counts(qs)
    assert c["train"] + c["test"] == c["total"] == len(qs)
    assert c == {"train": sum(assign_split(q.query) == "train" for q in qs),
                 "test": sum(assign_split(q.query) == "test" for q in qs), "total": len(qs)}
    assert c["test"] >= 5


def test_edits_after_freeze_are_detected(tmp_path):
    p, lock = tmp_path / "u.json", tmp_path / "u.sha256"
    shutil.copy(UNANSWERABLE_FILE, p)
    shutil.copy(UNANSWERABLE_LOCK, lock)
    load_unanswerable(p, lock)  # unchanged copy loads
    data = json.loads(p.read_text(encoding="utf-8"))
    data["queries"][0]["query"] += " (edited)"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetError, match="changed after it was frozen"):
        load_unanswerable(p, lock)


def test_schema_errors(tmp_path):
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"queries": [{"query_id": "x", "query": "q", "label": "answerable", "category": "out_of_scope",
                                          "rationale": "r", "absence_check": {}}]}), encoding="utf-8")
    with pytest.raises(DatasetError, match="label"):
        load_unanswerable(p, require_lock=False)
