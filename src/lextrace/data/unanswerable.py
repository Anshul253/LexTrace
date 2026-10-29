"""The frozen set of unanswerable queries.

The set lives in `data/unanswerable_queries.json`; its SHA-256 is locked in `data/unanswerable_queries.sha256`.
Every load verifies the lock, so any edit after the freeze is detected (and the Git history shows when it was
committed). Splits use the same `assign_split` as the benchmark."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..paths import UNANSWERABLE_FILE, UNANSWERABLE_LOCK
from .benchmark import DatasetError, assign_split
from .corpus import Corpus

CATEGORIES = ("absent_clause", "contract_not_in_corpus", "out_of_scope")


@dataclass(frozen=True)
class UnanswerableQuery:
    query_id: str
    query: str
    category: str
    target_doc_id: Optional[str]
    rationale: str
    patterns: tuple
    scope: str  # target_doc | corpus

    @property
    def split(self) -> str:
        return assign_split(self.query)


def file_sha256(path: Path = UNANSWERABLE_FILE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_unanswerable(path: Path = UNANSWERABLE_FILE, lock: Optional[Path] = UNANSWERABLE_LOCK,
                      require_lock: bool = True) -> List[UnanswerableQuery]:
    if not path.exists():
        raise DatasetError(f"unanswerable query file not found: {path}")
    if lock is not None and require_lock:
        if not lock.exists():
            raise DatasetError(f"lock file {lock} missing; the unanswerable set must be frozen before use")
        expected = lock.read_text(encoding="utf-8").split()[0]
        actual = file_sha256(path)
        if expected != actual:
            raise DatasetError(f"{path.name} changed after it was frozen (sha256 {actual[:12]} != locked {expected[:12]})")
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("queries") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise DatasetError(f"{path} must contain a non-empty 'queries' list")
    out, seen = [], set()
    for i, r in enumerate(items):
        for key in ("query_id", "query", "label", "category", "rationale", "absence_check"):
            if key not in r:
                raise DatasetError(f"queries[{i}] lacks {key!r}")
        if r["label"] != "unanswerable":
            raise DatasetError(f"queries[{i}].label must be 'unanswerable'")
        if r["category"] not in CATEGORIES:
            raise DatasetError(f"queries[{i}].category {r['category']!r} not in {CATEGORIES}")
        if r["query_id"] in seen:
            raise DatasetError(f"duplicate query_id {r['query_id']}")
        seen.add(r["query_id"])
        chk = r["absence_check"]
        out.append(UnanswerableQuery(r["query_id"], r["query"], r["category"], r.get("target_doc_id"), r["rationale"],
                                     tuple(chk.get("patterns", [])), chk.get("scope", "target_doc")))
    return out


def split_counts(queries: Sequence[UnanswerableQuery]) -> Dict[str, int]:
    c = Counter(q.split for q in queries)
    return {"train": c.get("train", 0), "test": c.get("test", 0), "total": len(queries)}


def absence_report(queries: Sequence[UnanswerableQuery], corpus: Corpus) -> List[Dict]:
    """Re-run each query's recorded keyword check. A hit means the query may be answerable and must be reviewed."""
    rows = []
    for q in queries:
        if q.scope == "target_doc":
            if not q.target_doc_id or q.target_doc_id not in corpus:
                rows.append({"query_id": q.query_id, "ok": False, "reason": "target document not in corpus"})
                continue
            docs = [q.target_doc_id]
        else:
            docs = corpus.doc_ids()
        hits = []
        for pat in q.patterns:
            rx = re.compile(pat, re.I)
            for d in docs:
                for m in rx.finditer(corpus.text(d)):
                    hits.append({"pattern": pat, "doc_id": d, "start": m.start(), "match": m.group(0)})
        rows.append({"query_id": q.query_id, "ok": not hits, "n_hits": len(hits), "hits": hits[:5]})
    return rows
