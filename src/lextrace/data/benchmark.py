"""LegalBench-RAG CUAD benchmark: parsing, the authors' mini selection, and the prescribed train/test split.

`mini` and `assign_split` reproduce the prescribed procedures exactly; tests pin their behaviour. Gold spans are
carried on `BenchmarkQuery.gold` for SCORING ONLY: the retrieval pipeline's entry point receives a query string."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..paths import BENCHMARK_FILE


class DatasetError(RuntimeError):
    """The benchmark file is missing or malformed."""


@dataclass(frozen=True)
class GoldSpan:
    doc_id: str  # path relative to the corpus root, e.g. "cuad/Foo.txt"
    start: int
    end: int
    answer: str


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    query: str
    split: str
    gold: tuple  # tuple[GoldSpan, ...]

    @property
    def doc_ids(self) -> List[str]:
        return sorted({g.doc_id for g in self.gold})


def mini(tests: List[Dict[str, Any]], n: int = 194) -> List[Dict[str, Any]]:
    """The benchmark authors' LegalBench-RAG-mini selection (seed = first snippet's file_path)."""
    if len(tests) <= n:
        return tests

    def key(t: Dict[str, Any]) -> float:
        random.seed(t["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def assign_split(query: str) -> str:
    """Prescribed split: SHA-256 of the stripped UTF-8 query; test iff the integer digest mod 10 < 3."""
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"


def validate_tests(tests: Any) -> List[Dict[str, Any]]:
    """Schema check of the raw `tests` list. Raises DatasetError naming the first bad record."""
    if not isinstance(tests, list) or not tests:
        raise DatasetError("'tests' must be a non-empty list")
    for i, t in enumerate(tests):
        if not isinstance(t, dict):
            raise DatasetError(f"tests[{i}] is not an object")
        if not isinstance(t.get("query"), str) or not t["query"].strip():
            raise DatasetError(f"tests[{i}].query missing or empty")
        snippets = t.get("snippets")
        if not isinstance(snippets, list) or not snippets:
            raise DatasetError(f"tests[{i}].snippets missing or empty")
        for j, s in enumerate(snippets):
            where = f"tests[{i}].snippets[{j}]"
            if not isinstance(s, dict) or not isinstance(s.get("file_path"), str):
                raise DatasetError(f"{where}.file_path missing")
            span = s.get("span")
            if (not isinstance(span, (list, tuple)) or len(span) != 2 or not all(isinstance(x, int) for x in span)
                    or not 0 <= span[0] < span[1]):
                raise DatasetError(f"{where}.span must be [start, end] with 0 <= start < end, got {span!r}")
            if not isinstance(s.get("answer"), str):
                raise DatasetError(f"{where}.answer missing")
    return tests


def load_tests(path: Path = BENCHMARK_FILE) -> List[Dict[str, Any]]:
    if not path.exists():
        raise DatasetError(f"benchmark file not found: {path}. Run `python -m lextrace prepare` first.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DatasetError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict) or "tests" not in raw:
        raise DatasetError(f"{path} has no top-level 'tests' key")
    return validate_tests(raw["tests"])


def to_queries(selected: Sequence[Dict[str, Any]]) -> List[BenchmarkQuery]:
    """Assign stable ids by position in the mini order (cuad-mini-000 ...) and the prescribed split."""
    out = []
    for i, t in enumerate(selected):
        gold = tuple(GoldSpan(s["file_path"], int(s["span"][0]), int(s["span"][1]), s["answer"]) for s in t["snippets"])
        out.append(BenchmarkQuery(f"cuad-mini-{i:03d}", t["query"], assign_split(t["query"]), gold))
    return out


def load_mini_queries(path: Path = BENCHMARK_FILE, n: int = 194) -> List[BenchmarkQuery]:
    return to_queries(mini(load_tests(path), n))


def split_queries(queries: Sequence[BenchmarkQuery], split: str) -> List[BenchmarkQuery]:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return [q for q in queries if q.split == split]


def referenced_documents(queries: Sequence[BenchmarkQuery]) -> List[str]:
    return sorted({d for q in queries for d in q.doc_ids})
