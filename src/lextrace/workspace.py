"""Loading helpers shared by the CLI, web app, evaluator and loop (one place that knows the data layout)."""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from .data.benchmark import BenchmarkQuery, load_mini_queries, referenced_documents, split_queries
from .data.corpus import Corpus
from .data.unanswerable import UnanswerableQuery, load_unanswerable


@lru_cache(maxsize=1)
def all_benchmark_queries() -> Tuple[BenchmarkQuery, ...]:
    return tuple(load_mini_queries())


@lru_cache(maxsize=1)
def corpus() -> Corpus:
    """The indexed corpus: exactly the contracts referenced by the 194 mini queries (as the benchmark authors did)."""
    return Corpus.load(referenced_documents(all_benchmark_queries()))


def benchmark_split(split: str) -> List[BenchmarkQuery]:
    return split_queries(all_benchmark_queries(), split)


def unanswerable_split(split: str) -> List[UnanswerableQuery]:
    if split not in ("train", "test"):
        raise ValueError(f"split must be train or test, got {split!r}")
    return [q for q in load_unanswerable() if q.split == split]
