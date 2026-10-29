"""Repository-relative paths. Every path is derived from the repository root so commands work from any cwd.

`CLAUSELENS_DATA_DIR` may point the raw dataset elsewhere (e.g. a shared drive); nothing else is configurable here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DIR = Path(os.environ.get("CLAUSELENS_DATA_DIR", DATA_DIR / "raw"))
BENCHMARK_FILE = RAW_DIR / "benchmarks" / "cuad.json"
CORPUS_DIR = RAW_DIR / "corpus"
MANIFEST_FILE = DATA_DIR / "dataset_manifest.json"
UNANSWERABLE_FILE = DATA_DIR / "unanswerable_queries.json"
UNANSWERABLE_LOCK = DATA_DIR / "unanswerable_queries.sha256"
LEXICON_FILE = CONFIG_DIR / "lexicons" / "legal_concepts.yaml"
EXPERIMENTS_DIR = ROOT / "experiments"
ARTIFACTS_DIR = ROOT / "artifacts"
EMBED_CACHE_DIR = ARTIFACTS_DIR / "embeddings"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
TEST_LEDGER = EXPERIMENTS_DIR / "test_runs.jsonl"
