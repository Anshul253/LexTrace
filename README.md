# LexTrace — Contract Intelligence

LexTrace answers natural-language questions about contracts by **finding the clause that answers them**. For a
question it returns a ranked list of passages, each with its source contract, its exact character offsets in the
original file (`document[start:end]` is the passage, byte for byte), its score, rank and retrieval method — or it
returns **nothing and says so** when no passage is relevant enough. It is built and measured on the CUAD part of
LegalBench-RAG (the 194-query "mini" selection), runs fully locally on a CPU, and needs no API keys.

It is an evidence-retrieval aid, not legal advice: an empty result does not prove that a contract lacks a clause.

Measured results, the improvement history and error analysis are in **[REPORT.md](REPORT.md)**, not here.

## Features
- **Retrieval methods** (all configurable, all evaluated): BM25 (`rank-bm25`), TF-IDF (scikit-learn), dense
  bi-encoder (sentence-transformers, `all-MiniLM-L6-v2` or `bge-small-en-v1.5`), hybrid fusion (reciprocal-rank or
  weighted min-max), optional cross-encoder reranking, optional concept-lexicon query expansion.
- **Contract routing**: questions that name their contract ("Consider the … Agreement between A and B; …") are routed
  to the best-matching indexed contract by character-n-gram similarity of the title against each contract's opening text.
- **Chunking**: fixed windows (with overlap), paragraph-aware, or clause/heading-aware; every chunk is an offset range
  over the untouched original text.
- **Abstention**: one or two gates over confidence signals (top score, normalised BM25, dense cosine, query-term
  coverage, contract-match similarity), thresholds calibrated on train only under a declared false-abstention budget.
- **Evaluation**: LegalBench-RAG character-level precision/recall@k (k = 1…64), span-hit@k, document-hit@k, MRR,
  abstention in both directions with confidence intervals, signal AUROC, latency; failure diagnosis per query.
- **Self-improvement loop**: one command; diagnoses train failures, proposes changes from a pre-registered rule
  catalogue, accepts/rejects them with a declared objective and guardrails, and logs every iteration.
- **Interfaces**: Streamlit web app (search workspace with evidence cards and a document inspector, evaluation,
  experiments and configuration views) and a Typer CLI.
- **Leakage guards**: test evaluation needs an explicit flag and is recorded in a ledger; thresholds can only be
  calibrated on train rows; the loop only loads train data; gold spans never reach the search entry point.

## Setup
**Prerequisites:** Python **3.9.11** was used for all reported results (Windows 11, CPU only; 3.10/3.11 should work but
were not verified). About 2 GB of disk for PyTorch and models; network access once for packages, the dataset (≈6 MB)
and the embedding models (≈90 MB MiniLM, ≈130 MB bge-small, ≈90 MB cross-encoder; downloaded on first use).

```bash
git clone <this-repo> lextrace && cd lextrace
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
```
No API keys or `.env` are needed (see `.env.example` for optional variables). No LLM is used.

### Dataset
Source: the public Hugging Face mirror [`awinml/legalbench-rag`](https://huggingface.co/datasets/awinml/legalbench-rag)
of the LegalBench-RAG release (Pipitone & Alami, 2024, [arXiv:2408.10343](https://arxiv.org/abs/2408.10343)), pinned
to revision `2b9c248b…`. CUAD contracts are licensed CC BY 4.0 by The Atticus Project; check the dataset card before
redistribution. The corpus is **not** committed; download it with:

```bash
python -m lextrace prepare      # downloads benchmarks/cuad.json + the 29 referenced contracts, verifies SHA-256
python -m lextrace validate     # mini selection, split counts, document[start:end] == answer, unanswerable lock
```
`prepare` is idempotent and writes `data/dataset_manifest.json` (committed) with every file's hash. If a download
fails it prints the recovery steps: check access to huggingface.co (or set `HF_ENDPOINT`), or fetch the files
manually from the dataset page into `data/raw/` keeping their paths (`benchmarks/cuad.json`, `corpus/cuad/*.txt`).

Layout after preparation:
```
data/raw/benchmarks/cuad.json        # full CUAD benchmark (4,042 queries; only 194 are used)
data/raw/corpus/cuad/*.txt           # the 29 contracts referenced by the 194 queries
data/unanswerable_queries.json       # 40 hand-written unanswerable queries (frozen, committed)
data/unanswerable_queries.sha256     # lock: every load verifies this hash
```

## How to run
```bash
streamlit run app.py                                    # web app at http://localhost:8501
python -m lextrace query "Consider the Gas Franchise Agreement between the Town of Vinton and Roanoke Gas Company; What is the term of this agreement?" -k 3
python -m lextrace query "<question>" --json          # full response incl. offsets, signals, routing
python -m lextrace query "<question>" --show-candidates   # on abstention, list the insufficient passages
python -m lextrace build-index --config final         # chunk + index (+ cache embeddings) and report sizes
python -m lextrace evaluate --split train --config final
python -m lextrace analyze --config final             # error-analysis records -> experiments/error_analysis.json
python -m lextrace compare --base final               # BM25 vs TF-IDF vs dense vs hybrid, train only
python -m lextrace improve --start baseline --iterations 14   # self-improvement loop (train only)
python -m lextrace evaluate --split test --config final --confirm-final --note "final reporting"
python -m lextrace tables                             # markdown tables from saved results
python -m lextrace test                               # unit tests;  add --integration for dataset-backed tests
```
CLI exit codes: `0` passages returned · `10` abstained (not an error) · `2` invalid input / refused action · `3`
missing data or bad configuration. `python -m lextrace --help` lists every command.

The web app has four views (left navigation): **Search** (question box, example questions, ranked evidence cards with
offsets and scores, an inspector showing the passage highlighted in its surrounding text, routing/abstention
signals, and an explicit abstention state), **Evaluation** and **Experiments** (read the saved result files and the
loop log), and **Configuration** (active pipeline, gates, dataset provenance). The active configuration is chosen in
the sidebar; only configurations with calibrated thresholds are offered.

## Architecture
```
question ─► validate ─► parse "Consider the <contract>; <question>" ─► ContractRouter (char 3-5-gram TF-IDF)
        ─► first-stage retriever(s) over chunks of the routed contract(s): BM25 | TF-IDF | dense | hybrid (RRF)
        ─► optional cross-encoder rerank ─► drop overlapping passages ─► ranked candidates
        ─► confidence signals ─► abstention gates ─► SearchResponse (passages | abstention reason)
                                                         │
                                CLI · Streamlit app · evaluator · improvement loop
```
| Module (`src/lextrace/`) | Responsibility |
|---|---|
| `data/benchmark.py` | schema validation, the authors' `mini()` selection, the prescribed `assign_split`, query objects |
| `data/corpus.py`, `data/download.py`, `data/validate.py` | text-mode UTF-8 loading, pinned download + checksums, offset validation |
| `data/unanswerable.py` | frozen unanswerable set: schema, hash lock, split, keyword absence checks |
| `chunking/` | fixed / paragraph / clause chunkers returning original-text offsets |
| `retrieval/` | tokeniser + lexicon expansion, BM25 / TF-IDF / dense retrievers, fusion, contract router |
| `pipeline.py` | `Engine.search(query)` — the only inference entry point (receives just the string) |
| `abstention.py` | signals, gate decision, train-only threshold calibration |
| `evaluation/` | metrics, evaluator (ranking and end-to-end views), diagnosis, error records, tables, test ledger |
| `optimization/` | declared objective + acceptance rule, rule-based proposer, loop, method comparison |
| `ui/` | view models (testable without a browser) and the design-system stylesheet; `app.py` is the Streamlit shell |
| `cli.py` | Typer commands |

**Configuration** lives in `configs/` (`baseline.yaml`, `final.yaml`; the loop also writes every candidate to
`experiments/configs/`). YAML files may `extends:` another; unknown keys are rejected. Key options:

| Key | Values | Meaning |
|---|---|---|
| `chunking.strategy / size / overlap` | fixed · paragraph · clause; chars | how contracts are cut into passages |
| `routing.mode / top_docs / query_part` | none · hard · soft; int; full · question | contract routing, and whether the contract name is removed from the retrieval query |
| `retrieval.method` | bm25 · tfidf · dense · hybrid | first-stage retriever |
| `retrieval.hybrid.components / fusion / weights` | list; rrf · weighted | fusion setup |
| `retrieval.dense.model / query_prefix` | HF model id | bi-encoder |
| `retrieval.expansion` | none · lexicon | concept lexicon in `configs/lexicons/legal_concepts.yaml` |
| `retrieval.tokenizer.*`, `bm25_k1`, `bm25_b`, `candidates` | | lexical settings, ranked-list length |
| `rerank.enabled / model / top_n / weight` | | cross-encoder reranking |
| `postprocess.dedupe_overlaps / expand_chars` | | overlap removal, passage widening |
| `abstention.gates[] {signal, threshold}` / `max_false_abstention` | `null` = calibrate on train | abstention |
| `evaluation.ks / primary_k` | | metric cutoffs |

## Reproducing the reported results
From a fresh clone with the dataset prepared (times on a 2020s laptop CPU):

| Step | Command | Time |
|---|---|---|
| validate data | `python -m lextrace validate` | ~5 s |
| baseline train metrics | `python -m lextrace evaluate --split train --config baseline` | ~10 s |
| improvement loop → `configs/final.yaml`, `experiments/experiment_log.jsonl` | `python -m lextrace improve --start baseline --iterations 14` | ~15–25 min (first run downloads models) |
| final train metrics | `python -m lextrace evaluate --split train --config final` | ~1 min |
| method comparison | `python -m lextrace compare --base final` | ~5 min |
| error analysis | `python -m lextrace analyze --config final` | ~1 min |
| held-out test (final reporting only) | `python -m lextrace evaluate --split test --config baseline --confirm-final` and `… --config final --confirm-final` | ~1 min |
| tables for the report | `python -m lextrace tables` | instant |

`baseline.yaml` has uncalibrated thresholds (`null`); `evaluate --split train --config baseline --save-calibrated`
calibrates them on train and writes `configs/baseline_calibrated.yaml`, which is the file used for the baseline's
test evaluation (test evaluation refuses uncalibrated configs, so it can never fit anything on test). The exact
command sequence used for the report is in REPORT.md §5. Everything is deterministic on CPU; dense scores can differ
in the last decimals across hardware/library versions.

## Testing
```bash
python -m pytest -q                     # unit tests: synthetic fixtures, no download, no model (~10 s)
python -m pytest -q -m integration      # dataset-backed checks (needs `prepare`)
```
Unit tests cover the exact mini selection and split (pinned values and parity with the authors' function), schema and
malformed records, text-mode offsets, offset validation, chunk offset mapping for every strategy, ranking, routing,
invalid queries, abstention and calibration, character-level metrics against hand-computed values and a port of the
authors' formula, unanswerable-set integrity and tamper detection, train/test isolation guards, the acceptance rule,
proposer, a full fixture run of the loop, configuration loading, the CLI and the web view models.

## Assumptions
| # | Assumption | Why |
|---|---|---|
| A1 | The paper's character metrics are implemented as in the authors' code (`QAResult`): precision denominator includes text from wrong documents; empty retrieval scores 0; macro mean over queries. We take the union of returned characters, identical to the authors' formula when passages don't overlap (the pipeline removes overlaps; a parity test checks it). | The authors' code is the unambiguous reference; the union avoids double-counting overlaps. |
| A2 | Retrieval tables use the **ranking view** (before the gate); an **end-to-end** recall (abstained = 0) is also reported. | Separates retrieval quality from the abstention decision. |
| A3 | Primary cutoff k = 4; k ∈ {1,2,4,8,16,32,64} reported. | A reviewer reads a handful of passages; the grid matches the paper. |
| A4 | Index = the 29 contracts referenced by the 194 mini queries (all splits), as the authors did. Contract *selection* uses document ids only, never gold spans. | Stated by the benchmark setup; indexing train-only contracts would make test questions unanswerable by construction. |
| A5 | Questions name their contract in the benchmark's "Consider the …;" form, and routing uses it. Questions without it search all contracts. | Matches the benchmark's question format; documented limitation for real users. |
| A6 | Unanswerable queries: 40, in three categories (clause absent from a named indexed contract; contract not in the corpus; question outside what a contract records). Absence verified by recorded keyword checks with zero hits plus reading the matched context; the two shortest contracts were read in full. | A paraphrased clause could still be missed — see REPORT limitations. |
| A7 | Abstention thresholds are calibrated on **train** only: at most 10% of train answerable queries refused; among such thresholds the one abstaining on most train unanswerable queries. | Train unanswerables are allowed for tuning; test ones are never used. |
| A8 | No LLM. | Retrieval-first assignment, confidential contracts, reproducibility; see REPORT §4. |
| A9 | Python 3.9.11 (the interpreter available on the development machine). | Code avoids 3.10+ syntax. |
| A10 | The concept lexicon is written from the public CUAD clause categories before any experiment ran. | It encodes domain vocabulary, not benchmark answers. |

## Limitations and troubleshooting
- Routing depends on the user naming the contract; a wrong or missing name searches the wrong or all contracts.
- Only plain-text contracts; no PDFs/OCR, tables, or resolution of cross-references and definitions.
- Abstention is imperfect in both directions (numbers in REPORT.md); never read an empty result as "no such clause".
- *"benchmark file not found"* → run `python -m lextrace prepare`. *"lock file … changed"* → the unanswerable set
  was edited after it was frozen; restore it with `git checkout data/unanswerable_queries.json`.
- *Model download errors* → check access to huggingface.co, set `HF_HOME` to a writable cache, or use a config
  without dense retrieval (e.g. a BM25 config).
- *"uncalibrated thresholds"* → evaluate that config on train with `--save-calibrated` first.
- First web-app load builds embeddings (cached under `artifacts/embeddings/`); later loads are fast.

## AI tool usage
Claude Code (Anthropic) was used as an AI coding assistant throughout the development of this project. It was used
to research technical approaches, understand relevant documentation, explore retrieval strategies and evaluate
architectural choices, and it also generated most of the implementation code, the tests, the web interface and first
drafts of this README and REPORT.md, ran the experiments, and helped debug failures. The overall approach,
implementation decisions and evaluation methodology were reviewed throughout the process, and the generated code and
experimental outputs were examined and validated. The project was developed through an iterative process of
understanding requirements, implementing components, testing functionality and refining the system based on observed
results. No AI model is used at runtime in the final retrieval pipeline. Reported evaluation results are based on the
saved experiment outputs in the repository.
